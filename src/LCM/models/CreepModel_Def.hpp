//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <MiniTensor.h>
#include "LocalNonlinearSolver.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Teuchos_TestForException.hpp"

namespace LCM {

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
CreepModel<EvalT, Traits>::CreepModel(
    Teuchos::ParameterList*              p,
    const Teuchos::RCP<Albany::Layouts>& dl)
    : LCM::ConstitutiveModel<EvalT, Traits>(p, dl),
      creep_initial_guess_(p->get<RealType>("Initial Creep Guess", 1.1e-4)),

      // sat_mod_(p->get<RealType>("Saturation Modulus", 0.0)),
      // sat_exp_(p->get<RealType>("Saturation Exponent", 0.0)),

      // below is what we called C_2 in the functions
      strain_rate_expo_(p->get<RealType>("Strain Rate Exponent", 1.0)),
      // below is what we called A in the functions
      relaxation_para_(
          p->get<RealType>("Relaxation Parameter of Material_A", 0.1)),
      // below is what we called Q/R in the functions, users can give them
      // values here
      activation_para_(
          p->get<RealType>("Activation Parameter of Material_Q/R", 500.0)),
      // Maximum allowable attempts for the return mapping algorithm
      max_return_map_count(p->get<int>("Max Return Mapping Attempts", 100)),
      // Tolerance on the return mapping algorithm
      return_map_tolerance(p->get<RealType>("Return Mapping Tolerance", 1.0e-10))

{
  // retrive appropriate field name strings
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string eqps_string   = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string      = (*field_name_map_)["F"];
  std::string J_string      = (*field_name_map_)["J"];

  // define the dependent fields
  this->dep_field_map_.insert(std::make_pair(F_string, dl->qp_tensor));
  this->dep_field_map_.insert(std::make_pair(J_string, dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Poissons Ratio", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Elastic Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Yield Strength", dl->qp_scalar));
  this->dep_field_map_.insert(
      std::make_pair("Hardening Modulus", dl->qp_scalar));
  this->dep_field_map_.insert(std::make_pair("Delta Time", dl->workset_scalar));

  // define the evaluated fields
  this->eval_field_map_.insert(std::make_pair(cauchy_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(Fp_string, dl->qp_tensor));
  this->eval_field_map_.insert(std::make_pair(eqps_string, dl->qp_scalar));
  if (have_temperature_) {
    this->eval_field_map_.insert(std::make_pair(source_string, dl->qp_scalar));
  }

  // define the state variables
  //
  // stress
  this->num_state_variables_++;
  this->state_var_names_.push_back(cauchy_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(false);
  this->state_var_output_flags_.push_back(
      p->get<bool>("Output Cauchy Stress", false));
  //
  // Fp
  this->num_state_variables_++;
  this->state_var_names_.push_back(Fp_string);
  this->state_var_layouts_.push_back(dl->qp_tensor);
  this->state_var_init_types_.push_back("identity");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output Fp", false));
  //
  // eqps
  this->num_state_variables_++;
  this->state_var_names_.push_back(eqps_string);
  this->state_var_layouts_.push_back(dl->qp_scalar);
  this->state_var_init_types_.push_back("scalar");
  this->state_var_init_values_.push_back(0.0);
  this->state_var_old_state_flags_.push_back(true);
  this->state_var_output_flags_.push_back(p->get<bool>("Output eqps", false));
  //
  // mechanical source
  if (have_temperature_) {
    this->num_state_variables_++;
    this->state_var_names_.push_back(source_string);
    this->state_var_layouts_.push_back(dl->qp_scalar);
    this->state_var_init_types_.push_back("scalar");
    this->state_var_init_values_.push_back(0.0);
    this->state_var_old_state_flags_.push_back(false);
    this->state_var_output_flags_.push_back(
        p->get<bool>("Output Mechanical Source", false));
  }
}

// void creepprint(double x)
//{
//  fprintf(stderr, "%a\n", x);
//}

// void creepprint(FadType const& x)
//{
//  fprintf(stderr, "%a [", x.val());
//  for (int i = 0; i < x.size(); ++i)
//    fprintf(stderr, " %a", x.dx(i));
//  fprintf(stderr, "\n");
//}

//------------------------------------------------------------------------------
template <typename EvalT, typename Traits>
void
CreepModel<EvalT, Traits>::computeState(
    typename Traits::EvalData workset,
    DepFieldMap               dep_fields,
    FieldMap                  eval_fields)
{
  static int  times_called  = 0;
  std::string cauchy_string = (*field_name_map_)["Cauchy_Stress"];
  std::string Fp_string     = (*field_name_map_)["Fp"];
  std::string eqps_string   = (*field_name_map_)["eqps"];
  std::string source_string = (*field_name_map_)["Mechanical_Source"];
  std::string F_string      = (*field_name_map_)["F"];
  std::string J_string      = (*field_name_map_)["J"];

  // extract dependent MDFields
  auto def_grad          = *dep_fields[F_string];
  auto J                 = *dep_fields[J_string];
  auto poissons_ratio    = *dep_fields["Poissons Ratio"];
  auto elastic_modulus   = *dep_fields["Elastic Modulus"];
  auto yield_strength    = *dep_fields["Yield Strength"];
  auto hardening_modulus = *dep_fields["Hardening Modulus"];
  auto delta_time        = *dep_fields["Delta Time"];

  ScalarT dt = delta_time(0);

  // extract evaluated MDFields
  auto                  stress = *eval_fields[cauchy_string];
  auto                  Fp     = *eval_fields[Fp_string];
  auto                  eqps   = *eval_fields[eqps_string];
  PHX::MDField<ScalarT> source;
  if (have_temperature_) { source = *eval_fields[source_string]; }

  // get State Variables
  Albany::MDArray Fpold   = (*workset.stateArrayPtr)[Fp_string + "_old"];
  Albany::MDArray eqpsold = (*workset.stateArrayPtr)[eqps_string + "_old"];

  ScalarT kappa, mu, mubar, K, Y;
  // new parameters introduced here for being the temperature dependent, they
  // are the last two listed below
  ScalarT Jm23, p, dgam_creep, dgam_plastic, f, smag,
      temp_adj_relaxation_para_;
  ScalarT sq23(std::sqrt(2. / 3.));

  minitensor::Tensor<ScalarT> F(num_dims_), 
                              be(num_dims_), 
                              s(num_dims_),
                              sigma(num_dims_),
                              N(num_dims_), 
                              A(num_dims_), 
                              expA(num_dims_),
                              Fpnew(num_dims_),
                              Fpn(num_dims_), 
                              Fpinv(num_dims_),
                              Cpinv(num_dims_);
  minitensor::Tensor<ScalarT> I(minitensor::eye<ScalarT>(num_dims_));

  for (int cell(0); cell < workset.numCells; ++cell) {
    for (int pt(0); pt < num_pts_; ++pt) {
      kappa = elastic_modulus(cell, pt) /
              (3. * (1. - 2. * poissons_ratio(cell, pt)));
      mu   = elastic_modulus(cell, pt) / (2. * (1. + poissons_ratio(cell, pt)));
      K    = hardening_modulus(cell, pt);
      Y    = yield_strength(cell, pt);
      Jm23 = std::pow(J(cell, pt), -2. / 3.);

      // ----------------------------  temperature dependent coefficient

      // the effective 'B' we had before in the previous models, with mu
      if (have_temperature_) {
        temp_adj_relaxation_para_ =
            relaxation_para_ *
            std::exp(-activation_para_ / temperature_(cell, pt));
      } else {
        temp_adj_relaxation_para_ =
            relaxation_para_ * std::exp(-activation_para_ / 303.0);
      }

      // fill local tensors
      F.fill(def_grad, cell, pt, 0, 0);

      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fpn(i, j) = ScalarT(Fpold(cell, pt, i, j));
        }
      }

      // compute trial state
      Fpinv = minitensor::inverse(Fpn);
      Cpinv = Fpinv * minitensor::transpose(Fpinv);
      be    = Jm23 * F * Cpinv * minitensor::transpose(F);

      s = mu * minitensor::dev(be);
      mubar = minitensor::trace(be) * mu / (num_dims_);

      // plastic direction
      N = s / minitensor::norm(s);

      // check if creep is large enough to calculate
      ScalarT d = 2.0 * mubar * dt * temp_adj_relaxation_para_ * std::pow( smag, strain_rate_expo_-1.0);
      dgam_creep = 0.0;
      if (d > 1.0E-12) 
      {
        smag = minitensor::norm(s);

        ScalarT eta_old = 1.0e-7;
        ScalarT eta_new = 0.0;

        ScalarT c_inv = 1.0/strain_rate_expo_;

        bool converged = false;
        int  count     = 0;
        while( !converged)
        {
          ScalarT d_bar = std::pow( d, c_inv);

          ScalarT f  = std::pow(eta_old, c_inv) + d_bar * (eta_old -1.0);
          ScalarT df = 1.0/( strain_rate_expo_ * std::pow( eta_old, 1.0-c_inv)) + d_bar;

          eta_new = eta_old - f/df;

          converged = ( std::abs( eta_new - eta_old) < return_map_tolerance);

          TEUCHOS_TEST_FOR_EXCEPTION(
            count == max_return_map_count,
            std::runtime_error,
            "\n" << "Could not converge for creep return map!\n"
                 << "std::abs(eta_new - eta_old) = "
                   <<  std::abs(eta_new - eta_old) << "\n"
                 << "d     = " << d << "\n" 
                 << "d_bar = " << d_bar << "\n" 
                 << "f     = " << f << "\n" 
                 << "df    = " << df << "\n" 
                 << std::endl);
        }
        ScalarT eta = eta_new;

        // update s to include creep correction
        ScalarT smag_old = smag;
        smag = (1.0-eta) * smag_old;

        // calculate delta gamma creep with new stress
        dgam_creep = eta * smag_old / (2.0 * mubar);

        // Correct for the amount of stress reduced through creep
        s -= 2.0 * mubar * dgam_creep * N;
      }  

      ScalarT smag_cr = minitensor::norm(s);
      f = smag_cr - sq23 * (Y + K * eqpsold(cell, pt));
      if (f > 0.0)  // Material is yielding...
      {
        ScalarT xi = 2.0 * mubar * dt * temp_adj_relaxation_para_ * strain_rate_expo_
                  * std::pow( smag_cr, strain_rate_expo_-1.0);
        ScalarT xi_1 = xi + 1.0;

        ScalarT top = f * xi_1;
        ScalarT bot = 2.0 * (mubar + K * xi_1/3.0);

        dgam_plastic = top/bot;

        // update s
        s -= 2.0 * mubar * dgam_plastic * N + f * N -
             2. * mubar * (1. + K / (3. * mubar)) * dgam_plastic * N;

        // update eqps
        eqps(cell, pt) = eqpsold(cell, pt) + sq23 * dgam_plastic;
      }
      else // No yielding
      {
        dgam_plastic = 0.0;
        eqps(cell, pt) = eqpsold(cell, pt);
      }

      // exponential map to get Fpnew
      A     = (dgam_plastic+dgam_creep) * N;
      expA           = minitensor::exp(A);
      Fpnew = expA * Fpn;
      for (int i(0); i < num_dims_; ++i) {
        for (int j(0); j < num_dims_; ++j) {
          Fp(cell, pt, i, j) = Fpnew(i, j);
        }
      }

      p = 0.5 * kappa * (J(cell, pt) - 1. / (J(cell, pt)));

      // compute stress
      sigma = p * I + s / J(cell, pt);
      for (int i(0); i < num_dims_; ++i) 
      {
        for (int j(0); j < num_dims_; ++j) 
        {
          stress(cell, pt, i, j) = sigma(i, j);
        }
      }
    } //For each qp 
  } // For each cell

  if (have_temperature_) {
    for (int cell(0); cell < workset.numCells; ++cell) {
      for (int pt(0); pt < num_pts_; ++pt) {
        ScalarT three_kappa =
            elastic_modulus(cell, pt) / (1.0 - 2.0 * poissons_ratio(cell, pt));
        F.fill(def_grad, cell, pt, 0, 0);
        ScalarT J = minitensor::det(F);
        sigma.fill(stress, cell, pt, 0, 0);
        sigma -= three_kappa * expansion_coeff_ * (1.0 + 1.0 / (J * J)) *
                 (temperature_(cell, pt) - ref_temperature_) * I;
        for (int i = 0; i < num_dims_; ++i) {
          for (int j = 0; j < num_dims_; ++j) {
            stress(cell, pt, i, j) = sigma(i, j);
          }
        }
      }
    }
  }
}

}  // namespace LCM
