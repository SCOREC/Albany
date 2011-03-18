/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef PISDWDF_HPP
#define PISDWDF_HPP

#include "Phalanx_ConfigDefs.hpp"
#include "Phalanx_Evaluator_WithBaseImpl.hpp"
#include "Phalanx_Evaluator_Derived.hpp"
#include "Phalanx_MDField.hpp"

#include "Sacado_Fad_SLFad.hpp"

/** \brief Nonlinear Elasticity Energy Potential

    This evaluator computes a energy density for nonlinear elastic material

*/
namespace LCM {

template<typename EvalT, typename Traits>
class PisdWdF : public PHX::EvaluatorWithBaseImpl<Traits>,
		public PHX::EvaluatorDerived<EvalT, Traits>  {

public:

  PisdWdF(const Teuchos::ParameterList& p);

  void postRegistrationSetup(typename Traits::SetupData d,
			     PHX::FieldManager<Traits>& vm);

  void evaluateFields(typename Traits::EvalData d);

private:

  typedef typename EvalT::ScalarT ScalarT;
  typedef typename EvalT::MeshScalarT MeshScalarT;

  // Stuff needed for AD of energy functional
  //typedef typename Sacado::Fad::SLFad<ScalarT, 9> EnergyFadType;
  typedef typename Sacado::Fad::SLFad<ScalarT, 9> EnergyFadType;
  EnergyFadType computeEnergy(ScalarT& kappa, ScalarT& mu,
                          Intrepid::FieldContainer<EnergyFadType>& W);


  // Input:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim,Dim> defgrad;
  PHX::MDField<ScalarT,Cell,QuadPoint> elasticModulus;
  PHX::MDField<ScalarT,Cell,QuadPoint> poissonsRatio;

  unsigned int numQPs;
  unsigned int numDims;

  // Output:
  PHX::MDField<ScalarT,Cell,QuadPoint,Dim> P;
};
}

#endif
