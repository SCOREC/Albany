//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_FLUX_DIVERGENCE_RESIDUAL_HPP
#define LANDICE_FLUX_DIVERGENCE_RESIDUAL_HPP

#include "PHAL_SeparableScatterScalarResponse.hpp"
#include "Shards_CellTopology.hpp"

namespace LandIce {
/**
 * \brief Response Description
 */
  template<typename EvalT, typename Traits>
  class FluxDivergenceResidual : public PHX::EvaluatorWithBaseImpl<Traits>,
        public PHX::EvaluatorDerived<EvalT, Traits>
  {
  public:
    typedef typename EvalT::ScalarT ScalarT;
    typedef typename EvalT::MeshScalarT MeshScalarT;
    typedef typename EvalT::ParamScalarT ParamScalarT;

    FluxDivergenceResidual(Teuchos::ParameterList& p,
       const Teuchos::RCP<Albany::Layouts>& dl);

    void postRegistrationSetup(typename Traits::SetupData d,
             PHX::FieldManager<Traits>& vm);

    void evaluateFields(typename Traits::EvalData d);

  private:

    Teuchos::RCP<shards::CellTopology> cellType;
    std::vector<std::vector<int> >  sideNodes;
    std::string sideName;
    
    int numCells;
    int numNodes;
    int sideDim;
    int numSideNodes;
    bool upwindStabilization;

    PHX::MDField<const MeshScalarT, Cell, Node, Dim> coords;
    PHX::MDField<const ParamScalarT> H;
    PHX::MDField<const ScalarT> vel;
    PHX::MDField<const ScalarT> flux_div;

    PHX::MDField<ScalarT> residual;
  };

} // Namespace LandIce

#endif // LANDICE_RESPONSE_SURFACE_VELOCITY_MISMATCH_HPP
