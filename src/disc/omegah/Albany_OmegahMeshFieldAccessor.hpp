#ifndef ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP
#define ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP

#include "Albany_AbstractMeshFieldAccessor.hpp"
#include "Albany_DiscretizationUtils.hpp" // Temporary, for NotYetIMplemented exception

#include "Omega_h_mesh.hpp"

#include <map>
#include <vector>

namespace Albany {

class OmegahMeshFieldAccessor : public AbstractMeshFieldAccessor
{
public:
  OmegahMeshFieldAccessor (const Teuchos::RCP<Omega_h::Mesh>& mesh);
  ~OmegahMeshFieldAccessor () = default;

  void addStateStruct(const Teuchos::RCP<StateStruct>& st) override;

  void createStateArrays (const WorksetArray<int>& worksets_sizes) override;
  void transferNodeStatesToElemStates () override;

  // TODO: move this in the base class?
  void addFieldOnMesh (const std::string& name,
                       const int entityDim,
                       const int numComps);

  void setFieldOnMesh (const std::string& name,
                       const int entityDim,
                       const Teuchos::RCP<const Thyra_MultiVector>& mv);

  // Read from mesh methods
  void fillSolnVector (Thyra_Vector&        /* soln */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnVector");
  }

  void fillVector (Thyra_Vector&        field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void fillSolnMultiVector (Thyra_MultiVector&   /* soln */,
                            const dof_mgr_ptr_t& /* sol_dof_mgr */,
                            const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnMultiVector");
  }

  void fillSolnSensitivity(Thyra_MultiVector&                    /* dxdp */,
                           const Teuchos::RCP<const DOFManager>& /* solution_dof_mgr */,
                           const bool                            /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::fillSolnSensitivity");
  }

  // Write to mesh methods
  void saveVector (const Thyra_Vector&  field_vector,
                   const std::string&   field_name,
                   const dof_mgr_ptr_t& field_dof_mgr,
                   const bool           overlapped) override;

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const Thyra_Vector&  /* soln_dot */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveSolnVector (const Thyra_Vector&  /* soln */,
                       const mv_ptr_t&      /* soln_dxdp */,
                       const Thyra_Vector&  /* soln_dot */,
                       const Thyra_Vector&  /* soln_dotdot */,
                       const dof_mgr_ptr_t& /* sol_dof_mgr */,
                       const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnVector");
  }

  void saveResVector (const Thyra_Vector&  /* res */,
                      const dof_mgr_ptr_t& /* dof_mgr */,
                      const bool           /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveResVector");
  }

  void saveSolnMultiVector (const Thyra_MultiVector& /* soln */,
                            const mv_ptr_t&          /* soln_dxdp */,
                            const dof_mgr_ptr_t&     /* node_vs */,
                            const bool               /* overlapped */) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,NotYetImplemented,"OmegahMeshFieldAccessor::saveSolnMultiVector");
  }

  void setSolutionFieldsMetadata (const int neq) override;

  // To be called after adaptation, where the stored tags are no longer valid and need to be reset
  void reset_mesh_tags ();

  // Names of the fields stored on mesh vertices. Omega_h drops any tag that is not
  // registered in AdaptOpts::xfer_opts, so these must be registered for interpolation
  // before adapting, or they would come back zero-filled on the new mesh.
  std::vector<std::string> get_nodal_field_names () const;

  // DIAGNOSTIC: round-trip check for the packed multi-component solution tag.
  // saveVector packs a Thyra vector into the mesh tag; fillVector reads it back after
  // the mesh (and the dof manager) have been rebuilt by adaptation. For a vertex that
  // SURVIVED the adaptation the two must agree exactly.
  //
  // Identifying survivors is the hard part. The 'adapt_probe_id' marker alone is NOT
  // enough: it is transferred with LINEAR_INTERP, so a vertex splitting an edge whose
  // endpoints are k-1 and k+1 gets the exact integer k, colliding with the survivor
  // that legitimately carries k. We therefore ALSO record the vertex coordinates and
  // require both to match: a survivor keeps its coordinates, whereas a split vertex
  // sits at the midpoint of the edge it split.
  void probe_record (const std::string& field_name, int dim, int ncomps);
  void probe_compare (const std::string& field_name, int dim, int ncomps,
                      const Teuchos::ArrayRCP<const ST>& thyra_data,
                      const DOFManager& dof_mgr);

protected:
  Teuchos::RCP<Omega_h::Mesh>   m_mesh;

  // Data recorded for one vertex at save time (pre-adapt)
  struct ProbeEntry {
    std::vector<ST> vals;   // the ncomps packed tag components
    std::vector<ST> coords; // vertex coordinates, used to disambiguate survivors
  };
  // global vertex id (pre-adapt) -> recorded data
  std::map<Omega_h::GO,ProbeEntry> m_probe_vals;

  struct TagHandle {
    Omega_h::Write<ST> array;
    int                ent_dim;
    int                ncomps;
  };
  std::map<std::string,TagHandle> m_tags;
};

} // namespace Albany

#endif // ALBANY_OMEGAH_MESH_FIELD_ACCESSOR_HPP
