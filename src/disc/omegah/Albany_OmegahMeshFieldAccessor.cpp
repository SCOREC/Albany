#include "Albany_OmegahMeshFieldAccessor.hpp"
#include "Albany_ThyraUtils.hpp"
#include "Albany_OmegahUtils.hpp"
#include "OmegahGhost.hpp"
#include <Omega_h_map.hpp>

#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>

namespace Albany {

OmegahMeshFieldAccessor::
OmegahMeshFieldAccessor (const Teuchos::RCP<Omega_h::Mesh>& mesh)
 : m_mesh (mesh)
{
  // Nothing to do here
}

void OmegahMeshFieldAccessor::
addFieldOnMesh (const std::string& name,
                const int entityDim,
                const int numComps)
{
  Omega_h::Write<ST> f(m_mesh->nents(entityDim)*numComps,name);
  if (m_mesh->has_tag(entityDim,name)) {
    auto tag = m_mesh->get_tag<ST>(entityDim,name);
    TEUCHOS_TEST_FOR_EXCEPTION (tag==nullptr, std::logic_error,
        "Error! Tag '" + name + "' already exists on entity dim " + std::to_string(entityDim) +
        " but has a non-real (non-double) type. Cannot use it as an Albany field.\n");
    TEUCHOS_TEST_FOR_EXCEPTION (tag->ncomps()!=numComps, std::logic_error,
        "Error! Attempt to re-define tag with different number of components.\n"
        " - tag name: " + name + "\n"
        " - tag ncomps: " + std::to_string(tag->ncomps()) + "\n"
        " - new ncomps: " + std::to_string(numComps) + "\n");

    auto const old_size = tag->array().size();
    auto const new_size = f.size();
    TEUCHOS_TEST_FOR_EXCEPTION (old_size != new_size, std::logic_error,
        "Error! Attempt to re-define tag with incompatible storage size.\n"
        "The mesh entity count appears to have changed since the tag was created.\n"
        " - tag name: " + name + "\n"
        " - entity dimension: " + std::to_string(entityDim) + "\n"
        " - tag array size: " + std::to_string(old_size) + "\n"
        " - expected array size: " + std::to_string(new_size) + "\n"
        " - tag ncomps: " + std::to_string(tag->ncomps()) + "\n"
        " - current numComps: " + std::to_string(numComps) + "\n");
    // Copy existing data, then remove so we can re-add with OUR (writable) array
    Omega_h::copy_into(tag->array(),f);
    m_mesh->remove_tag(entityDim,name);
  }
  m_mesh->add_tag<ST>(entityDim,name,numComps,f,false);
  m_tags[name].array = f;
  m_tags[name].ncomps = numComps;
  m_tags[name].ent_dim = entityDim;
}

void OmegahMeshFieldAccessor::
setFieldOnMesh (const std::string& name,
                const int entityDim,
                const Teuchos::RCP<const Thyra_MultiVector>& mv)
{
  auto tag = m_mesh->get_tag<ST>(entityDim,name);
  TEUCHOS_TEST_FOR_EXCEPTION (tag->ncomps()!=mv->domain()->dim(), std::logic_error,
      "Error! Cannot copy MV on mesh tag, since the number of vecs does not match the tag ncomps.\n"
      "  - tag name: " + name + "\n"
      "  - tag ncomps: " << tag->ncomps() << "\n"
      "  - MV num vecs: " << mv->domain()->dim() << "\n");

  // Create 1d view of input MV
  auto dev_mv = getDeviceData(mv);
  int ncmps = dev_mv.extent(1);
  int mv_nents = dev_mv.extent(0);
  int tag_nents = Omega_h::divide_no_remainder(tag->array().size(), tag->ncomps());

  TEUCHOS_TEST_FOR_EXCEPTION (tag_nents != m_mesh->nents(entityDim), std::runtime_error,
      "Error! Something is amiss with the registered tag size.\n"
      "  - tag name: " + name + "\n"
      "  - entity dim: " << entityDim << "\n"
      "  - tag num ents: " << tag_nents << "\n"
      "  - mesh num ents: " << m_mesh->nents(entityDim) << "\n");

  // The MV contains only owned entities. In Omega_h's GHOSTED parting, owned entities
  // are NOT guaranteed to be at positions 0..nowned-1 in the entity arrays, so we must
  // use collect_marked to find the actual Omega_h positions for owned entities.
  // The caller is responsible for syncing ghost data afterwards via sync_tag.
  TEUCHOS_TEST_FOR_EXCEPTION (mv_nents > tag_nents, std::logic_error,
      "Error! Unexpected number of entities in input MV.\n"
      "  - tag name: " + name + "\n"
      "  - entity dim: " << entityDim << "\n"
      "  - num owned+ghosted ents: " << tag_nents << "\n"
      "  - MV num ents: " << mv_nents << "\n");

  // Copy into tag. WARNING: tags have entity id striding slower, while the input mv makes
  // entity id stride faster (it's a 2d view with layout left).
  // IMPORTANT: In Omega_h's GHOSTED parting, owned entities are NOT guaranteed to be at
  // positions 0..nowned-1. We must map Albany LID k -> Omega_h position via collect_marked.
  auto owned_positions = Omega_h::collect_marked(m_mesh->owned(entityDim));
  TEUCHOS_TEST_FOR_EXCEPTION (owned_positions.size() != mv_nents, std::logic_error,
      "Error! Mismatch between Albany MV size and Omega_h owned entity count.\n"
      "  - field name: " + name + "\n"
      "  - mv_nents: " << mv_nents << "\n"
      "  - nowned: " << owned_positions.size() << "\n");

  Kokkos::RangePolicy<> policy(0,mv_nents*ncmps);
  auto tag_view = m_tags.at(name).array.view();
  auto lambda = KOKKOS_LAMBDA(int idx) {
    int ient = idx % mv_nents;  // Albany LID
    int icmp = idx / mv_nents;
    int omegah_pos = owned_positions[ient];  // actual Omega_h entity position
    tag_view (omegah_pos*ncmps + icmp) = dev_mv(ient,icmp);
  };
  Kokkos::parallel_for(policy,lambda);
  Kokkos::fence();
}

void OmegahMeshFieldAccessor::
addStateStruct(const Teuchos::RCP<StateStruct>& st)
{
  auto product = [](const auto& vec, int start) {
    return std::accumulate(vec.begin()+start, vec.end(), 1, std::multiplies<int>());
  };

  auto get_ent_dim_and_ncomp = [&] (const StateStruct& st) {
    std::pair<int,int> dim_ncomp;
    switch (st.stateType()) {
      case StateStruct::NodeState:
        dim_ncomp.first  =  0;
        dim_ncomp.second = product(st.dim,st.entity==StateStruct::NodalData ? 1 : 2);
        break;
      case StateStruct::ElemState:
        dim_ncomp.first  = m_mesh->dim();
        dim_ncomp.second = product(st.dim,1);
        break;
      default:
        throw std::runtime_error(
            "Error! Invalid/unsupported state type.\n"
            "  - state name: " + st.name + "\n");
    }
    return dim_ncomp;
  };

  // nodal/nodal_parameter states  will be warranted a dof mgr later,
  // while elem_sis are states that can be processed by LoadStateField
  // and LoadSideSetStateField evaluators
  switch(st->entity) {
    case StateStruct::NodalDistParameter:
      nodal_parameter_sis.push_back(st);
      [[fallthrough]];
    case StateStruct::NodalDataToElemNode:
      nodal_sis.push_back(st);
      break;
    case StateStruct::ElemData:   [[fallthrough]];
    case StateStruct::ElemNode:   [[fallthrough]];
    case StateStruct::QuadPoint:
      elem_sis.push_back(st);
      break;
    default:
      throw std::runtime_error("Error! Unrecognized/unsupported state entity type.\n");
  }

  auto dim_ncomp = get_ent_dim_and_ncomp(*st);
  int ent_dim = dim_ncomp.first;
  int ncomp = dim_ncomp.second;
  if (ent_dim==-1) {
    if (ncomp==1) {
      mesh_scalar_states.emplace(st->name,st->initValue);
    } else {
      mesh_vector_states[st->name].resize(ncomp,st->initValue);
    }
  } else {
    addFieldOnMesh(st->name,ent_dim,ncomp);
  }

  if (st->layered) {
    // Need to also add the global vector state for the normalized layers coords
    auto nlayers = st->dim.back();
    mesh_vector_states[st->name+"_NLC"].resize(nlayers);
  }
}

void OmegahMeshFieldAccessor::createStateArrays (const WorksetArray<int>& worksets_sizes)
{
  // Elem states
  int num_ws = worksets_sizes.size();
  elemStateArrays.clear();
  elemStateArrays.resize(worksets_sizes.size());
  for (const auto& st : elem_sis) {
    auto data = m_tags.at(st->name).array.data();
    auto dim = st->dim;
    int stride = 1;
    dim[0] = 1; // We don't use the extent of the elem tag to compute stride
    for (auto d : dim) stride *= d;

    for (int ws=0; ws<num_ws; ++ws) {
      int num_elems = worksets_sizes[ws];
      switch (dim.size()) {
        case 1:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems); break;
        case 2:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1]); break;
        case 3:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1],dim[2]); break;
        case 4:
          elemStateArrays[ws][st->name].reset_from_dev_ptr(data,num_elems,dim[1],dim[2],dim[3]); break;
        default:
          throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
      }
      data += worksets_sizes[ws]*stride;
    }
  }

  // Nodal states
  // NOTE: nodal states have just 1 workset
  nodeStateArrays.clear();
  nodeStateArrays.resize(1);
  int num_nodes = m_mesh->nverts();
  for (const auto& st : nodal_sis) {
    auto data = m_tags.at(st->name).array.data();
    auto dim = st->dim;
    if (st->entity != StateStruct::NodalData) {
      // Add an elem state array, which the SaveStateField/SaveSideSetStateField evaluators will use
      for (int ws=0; ws<worksets_sizes.size(); ++ws) {
        auto& state = elemStateArrays[ws][st->name];
        switch (dim.size()) {
          case 2:
            state.resize(st->name,worksets_sizes[ws],dim[1]); break;
          case 3:
            state.resize(st->name,worksets_sizes[ws],dim[1],dim[2]); break;
          case 4:
            state.resize(st->name,worksets_sizes[ws],dim[1],dim[2],dim[3]); break;
          default:
            throw std::runtime_error("Error! Unsupported rank for elem state '" + st->name + "'.\n");
        }
      }
      // Remove <Cell> extent from dim, so we can use for sizing the nodeStateArray
      dim.erase(dim.begin());
    }
    switch (dim.size()) {
      case 1:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes); break;
      case 2:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes,dim[1]); break;
      case 3:
        nodeStateArrays[0][st->name].reset_from_dev_ptr(data,num_nodes,dim[1],dim[2]); break;
      default:
        throw std::runtime_error("Error! Unsupported rank for node state '" + st->name + "'.\n");
    }
  }

  // Global states
  for (const auto& st : global_sis) {
    auto& state = globalStates[st->name];
    if (st->dim.size()==1) {
      state.reset_from_host_ptr(&mesh_scalar_states[st->name],1);
    } else if (st->dim.size()==1) {
      state.reset_from_host_ptr(mesh_vector_states[st->name].data(),st->dim[0]);
    } else {
      throw std::runtime_error("Error! Unsupported rank for global state '" + st->name + "'.\n");
    }
  }
}

void OmegahMeshFieldAccessor::transferNodeStatesToElemStates ()
{
  auto elem_nodes_h = hostRead(OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(*m_mesh, Omega_h::VERT));
  int num_elem_nodes = Omega_h::element_degree(m_mesh->family(), m_mesh->dim(), 0);
  int num_ws = elemStateArrays.size();

  for (const auto& st : nodal_sis) {
    if (st->entity==StateStruct::NodalData)
      continue;
    const auto& dim = st->dim;
    const auto rank = st->dim.size();

    TEUCHOS_TEST_FOR_EXCEPTION (dim[1] != static_cast<size_t>(num_elem_nodes), std::runtime_error,
        "Error! State struct dim[1] does not match actual num_elem_nodes.\n"
        "  - state name: " + st->name + "\n"
        "  - dim[1]: " << dim[1] << "\n"
        "  - num_elem_nodes: " << num_elem_nodes << "\n");

    const auto& node_state = m_mesh->get_tag<ST>(0,st->name)->array();
    auto  node_state_h = hostRead(node_state);

    int elem_offset = 0;
    for (int ws=0; ws<num_ws; ++ws) {
      auto& elem_state = elemStateArrays[ws][st->name];
      auto& elem_state_h = elem_state.host();
      int ws_num_elems = elem_state_h.extent(0);

      for (int i=0; i<ws_num_elems; ++i) {
        int global_elem_idx = elem_offset + i;
        for (int j=0; j<num_elem_nodes; ++j) {
          // elem_nodes_h uses omega_h LIDs; node_state_h is indexed by omega_h LID.
          // Ghost node data is valid because sync_tag was called after loading fields.
          auto node_lid = elem_nodes_h[global_elem_idx*num_elem_nodes+j];
          switch(rank) {
            case 2:
              elem_state_h(i, j) = node_state_h[node_lid];
              break;
            case 3:
              for (size_t k=0; k<dim[2]; ++k) {
                elem_state_h(i, j, k) = node_state_h[node_lid*dim[2]+k];
              } break;
            case 4:
              for (size_t k=0; k<dim[2]; ++k) {
                for (size_t l=0; l<dim[3]; ++l) {
                  elem_state_h(i, j, k, l) = node_state_h[node_lid*dim[2]*dim[3]+k*dim[3]+l];
                }
              } break;
          }
        }
      }
      elem_state.sync_to_dev();
      elem_offset += ws_num_elems;
    }
  }
}

void OmegahMeshFieldAccessor::
fillVector (Thyra_Vector&        field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }

  // For now, assume only 1 entity has dofs (verts or elems)
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "[OmegahMeshFieldAccessor::fillVector] Only P0 or P1 fields supported for now.\n");
  auto dim = entity_dims_with_dofs[0];

  // TODO: you may want to do this on device, but you need an overload of
  //       the getNonconstDeviceData util that accepts a ref not an RCP.
  const auto& elem_dof_lids = field_dof_mgr->elem_dof_lids().host();
  const auto& elems = field_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int nelems = elems.size();
  const int ncomps = field_dof_mgr->getNumFields();

  auto owned_h = hostRead(m_mesh->owned(dim));
  auto mesh_data_h  = hostRead(m_mesh->get_array<ST>(dim,field_name));
  auto thyra_data_h = getNonconstLocalData(field_vector);
  auto elem_ents_h = hostRead(OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(*m_mesh,dim));
  const auto isSimplex = (m_mesh->family() == OMEGA_H_SIMPLEX);
  const auto nents_per_elem = isSimplex ? Omega_h::simplex_degree(m_mesh->dim(),dim) :
                                          Omega_h::hypercube_degree(m_mesh->dim(),dim);
  // DIAGNOSTIC: the loop below visits each dof once per adjacent element, so the raw
  // write count exceeds the vector size. What must match the vector size is the number
  // of DISTINCT lids written: any lid never written keeps whatever the (uninitialized)
  // allocation held, which silently corrupts the solution read back after adaptation.
  std::vector<bool> lid_written(thyra_data_h.size(),false);
  long long num_writes = 0;
  int max_lid_seen = -1;

  for (int ielem=0; ielem<nelems; ++ielem) {
    for (int icmp=0; icmp<field_dof_mgr->getNumFields(); ++icmp) {
      const auto& offsets = field_dof_mgr->getGIDFieldOffsets(icmp);
      for (int ient=0; ient<nents_per_elem; ++ient) {
        auto ent_lid = elem_ents_h[ielem*nents_per_elem+ient];
        if (overlapped or owned_h[ent_lid]) {
          auto lid = elem_dof_lids(ielem,offsets[ient]);
          // We may have lid<0 if the dof mgr is restricted to a mesh part
          // This happens for dirichlet BCs fields.
          if (lid>=0) {
            thyra_data_h[lid] = mesh_data_h[ent_lid*ncomps + icmp];
            ++num_writes;
            max_lid_seen = std::max(max_lid_seen,static_cast<int>(lid));
            if (lid<static_cast<int>(lid_written.size())) {
              lid_written[lid] = true;
            }
          }
        }
      }
    }
  }

  const auto num_distinct =
    std::count(lid_written.begin(),lid_written.end(),true);
  if (!m_mesh->comm()->rank()) {
    std::cout << "[fillVector] field '" << field_name << "' (ent dim " << dim << "):\n"
              << "  - vector local size : " << thyra_data_h.size() << "\n"
              << "  - distinct lids set : " << num_distinct << "\n"
              << "  - total writes      : " << num_writes << "\n"
              << "  - max lid seen      : " << max_lid_seen << "\n"
              << "  - mesh ents (dim)   : " << m_mesh->nents(dim)
              << ", ncomps " << ncomps
              << ", mesh array size " << mesh_data_h.size() << "\n";
    if (num_distinct != static_cast<long long>(thyra_data_h.size())) {
      std::cout << "  *** MISMATCH: " << (thyra_data_h.size()-num_distinct)
                << " dof(s) were never written ***\n";
    }
  }

  probe_compare(field_name,dim,ncomps,thyra_data_h,*field_dof_mgr);
}

void OmegahMeshFieldAccessor::
saveVector (const Thyra_Vector&  field_vector,
            const std::string&   field_name,
            const dof_mgr_ptr_t& field_dof_mgr,
            const bool           overlapped)
{
  // Figure out if it's a nodal or elem field
  const auto& fp = field_dof_mgr->getGeometricFieldPattern();
  const auto& ftopo = field_dof_mgr->get_topology();
  std::vector<int> entity_dims_with_dofs;
  for (unsigned dim=0; dim<=ftopo.getDimension(); ++dim) {
    if (fp->getSubcellIndices(dim,0).size()>0) {
      entity_dims_with_dofs.push_back(dim);
    }
  }

  // For now, assume only 1 entity has dofs (verts or elems)
  TEUCHOS_TEST_FOR_EXCEPTION (entity_dims_with_dofs.size()!=1, std::runtime_error,
      "[OmegahMeshFieldAccessor::fillVector] Only P0 or P1 fields supported for now.\n");
  auto dim = entity_dims_with_dofs[0];

  TEUCHOS_TEST_FOR_EXCEPTION (not m_mesh->has_tag(dim,field_name), std::runtime_error,
      "Error! Field '" + field_name + "' was not found as a tag in the mesh.\n");

  // TODO: you may want to do this on device, but you need an overload of
  //       the getNonconstDeviceData util that accepts a ref not an RCP.
  const auto& elem_dof_lids = field_dof_mgr->elem_dof_lids().host();
  const auto& elems = field_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int nelems = elems.size();
  const int ncomps = field_dof_mgr->getNumFields();

  auto mesh_data_h = hostWrite<ST>(m_mesh->nents(dim)*ncomps,field_name);
  auto owned_h = hostRead(m_mesh->owned(dim));
  auto thyra_data_h = getLocalData(field_vector);
  auto elem_ents_h = hostRead(OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(*m_mesh,dim));
  const auto isSimplex = (m_mesh->family() == OMEGA_H_SIMPLEX);
  const auto nents_per_elem = isSimplex ? Omega_h::simplex_degree(m_mesh->dim(),dim) :
                                          Omega_h::hypercube_degree(m_mesh->dim(),dim);
  for (int ielem=0; ielem<nelems; ++ielem) {
    for (int icmp=0; icmp<field_dof_mgr->getNumFields(); ++icmp) {
      const auto& offsets = field_dof_mgr->getGIDFieldOffsets(icmp);
      for (int ient=0; ient<nents_per_elem; ++ient) {
        auto ent_lid = elem_ents_h[ielem*nents_per_elem+ient];
        if (overlapped or owned_h[ent_lid]) {
          auto lid = elem_dof_lids(ielem,offsets[ient]);
          // We may have lid<0 if the dof mgr is restricted to a mesh part
          // This happens for dirichlet BCs fields.
          if (lid>=0) {
            mesh_data_h[ent_lid*ncomps + icmp] = thyra_data_h[lid];
          }
        }
      }
    }
  }

  m_mesh->set_tag(dim,field_name,read(mesh_data_h.write()),false);
  m_mesh->sync_tag(dim,field_name); //update ghosts

  probe_record(field_name,dim,ncomps);
}

// See the note on these two in the header.
void OmegahMeshFieldAccessor::
probe_record (const std::string& field_name, int dim, int ncomps)
{
  // Only interesting for the packed multi-component solution tag on vertices.
  if (field_name!="solution" or dim!=0 or ncomps<2) {
    return;
  }

  auto tag_h    = hostRead(m_mesh->get_array<ST>(dim,field_name));
  auto globals_h = hostRead(m_mesh->globals(dim));
  auto owned_h   = hostRead(m_mesh->owned(dim));
  auto coords_h  = hostRead(m_mesh->coords());
  const int mdim = m_mesh->dim();

  // Record EVERY owned vertex. Sampling a few in local-index order cannot tell us
  // whether a discrepancy is systematic or a one-off, so the comparison sweeps all
  // of them and reports counts (with a few exemplars).
  m_probe_vals.clear();
  for (int v=0; v<m_mesh->nents(dim); ++v) {
    if (not owned_h[v]) continue;
    ProbeEntry e;
    e.vals.resize(ncomps);
    for (int c=0; c<ncomps; ++c) {
      e.vals[c] = tag_h[v*ncomps + c];
    }
    e.coords.resize(mdim);
    for (int d=0; d<mdim; ++d) {
      e.coords[d] = coords_h[v*mdim + d];
    }
    m_probe_vals[globals_h[v]] = std::move(e);
  }

  if (!m_mesh->comm()->rank()) {
    std::cout << "[probe] recorded " << m_probe_vals.size()
              << " owned vertices of '" << field_name << "' at SAVE time (pre-adapt)\n";
  }
}

void OmegahMeshFieldAccessor::
probe_compare (const std::string& field_name, int dim, int ncomps,
               const Teuchos::ArrayRCP<const ST>& thyra_data,
               const DOFManager& dof_mgr)
{
  if (field_name!="solution" or dim!=0 or ncomps<2 or m_probe_vals.empty()) {
    return;
  }
  if (m_mesh->comm()->rank()) {
    return;
  }

  auto tag_h     = hostRead(m_mesh->get_array<ST>(dim,field_name));

  // Follow vertices via the marker tag stamped before the adapt (see the note where
  // 'adapt_probe_id' is registered in OmegahDiscretization::adapt). Global ids are
  // renumbered by the adaptation and cannot be used for this.
  if (not m_mesh->has_tag(dim,"adapt_probe_id")) {
    std::cout << "[probe] no 'adapt_probe_id' tag on the mesh; skipping comparison\n";
    return;
  }
  auto probe_h = hostRead(m_mesh->get_array<ST>(dim,"adapt_probe_id"));

  // Build vertex -> (elem,local position) so we can find the dof lids for a vertex.
  const auto& elem_dof_lids = dof_mgr.elem_dof_lids().host();
  const auto& elems = dof_mgr.getAlbanyConnManager()->getElementsInBlock();
  auto elem_ents_h = hostRead(OmegahGhost::getDownAdjacentEntsInClosureOfOwnedElms(*m_mesh,dim));
  const auto isSimplex = (m_mesh->family() == OMEGA_H_SIMPLEX);
  const auto nents_per_elem = isSimplex ? Omega_h::simplex_degree(m_mesh->dim(),dim) :
                                          Omega_h::hypercube_degree(m_mesh->dim(),dim);

  auto coords_h = hostRead(m_mesh->coords());
  const int mdim = m_mesh->dim();

  // Vertex -> (elem, local position), so we can find the dof lids for a vertex.
  // Built once: doing this per probe would be O(nverts*nelems).
  std::vector<int> vert2elem(m_mesh->nents(dim),-1), vert2pos(m_mesh->nents(dim),-1);
  for (int ielem=0; ielem<static_cast<int>(elems.size()); ++ielem) {
    for (int ient=0; ient<nents_per_elem; ++ient) {
      const auto v = elem_ents_h[ielem*nents_per_elem+ient];
      if (v>=0 and v<static_cast<int>(vert2elem.size()) and vert2elem[v]<0) {
        vert2elem[v] = ielem;
        vert2pos[v]  = ient;
      }
    }
  }

  // Index the new mesh by marker value, so each saved vertex is a map lookup rather
  // than a linear scan. Several vertices can share a marker (see the header note),
  // hence a multimap; the coordinate check below picks the right one.
  std::multimap<long long,int> marker2vert;
  for (int i=0; i<m_mesh->nents(dim); ++i) {
    const auto m = probe_h[i];
    const auto nearest = std::llround(m);
    if (std::abs(m-static_cast<ST>(nearest)) < 1e-9) {
      marker2vert.emplace(nearest,i);
    }
  }

  const ST coord_tol = 1e-10;
  long long n_survivors = 0, n_mismatch = 0, n_absent = 0, n_ambiguous = 0;
  ST worst_diff = 0;
  long long worst_gid = -1;
  int n_shown = 0;
  const int max_show = 5;

  std::cout << "[probe] comparing at FILL time (post-adapt),"
               " survivors identified by marker + coordinates\n";

  for (const auto& [gid,saved] : m_probe_vals) {
    // Candidates carrying this marker value...
    auto range = marker2vert.equal_range(static_cast<long long>(gid));
    // ...of which the survivor is the one that is still at the SAME coordinates.
    int v = -1;
    int n_cand = 0;
    for (auto it=range.first; it!=range.second; ++it) {
      ++n_cand;
      bool same_pos = true;
      for (int d=0; d<mdim and same_pos; ++d) {
        if (std::abs(coords_h[it->second*mdim + d] - saved.coords[d]) > coord_tol) {
          same_pos = false;
        }
      }
      if (same_pos) { v = it->second; break; }
    }

    if (v<0) {
      // Either the vertex is gone (coarsening/swap), or it moved. Under 'Refine Only'
      // this should not happen: refinement only ADDS vertices.
      ++n_absent;
      continue;
    }
    if (n_cand>1) ++n_ambiguous;  // marker collision that coords resolved
    ++n_survivors;

    // This vertex provably survived: same marker AND same coordinates. Its values
    // must therefore be unchanged.
    ST maxdiff = 0;
    for (int c=0; c<ncomps; ++c) {
      maxdiff = std::max(maxdiff, std::abs(tag_h[v*ncomps+c]-saved.vals[c]));
    }
    if (maxdiff > worst_diff) { worst_diff = maxdiff; worst_gid = gid; }

    if (maxdiff > 1e-8) {
      ++n_mismatch;
      if (n_shown < max_show) {
        ++n_shown;
        std::cout << "  *** CHANGED: orig gid " << gid << " -> new lid " << v
                  << " (marker " << probe_h[v] << ", maxdiff " << maxdiff << ")\n";
        std::cout << "    coords    :";
        for (int d=0; d<mdim; ++d) std::cout << " " << saved.coords[d];
        std::cout << "\n    saved tag :";
        for (auto s : saved.vals) std::cout << " " << s;
        std::cout << "\n    new tag   :";
        for (int c=0; c<ncomps; ++c) std::cout << " " << tag_h[v*ncomps+c];
        std::cout << "\n";
        if (vert2elem[v]>=0) {
          std::cout << "    thyra vec :";
          for (int c=0; c<ncomps; ++c) {
            const auto& offsets = dof_mgr.getGIDFieldOffsets(c);
            auto lid = elem_dof_lids(vert2elem[v],offsets[vert2pos[v]]);
            if (lid>=0) std::cout << " " << thyra_data[lid];
            else        std::cout << " <neg lid>";
          }
          std::cout << "\n";
        }
      }
    }
  }

  std::cout << "  - recorded pre-adapt   : " << m_probe_vals.size() << "\n"
            << "  - survivors identified : " << n_survivors
            << " (of which " << n_ambiguous << " needed coords to break a marker tie)\n"
            << "  - survivors CHANGED    : " << n_mismatch << "\n"
            << "  - not found post-adapt : " << n_absent << "\n"
            << "  - worst |diff| on a survivor: " << worst_diff
            << " (orig gid " << worst_gid << ")\n";
  if (n_mismatch==0) {
    std::cout << "  => every identified survivor kept its values exactly\n";
  }
}

void OmegahMeshFieldAccessor::
setSolutionFieldsMetadata (const int neq)
{
  // For now, just add a tag of the proper length
  // TODO: if/when we add non-nodal FE types, this needs to be revisited
  addFieldOnMesh("solution",0,neq);
}

std::vector<std::string> OmegahMeshFieldAccessor::get_nodal_field_names () const
{
  std::vector<std::string> names;
  for (const auto& [name, tag_handle] : m_tags) {
    if (tag_handle.ent_dim==0) {
      names.push_back(name);
    }
  }
  return names;
}

void OmegahMeshFieldAccessor::reset_mesh_tags ()
{
  for (auto& [name, tag_handle] : m_tags) {
    auto& array = tag_handle.array;
    int dim     = tag_handle.ent_dim;
    int ncmp    = tag_handle.ncomps;
    if (m_mesh->has_tag(dim,name)) {
      auto tag = m_mesh->get_tag<ST>(dim,name);
      array = Omega_h::Write<ST>(tag->array().size(),name);
      Kokkos::deep_copy(array.view(),tag->array().view());
      m_mesh->set_tag(dim,name,read(array));
    } else {
      if (!m_mesh->comm()->rank()) {
        std::cout << "WARNING! Field '" << name << "' (dim " << dim << ") did not "
                     "survive the mesh adaptation and has been reset to zero.\n";
      }
      array = Omega_h::Write<ST>(m_mesh->nents(dim)*ncmp,name);
      m_mesh->add_tag(dim,name,ncmp,read(array));
    }
  }
}

} // namespace Albany
