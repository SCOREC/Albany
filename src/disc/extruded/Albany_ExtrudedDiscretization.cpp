//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <Albany_ExtrudedDiscretization.hpp>

#include <Albany_ExtrudedConnManager.hpp>
#include <Albany_CommUtils.hpp>
#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_Utils.hpp"
#include "Albany_StringUtils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_CombineAndScatterManager.hpp"

#if defined(ALBANY_OMEGAH)
#include "Albany_OmegahGenericMesh.hpp"
#endif

#include <limits>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include "Albany_ProblemUtils.hpp"

#include <PHAL_Dimension.hpp>

#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_ElemFieldPattern.hpp>

#include <iostream>
#include <string>

// Uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

namespace Albany {

ExtrudedDiscretization::
ExtrudedDiscretization (const Teuchos::RCP<Teuchos::ParameterList>&     discParams,
                        const int                                       neq,
                        const Teuchos::RCP<ExtrudedMesh>&               extruded_mesh,
                        const Teuchos::RCP<AbstractDiscretization>&     basal_disc,
                        const Teuchos::RCP<const Teuchos_Comm>&         comm,
                        const Teuchos::RCP<RigidBodyModes>&             rigidBodyModes,
                        const std::map<int, std::vector<std::string>>&  sideSetEquations)
 : m_comm(comm)
 , m_basal_disc (basal_disc)
 , m_sideSetEquations(sideSetEquations)
 , m_rigid_body_modes(rigidBodyModes)
 , m_extruded_mesh(extruded_mesh)
 , m_disc_params (discParams)
{
  setNumEq(neq);

  sideSetDiscretizations["basalside"] = basal_disc;
  sideSetDiscretizations["upperside"] = basal_disc;
}

void ExtrudedDiscretization::setNumEq (int neq)
{
  m_neq = neq;

  // On the basal mesh, make the solution vector numNodeLayers times longer
  int basal_neq = neq * m_extruded_mesh->layers_data.node.gid->numLayers;
  m_basal_disc->setNumEq(basal_neq);
}

void
ExtrudedDiscretization::setupMLCoords()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: setupMLCoords");
  if (m_rigid_body_modes.is_null()) { return; }
  if (!m_rigid_body_modes->isTekoUsed() && !m_rigid_body_modes->isMueLuUsed() && !m_rigid_body_modes->isFROSchUsed()) { return; }

  const int numDim = getNumDim();
  auto coordMV = Thyra::createMembers(getNodeVectorSpace(), numDim);
  auto coordMV_data = getNonconstLocalData(coordMV);

  // NOTE: you cannot use DOFManager dof gids as entity ID in stk, and viceversa.
  // All you can do is loop over dofs/nodes in an element, since you have the following guarantees:
  //  - elem GIDs are the same in DOFManager and stk mesh
  //  - nodes ordering is the same in DOFManager and stk mesh
  // We'll loop over certain nodes more than once, but this is a setup method, so it's fine
  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& elems = node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int   num_elems = elems.size();
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto& node_dofs = node_dof_mgr->getElementGIDs(ielem);
    const int num_nodes = node_dofs.size();
    for (int i=0; i<num_nodes; ++i) {
      LO node_lid = node_dof_mgr->indexer()->getLocalElement(node_dofs[i]);
      if (node_lid>=0) {
        double* X = &m_nodes_coordinates[numDim*node_lid];
        for (int j=0; j<numDim; ++j) {
          coordMV_data[j][node_lid] = X[j];
        }
      }
    }
  }

  m_rigid_body_modes->setCoordinatesAndComputeNullspace(
      coordMV,
      getVectorSpace(),
      getOverlapVectorSpace());
}

// NOTE: the 3d solution must be PERMUTED into the basal per-column layout before it
// can be stored on the basal mesh -- see the note on basal_cmp in
// ExtrudedMeshFieldAccessor. Forwarding the 3d vector straight to the basal disc
// packs dofs from the other nodes of a basal element into a vertex's column.
void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (soln_dxdp != Teuchos::null, std::runtime_error,
      "ExtrudedDiscretization::writeSolutionToMeshDatabase does not support sensitivities yet.");
  m_extruded_mesh->get_extruded_field_accessor()
      ->saveLayeredSolution(soln,solution_dof_name(),overlapped);
}

void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const bool overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (soln_dxdp != Teuchos::null, std::runtime_error,
      "ExtrudedDiscretization::writeSolutionToMeshDatabase does not support sensitivities yet.");
  auto mfa = m_extruded_mesh->get_extruded_field_accessor();
  mfa->saveLayeredSolution(soln,    solution_dof_name(),          overlapped);
  mfa->saveLayeredSolution(soln_dot,solution_dof_name()+"_dot",   overlapped);
}

void
ExtrudedDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const bool overlapped)
{
  TEUCHOS_TEST_FOR_EXCEPTION (soln_dxdp != Teuchos::null, std::runtime_error,
      "ExtrudedDiscretization::writeSolutionToMeshDatabase does not support sensitivities yet.");
  auto mfa = m_extruded_mesh->get_extruded_field_accessor();
  mfa->saveLayeredSolution(soln,       solution_dof_name(),           overlapped);
  mfa->saveLayeredSolution(soln_dot,   solution_dof_name()+"_dot",    overlapped);
  mfa->saveLayeredSolution(soln_dotdot,solution_dof_name()+"_dotdot", overlapped);
}

void
ExtrudedDiscretization::writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool overlapped)
{
  m_basal_disc->writeSolutionMVToMeshDatabase(soln,soln_dxdp,overlapped);
}

void
ExtrudedDiscretization::writeMeshDatabaseToFile(const double time,
                                                const bool   force_write_solution)
{
  m_basal_disc->writeMeshDatabaseToFile(time,force_write_solution);
}

void ExtrudedDiscretization::
writeWedgeVtk (const std::string& basename,
               const Teuchos::RCP<const Thyra_Vector>& solution,
               const bool solution_is_overlapped) const
{
  // The 3d mesh is never stored: it is the basal mesh plus the layered numbering.
  // Materialize it here. Nodes come straight from m_nodes_coordinates (which is
  // indexed by OVERLAP node lid, see computeCoordinates), and elements are assembled
  // by walking the basal elements x layers, exactly as ExtrudedConnManager does.
  const auto& layers_data = m_extruded_mesh->layers_data;
  const int num_layers = layers_data.cell.lid->numLayers;
  const int mesh_dim   = getNumDim();

  const auto& node_dof_mgr   = getNodeDOFManager();
  const auto& node_indexer   = node_dof_mgr->ov_indexer();
  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elems    = basal_node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int num_basal_elems  = basal_elems.size();

  const int num_nodes = getLocalSubdim(getOverlapNodeVectorSpace());
  const int num_wedges = num_basal_elems * num_layers;

  // --- Cells: bottom triangle then top triangle == VTK_WEDGE (type 13) ordering,
  // which is also the layer-by-layer node ordering the extruded conn manager uses.
  std::vector<int> conn;
  conn.reserve(num_wedges*6);
  std::vector<int> cell_layer;   cell_layer.reserve(num_wedges);
  std::vector<int> cell_column;  cell_column.reserve(num_wedges);

  int num_bad_cells = 0;
  for (int ibelem=0; ibelem<num_basal_elems; ++ibelem) {
    const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ibelem);
    // The basal element must be a triangle for the wedge to make sense.
    TEUCHOS_TEST_FOR_EXCEPTION (basal_node_gids.size()!=3, std::runtime_error,
        "[ExtrudedDiscretization::writeWedgeVtk] Expected a triangular basal element, got "
        << basal_node_gids.size() << " nodes.\n");

    for (int ilay=0; ilay<num_layers; ++ilay) {
      int wedge[6];
      bool ok = true;
      for (int ilev=0; ilev<2; ++ilev) {          // bottom (ilay), then top (ilay+1)
        for (int n=0; n<3; ++n) {
          const GO ngid = layers_data.node.gid->getId(basal_node_gids[n], ilay+ilev);
          const LO nlid = node_indexer->getLocalElement(ngid);
          if (nlid<0) { ok = false; }
          wedge[3*ilev+n] = nlid;
        }
      }
      if (not ok) { ++num_bad_cells; continue; }  // element not fully local: skip it
      conn.insert(conn.end(), wedge, wedge+6);
      cell_layer.push_back(ilay);
      cell_column.push_back(ibelem);
    }
  }
  const int num_cells = cell_layer.size();

  // --- Nodal solution data. Map dofs to nodes via the element loop: a dof manager
  // gives dofs per element, and the node ordering inside an element matches the node
  // dof manager's, so (elem,node,eq) -> dof lid is unambiguous.
  const int neq = m_neq;
  std::vector<std::vector<double>> soln_at_nodes;
  Teuchos::RCP<const Thyra_Vector> soln_ov;
  if (not solution.is_null()) {
    if (solution_is_overlapped) {
      soln_ov = solution;
    } else {
      // Bring the owned solution to the overlap distribution, so ghost nodes have values.
      auto cas = createCombineAndScatterManager(getVectorSpace(),getOverlapVectorSpace());
      auto tmp = Thyra::createMember(getOverlapVectorSpace());
      tmp->assign(0.0);
      cas->scatter(*solution,*tmp,CombineMode::INSERT);
      soln_ov = tmp;
    }

    auto soln_data = getLocalData(soln_ov);
    const auto& dof_mgr = getDOFManager();
    const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();

    soln_at_nodes.assign(neq, std::vector<double>(num_nodes,0.0));
    for (int ibelem=0; ibelem<num_basal_elems; ++ibelem) {
      const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ibelem);
      for (int ilay=0; ilay<num_layers; ++ilay) {
        const int ielem3d = layers_data.cell.lid->getId(ibelem,ilay);
        for (int eq=0; eq<neq; ++eq) {
          const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
          // offsets are ordered layer-by-layer: first the 3 bottom nodes, then the 3 top
          for (size_t k=0; k<offsets.size(); ++k) {
            const LO dof_lid = elem_dof_lids(ielem3d,offsets[k]);
            if (dof_lid<0) { continue; }
            const int ilev = k/3;           // 0 = bottom, 1 = top
            const int n    = k%3;
            const GO ngid  = layers_data.node.gid->getId(basal_node_gids[n], ilay+ilev);
            const LO nlid  = node_indexer->getLocalElement(ngid);
            if (nlid<0) { continue; }
            soln_at_nodes[eq][nlid] = soln_data[dof_lid];
          }
        }
      }
    }
  }

  // --- Extruded mesh fields (ice_thickness, surface_height, temperature, ...).
  // These are NOT in the solution vector: they live on the basal mesh, viewed through
  // the extruded field accessor as ELEMENT states with layout (elem_in_ws, node_in_elem
  // [, cmp]) -- see the note in ExtrudedMeshFieldAccessor about 3d states all being elem
  // states. Scatter them to nodes so a bad extrude/interpolate is visible next to the
  // solution it corrupts. A node shared by several elements is written more than once;
  // the values agree wherever the transfer is correct, which is the point of looking.
  std::vector<std::pair<std::string,std::vector<double>>> mesh_fields;
  {
    const auto mfa = m_extruded_mesh->get_field_accessor();
    const auto& elem_states = mfa->getElemStates();
    // Use the NODE dof mgr (1 dof per node), so a dof lid IS a node lid -- the same
    // lid space m_nodes_coordinates and the wedge connectivity are indexed by.
    const auto& node_dm = getNodeDOFManager();
    const auto& elem_dof_lids = node_dm->elem_dof_lids().host();
    const auto& node_offsets = node_dm->getGIDFieldOffsets(0);

    for (const auto& st : mfa->getAllSIS()) {
      const auto& name = st->name;
      // Only node-ish states scatter to points; skip elem/global/QP states.
      if (st->entity!=StateStruct::NodalData and
          st->entity!=StateStruct::ElemNode and
          st->entity!=StateStruct::NodalDataToElemNode and
          st->entity!=StateStruct::NodalDistParameter) { continue; }

      // Number of components: rank-2 state is scalar, rank-3 carries a component dim.
      int ncmp = 1;
      bool found = false, bad_rank = false;
      for (int ws=0; ws<static_cast<int>(m_workset_sizes.size()); ++ws) {
        auto it = elem_states[ws].find(name);
        if (it==elem_states[ws].end()) { continue; }
        const auto r = it->second.host().rank();
        if (r<2 or r>3) { bad_rank = true; break; }
        ncmp = (r==3) ? it->second.host().extent(2) : 1;
        found = true;
        break;
      }
      if (not found or bad_rank) { continue; }   // not a per-node state we can plot

      std::vector<std::vector<double>> vals(ncmp, std::vector<double>(num_nodes,0.0));
      for (int ws=0; ws<static_cast<int>(m_workset_sizes.size()); ++ws) {
        auto it = elem_states[ws].find(name);
        if (it==elem_states[ws].end()) { continue; }
        auto state_h = it->second.host();
        auto ws_elems = m_workset_elements.host();
        for (int ie=0; ie<m_workset_sizes[ws]; ++ie) {
          const int elem_lid = ws_elems(ws,ie);
          for (size_t k=0; k<node_offsets.size(); ++k) {
            const LO dof_lid = elem_dof_lids(elem_lid,node_offsets[k]);
            if (dof_lid<0) { continue; }
            // For the scalar (component 0) dof mgr, dof lid == node lid.
            const LO nlid = dof_lid;
            if (nlid>=num_nodes) { continue; }
            for (int c=0; c<ncmp; ++c) {
              vals[c][nlid] = (state_h.rank()==3) ? state_h(ie,k,c) : state_h(ie,k);
            }
          }
        }
      }
      for (int c=0; c<ncmp; ++c) {
        std::string nm = name + (ncmp>1 ? "_"+std::to_string(c) : "");
        std::replace(nm.begin(),nm.end(),' ','_');
        mesh_fields.emplace_back(nm,std::move(vals[c]));
      }
    }
  }

  // DIAGNOSTIC: resolve the layout of the packed basal 'solution' tag empirically.
  // The basal Omega_h mesh stores the whole 3d solution at each basal vertex as one
  // tag of neq*(num_layers+1) components. Which slot holds which (layer,equation) is
  // NOT self-evident: 'velocity' is packed component-major (icomp*numLayers+il, see
  // readLayeredVectorFileSerial) while the solution is built through the layered
  // numbering, so the two conventions coexist in the same mesh. Rather than infer it,
  // print -- for a few INTERIOR basal vertices -- all the tag slots next to the
  // cmp_* values of the 6 nodes of that vertex's column, taken from the same vector
  // this file is being written from.
  //   - if the column's cmp_0 values land at slots {0,neq,2*neq,...}: layer-major
  //     (slot = neq*layer + eq)
  //   - if they land at slots {0,1,2,...}: component-major (slot = layer + eq*nlev)
  //   - if they appear at neither, the tag and the read-back vector disagree, which
  //     is a genuine transfer bug rather than a layout misreading.
  // Interior vertices only: a boundary vertex has SDBC-prescribed dofs, which agree
  // trivially and so prove nothing about the layout.
#if defined(ALBANY_OMEGAH)
  if (not soln_at_nodes.empty()) {
    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    auto basal_ms = m_basal_disc->getMeshStruct();
    auto oh_mesh_struct = Teuchos::rcp_dynamic_cast<OmegahGenericMesh>(basal_ms);
    if (Teuchos::nonnull(oh_mesh_struct)) {
      auto oh_mesh = oh_mesh_struct->getOmegahMesh();
      const std::string tag_name = solution_dof_name();
      if (oh_mesh->has_tag(0,tag_name)) {
        auto tag_h    = Omega_h::HostRead<ST>(oh_mesh->get_array<ST>(0,tag_name));
        auto gids_h   = Omega_h::HostRead<Omega_h::GO>(oh_mesh->globals(0));
        auto coords_h = Omega_h::HostRead<Omega_h::Real>(oh_mesh->coords());
        const int bdim   = oh_mesh->dim();
        const int nverts = oh_mesh->nverts();
        const int ncomps = tag_h.size()/std::max(nverts,1);
        const int nlev   = num_layers+1;

        // A basal vertex is "interior" if it is not on any marked boundary part.
        std::vector<bool> on_bdry(nverts,false);
        for (const auto& nsname : basal_ms->meshSpecs[0]->nsNames) {
          for (int d=0; d<=oh_mesh->dim(); ++d) {
            if (not oh_mesh->has_tag(d,nsname)) { continue; }
            if (d!=0) { continue; }  // only vertex marks identify a vertex directly
            auto marked = Omega_h::HostRead<Omega_h::I8>(oh_mesh->get_array<Omega_h::I8>(d,nsname));
            for (int i=0; i<nverts and i<marked.size(); ++i) {
              if (marked[i]) { on_bdry[i] = true; }
            }
          }
        }

        *out << "[tag layout probe] basal '" << tag_name << "' tag: " << ncomps
             << " comps/vertex, neq=" << neq << ", node layers=" << nlev << "\n";

        // The layered numbering turns (basal entity, level) into a 3d id by striding
        // with numHorizEntities (LAYER ordering). If that count disagrees with the
        // number of basal vertices the tag is actually packed over, then values land a
        // fixed number of vertices away from where they are read -- which shows up as
        // one column's layer values scattered across several vertices' tags.
        const auto n_horiz_node_gid = layers_data.node.gid->numHorizEntities;
        const auto n_horiz_node_lid = layers_data.node.lid->numHorizEntities;
        const auto n_horiz_cell_gid = layers_data.cell.gid->numHorizEntities;
        *out << "  [counts] layered numHorizEntities: node.gid=" << n_horiz_node_gid
             << " node.lid=" << n_horiz_node_lid
             << " cell.gid=" << n_horiz_cell_gid << "\n"
             << "  [counts] omegah basal mesh: nverts=" << nverts
             << " nelems=" << oh_mesh->nelems() << "\n"
             << "  [counts] basal disc: num local nodes="
             << basal_ms->get_num_local_nodes()
             << " num local elems=" << basal_ms->get_num_local_elements() << "\n"
             << "  [counts] extruded: 3d nodes(ov)=" << num_nodes
             << " basal elems walked=" << num_basal_elems
             << " elem layers=" << num_layers << "\n";
        if (static_cast<long long>(n_horiz_node_gid)!=static_cast<long long>(nverts)) {
          *out << "  [counts] *** MISMATCH: layered node.gid numHorizEntities ("
               << n_horiz_node_gid << ") != omegah nverts (" << nverts << ").\n"
                  "      The tag is packed over nverts, but 3d ids stride by "
               << n_horiz_node_gid << ".\n";
        }

        // Walk basal elements to reach basal node GIDs, then match each to its
        // Omega_h vertex by global id.
        int n_shown = 0;
        for (int ibelem=0; ibelem<num_basal_elems and n_shown<3; ++ibelem) {
          const auto& bgids = basal_node_dof_mgr->getElementGIDs(ibelem);
          for (int n=0; n<3 and n_shown<3; ++n) {
            const GO bgid = bgids[n];
            // Find this basal node's Omega_h vertex by matching global ids.
            int vtx = -1;
            for (int v=0; v<nverts; ++v) {
              if (static_cast<GO>(gids_h[v])==bgid) { vtx = v; break; }
            }
            if (vtx<0 or on_bdry[vtx]) { continue; }

            *out << "  [vertex] basal gid " << bgid << " (oh vert " << vtx << ") at (";
            for (int d=0; d<bdim; ++d) { *out << coords_h[bdim*vtx+d] << (d+1<bdim?",":""); }
            *out << ")\n";

            *out << "    tag slots:";
            for (int c=0; c<ncomps; ++c) {
              *out << " [" << c << "]=" << tag_h[vtx*ncomps+c];
            }
            *out << "\n";

            // The same column, as written to the vtk file, bottom (layer 0) to top.
            for (int eq=0; eq<neq; ++eq) {
              *out << "    cmp_" << eq << " by layer:";
              for (int ilev=0; ilev<nlev; ++ilev) {
                const GO ngid = layers_data.node.gid->getId(bgid,ilev);
                const LO nlid = node_indexer->getLocalElement(ngid);
                if (nlid<0) { *out << " L" << ilev << "=<n/a>"; continue; }
                *out << " L" << ilev << "=" << soln_at_nodes[eq][nlid];
              }
              *out << "\n";
            }

            // The addressing chain, level by level, for this column. Three routes to
            // the same node must agree:
            //   (a) node.gid->getId(bgid,ilev) -> ov node indexer -> node LID
            //       (what this writer and computeCoordinates use; LAYER-ordered on
            //        the GID side, so level strides by numHorizEntities)
            //   (b) node.lid->getId(basal_lid,ilev)
            //       (hardcoded COLUMN, so level strides by 1 within a column block)
            //   (c) the solution dof mgr's lid for (eq 0) at that node
            //       (the layout the solution vector and the basal tag actually use:
            //        vertex-major, level striding by neq inside an 18-dof block)
            // If (a) and (c) disagree, the writer is reading the wrong dofs. If (a)
            // and (b) disagree, the GID and LID numberings are not describing the
            // same column. Either explains the tag/cmp mismatch without any of the
            // data itself being wrong.
            {
              // Find this basal node's LOCAL id, needed for route (b).
              const LO b_nlid = basal_node_dof_mgr->ov_indexer()->getLocalElement(bgid);
              *out << "    [addressing] basal gid " << bgid
                   << " basal lid " << b_nlid << "\n";
              const auto& sol_dm = getDOFManager();
              const auto& sol_elem_lids = sol_dm->elem_dof_lids().host();
              const auto& sol_offsets0  = sol_dm->getGIDFieldOffsets(0);
              for (int ilev=0; ilev<nlev; ++ilev) {
                const GO ngid_a = layers_data.node.gid->getId(bgid,ilev);
                const LO nlid_a = node_indexer->getLocalElement(ngid_a);
                const LO nlid_b = (b_nlid>=0)
                                ? layers_data.node.lid->getId(b_nlid,ilev) : LO(-1);
                // Route (c): the 3d element of this column at the element-layer just
                // below (or at, for level 0) this node level, then the dof offset for
                // the matching node position within that element.
                const int ilay = std::min(ilev,num_layers-1);
                const int side = (ilev==ilay) ? 0 : 1;   // 0=elem bottom, 1=elem top
                const int ielem3d = layers_data.cell.lid->getId(ibelem,ilay);
                const int k = side*3 + n;                // layer-by-layer node order
                LO dof_c = -1;
                if (k < static_cast<int>(sol_offsets0.size())) {
                  dof_c = sol_elem_lids(ielem3d,sol_offsets0[k]);
                }
                *out << "      L" << ilev
                     << ": (a) gid=" << ngid_a << " -> lid=" << nlid_a
                     << " | (b) lid=" << nlid_b
                     << " | (c) elem3d=" << ielem3d << " k=" << k
                     << " dof_lid=" << dof_c;
                if (nlid_a>=0) { *out << " | cmp_0(a)=" << soln_at_nodes[0][nlid_a]; }
                *out << "\n";
              }
            }
            ++n_shown;
          }
        }
        if (n_shown==0) {
          *out << "  (no interior basal vertex found to probe)\n";
        }
      }
    }
  }
#endif // ALBANY_OMEGAH

  // --- Write one legacy VTK file per rank.
  const int rank = m_comm->getRank();
  const std::string fname = basename + "_r" + std::to_string(rank) + ".vtk";
  std::ofstream f(fname);
  TEUCHOS_TEST_FOR_EXCEPTION (not f.is_open(), std::runtime_error,
      "[ExtrudedDiscretization::writeWedgeVtk] Could not open '" + fname + "' for writing.\n");
  f << std::scientific << std::setprecision(12);

  f << "# vtk DataFile Version 3.0\n"
    << "Albany extruded mesh (wedges)\n"
    << "ASCII\n"
    << "DATASET UNSTRUCTURED_GRID\n";

  f << "POINTS " << num_nodes << " double\n";
  for (int i=0; i<num_nodes; ++i) {
    for (int d=0; d<3; ++d) {
      f << (d<mesh_dim ? m_nodes_coordinates[mesh_dim*i+d] : 0.0) << (d<2 ? " " : "\n");
    }
  }

  f << "\nCELLS " << num_cells << " " << num_cells*7 << "\n";
  for (int c=0; c<num_cells; ++c) {
    f << "6";
    for (int k=0; k<6; ++k) { f << " " << conn[6*c+k]; }
    f << "\n";
  }
  f << "\nCELL_TYPES " << num_cells << "\n";
  for (int c=0; c<num_cells; ++c) { f << "13\n"; }   // 13 == VTK_WEDGE

  f << "\nCELL_DATA " << num_cells << "\n";
  f << "SCALARS layer int 1\nLOOKUP_TABLE default\n";
  for (int c=0; c<num_cells; ++c) { f << cell_layer[c] << "\n"; }
  f << "SCALARS basal_column int 1\nLOOKUP_TABLE default\n";
  for (int c=0; c<num_cells; ++c) { f << cell_column[c] << "\n"; }

  f << "\nPOINT_DATA " << num_nodes << "\n";
  if (not soln_at_nodes.empty()) {
    const auto& dof_mgr = getDOFManager();
    for (int eq=0; eq<neq; ++eq) {
      // Sanitize the field name: VTK does not allow spaces in dataset names.
      std::string nm = dof_mgr->getFieldString(eq);
      std::replace(nm.begin(),nm.end(),' ','_');
      f << "SCALARS " << nm << " double 1\nLOOKUP_TABLE default\n";
      for (int i=0; i<num_nodes; ++i) { f << soln_at_nodes[eq][i] << "\n"; }
    }
  }
  // The extruded mesh fields (ice_thickness, surface_height, temperature, ...).
  for (const auto& mf : mesh_fields) {
    f << "SCALARS " << mf.first << " double 1\nLOOKUP_TABLE default\n";
    for (int i=0; i<num_nodes; ++i) { f << mf.second[i] << "\n"; }
  }
  // Always emit the column geometry, so the file is useful even with no solution.
  f << "SCALARS z double 1\nLOOKUP_TABLE default\n";
  for (int i=0; i<num_nodes; ++i) {
    f << (mesh_dim>2 ? m_nodes_coordinates[mesh_dim*i+2] : 0.0) << "\n";
  }
  f.close();

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  if (rank==0) {
    *out << "[writeWedgeVtk] wrote " << m_comm->getSize() << " file(s) '"
         << basename << "_r*.vtk': " << num_cells << " wedges, "
         << num_nodes << " nodes, " << num_layers << " layers";
    if (not soln_at_nodes.empty()) { *out << ", " << neq << " solution field(s)"; }
    *out << ", " << mesh_fields.size() << " mesh field(s):";
    for (const auto& mf : mesh_fields) { *out << " " << mf.first; }
    *out << "\n";
  }
  if (num_bad_cells>0) {
    *out << "[writeWedgeVtk] rank " << rank << ": skipped " << num_bad_cells
         << " column element(s) with unmapped node LIDs.\n";
  }
}

// DIAGNOSTIC. See the note on the declaration in the header for the encoding and
// what a violation of it proves.
Teuchos::RCP<Thyra_Vector>
ExtrudedDiscretization::makeKnownValueSolution () const
{
  const auto& layers_data  = m_extruded_mesh->layers_data;
  const int   num_layers   = layers_data.cell.lid->numLayers;
  const int   neq          = m_neq;

  const auto& dof_mgr        = getDOFManager();
  const auto& elem_dof_lids  = dof_mgr->elem_dof_lids().host();
  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elems    = basal_node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int   num_basal_elems = basal_elems.size();

  // The stride that separates two node levels of the same column in the encoding.
  // It must be the SAME number the layered numbering uses to stride levels, or the
  // encoded value would not be decodable the way the header describes. Take it from
  // the numbering itself rather than from the local vertex count (which differs per
  // rank) so the values are globally unique and rank-independent.
  const GO nbasal = layers_data.node.gid->numHorizEntities;

  // Fill the OVERLAP vector first: elem_dof_lids are overlap lids (see writeWedgeVtk),
  // and every dof of every local element -- including ghosts -- must get a value.
  auto x_ov = Thyra::createMember(getOverlapVectorSpace());
  x_ov->assign(0.0);
  auto x_ov_data = getNonconstLocalData(x_ov);

  long long num_set = 0;
  for (int ibelem=0; ibelem<num_basal_elems; ++ibelem) {
    const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ibelem);
    TEUCHOS_TEST_FOR_EXCEPTION (basal_node_gids.size()!=3, std::runtime_error,
        "[makeKnownValueSolution] Expected a triangular basal element, got "
        << basal_node_gids.size() << " nodes.\n");
    for (int ilay=0; ilay<num_layers; ++ilay) {
      const int ielem3d = layers_data.cell.lid->getId(ibelem,ilay);
      for (int eq=0; eq<neq; ++eq) {
        const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
        // offsets are ordered layer-by-layer: the 3 bottom nodes, then the 3 top ones
        for (size_t k=0; k<offsets.size(); ++k) {
          const LO dof_lid = elem_dof_lids(ielem3d,offsets[k]);
          if (dof_lid<0) { continue; }
          const int ilev = ilay + static_cast<int>(k)/3;   // absolute NODE level
          const int n    = static_cast<int>(k)%3;
          const GO  g    = basal_node_gids[n];
          x_ov_data[dof_lid] = static_cast<ST>(eq)*1e6
                             + static_cast<ST>(g + nbasal*ilev);
          ++num_set;
        }
      }
    }
  }

  // Overlap -> owned. Shared dofs get the same value from every owner, so INSERT
  // cannot produce a rank-dependent answer here.
  auto x = Thyra::createMember(getVectorSpace());
  x->assign(0.0);
  auto cas = createCombineAndScatterManager(getVectorSpace(),getOverlapVectorSpace());
  cas->combine(*x_ov,*x,CombineMode::INSERT);

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
  if (m_comm->getRank()==0) {
    *out << "[knownValue] built a synthetic solution: value = eq*1e6 + (basal_gid + "
         << nbasal << "*level)\n"
         << "  - neq " << neq << ", node levels " << (num_layers+1)
         << ", layered node stride (numHorizEntities) " << nbasal << "\n"
         << "  - owned dofs " << getLocalSubdim(getVectorSpace())
         << ", overlap dofs " << getLocalSubdim(getOverlapVectorSpace())
         << ", dof writes " << num_set << "\n";
  }
  return x;
}

int ExtrudedDiscretization::
checkKnownValueTag (const std::string& context) const
{
#if defined(ALBANY_OMEGAH)
  auto basal_ms = m_basal_disc->getMeshStruct();
  auto oh_mesh_struct = Teuchos::rcp_dynamic_cast<OmegahGenericMesh>(basal_ms);
  if (Teuchos::is_null(oh_mesh_struct)) { return 0; }
  auto oh_mesh = oh_mesh_struct->getOmegahMesh();

  const std::string tag_name = solution_dof_name();
  if (not oh_mesh->has_tag(0,tag_name)) { return 0; }

  auto out = Teuchos::VerboseObjectBase::getDefaultOStream();

  auto tag_h   = Omega_h::HostRead<ST>(oh_mesh->get_array<ST>(0,tag_name));
  auto gids_h  = Omega_h::HostRead<Omega_h::GO>(oh_mesh->globals(0));
  auto owned_h = Omega_h::HostRead<Omega_h::I8>(oh_mesh->owned(0));

  const int nverts = oh_mesh->nverts();
  const int ncomps = nverts>0 ? tag_h.size()/nverts : 0;
  const int neq    = m_neq;
  const int nlev   = m_extruded_mesh->layers_data.cell.lid->numLayers + 1;
  const GO  nbasal = m_extruded_mesh->layers_data.node.gid->numHorizEntities;

  const int rank = m_comm->getRank();
  if (rank==0) {
    *out << "[knownValue] checking basal '" << tag_name << "' tag (" << context << "): "
         << ncomps << " comps/vertex, expecting neq*nlev = " << neq << "*" << nlev
         << " = " << neq*nlev << "\n";
  }
  if (ncomps != neq*nlev) {
    *out << "[knownValue] rank " << rank << ": *** tag has " << ncomps
         << " components, not neq*nlev = " << neq*nlev << "; cannot check slots.\n";
    return -1;
  }

  // Counts by failure mode. 'wrong vertex' means the value found belongs to a
  // DIFFERENT basal column than the vertex it sits on -- data crossed columns.
  // 'wrong level/eq' means the value belongs to this column but to another
  // (level,eq) slot -- data stayed in the column but was permuted vertically.
  long long n_checked=0, n_ok=0, n_wrong_vertex=0, n_wrong_slot=0, n_undecodable=0;
  int n_shown = 0;
  const int max_show = 8;

  for (int v=0; v<nverts; ++v) {
    if (not owned_h[v]) { continue; }
    const GO g = static_cast<GO>(gids_h[v]);
    for (int c=0; c<ncomps; ++c) {
      ++n_checked;
      const ST val = tag_h[v*ncomps+c];
      // Decode the value that is actually there.
      const long long ival = std::llround(val);
      if (std::abs(val-static_cast<ST>(ival)) > 1e-6 or ival < 0) {
        ++n_undecodable;
        if (n_shown<max_show) {
          ++n_shown;
          *out << "  [rank " << rank << "] vert gid " << g << " slot " << c
               << ": value " << val << " is not one of the encoded integers\n";
        }
        continue;
      }
      const long long got_eq  = ival/1000000;
      const long long rest    = ival%1000000;
      const long long got_lev = rest/nbasal;
      const long long got_g   = rest%nbasal;

      if (got_g != static_cast<long long>(g)) {
        ++n_wrong_vertex;
      } else if (got_eq>=neq or got_lev>=nlev) {
        // Right vertex, but the decoded (eq,level) is not even in range.
        ++n_wrong_slot;
      } else {
        // The value belongs to this vertex. The slot it sits in must be the one
        // ExtrudedMeshFieldAccessor::basal_cmp defines -- layer-major. (This used to
        // accept component-major too, while which packing saveVector produced was
        // still an open question; the convention is now fixed, so only it passes.)
        if (c == got_lev*neq + got_eq) { ++n_ok; }
        else                           { ++n_wrong_slot; }
      }

      if ((got_g!=static_cast<long long>(g) or c != got_lev*neq+got_eq)
          and n_shown<max_show) {
        ++n_shown;
        *out << "  [rank " << rank << "] vert gid " << g << " slot " << c
             << ": value " << val << " decodes to (eq " << got_eq
             << ", level " << got_lev << ", basal gid " << got_g << ")"
             << (got_g!=static_cast<long long>(g) ? "  <-- WRONG VERTEX"
                                                  : "  <-- WRONG SLOT")
             << " [layer-major slot would be " << (got_lev*neq+got_eq)
             << ", comp-major " << (got_eq*nlev+got_lev) << "]\n";
      }
    }
  }

  const long long n_bad = n_wrong_vertex + n_wrong_slot + n_undecodable;
  *out << "[knownValue] rank " << rank << " (" << context << "): checked "
       << n_checked << " owned slots: " << n_ok << " ok, "
       << n_wrong_vertex << " wrong vertex, " << n_wrong_slot << " wrong slot, "
       << n_undecodable << " undecodable\n";
  if (n_bad==0 and rank==0) {
    *out << "  => every owned tag slot holds the value its (vertex,level,eq) demands\n";
  }
  return static_cast<int>(n_bad);
#else
  (void) context;
  return 0;
#endif // ALBANY_OMEGAH
}

void ExtrudedDiscretization::
describeSolutionDofs (
    std::map<int,std::tuple<GO,int,std::vector<double>>>& lid_loc) const
{
  lid_loc.clear();

  const auto& layers_data = m_extruded_mesh->layers_data;
  const int   num_layers  = layers_data.cell.lid->numLayers;
  const int   mesh_dim    = getNumDim();

  const auto& dof_mgr       = getDOFManager();
  const auto& elem_dof_lids = dof_mgr->elem_dof_lids().host();
  const auto& node_indexer  = getNodeDOFManager()->ov_indexer();
  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elems   = basal_node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int num_basal_elems = basal_elems.size();

  for (int ibelem=0; ibelem<num_basal_elems; ++ibelem) {
    const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ibelem);
    const int num_basal_nodes = basal_node_gids.size();
    for (int ilay=0; ilay<num_layers; ++ilay) {
      const int ielem3d = layers_data.cell.lid->getId(ibelem,ilay);
      for (int eq=0; eq<m_neq; ++eq) {
        const auto& offsets = dof_mgr->getGIDFieldOffsets(eq);
        for (size_t k=0; k<offsets.size(); ++k) {
          const LO dof_lid = elem_dof_lids(ielem3d,offsets[k]);
          if (dof_lid<0 or lid_loc.count(dof_lid)) { continue; }

          const int iside = static_cast<int>(k)/num_basal_nodes;
          const int n     = static_cast<int>(k)%num_basal_nodes;
          const int ilev  = ilay + iside;
          const GO  bgid  = basal_node_gids[n];

          // 3d node coordinates, via the same route computeCoordinates uses.
          std::vector<double> xyz(mesh_dim,0.0);
          const GO ngid = layers_data.node.gid->getId(bgid,ilev);
          const LO nlid = node_indexer->getLocalElement(ngid);
          if (nlid>=0) {
            for (int d=0; d<mesh_dim; ++d) {
              xyz[d] = m_nodes_coordinates[mesh_dim*nlid + d];
            }
          }
          lid_loc.emplace(dof_lid,std::make_tuple(bgid,ilev,std::move(xyz)));
        }
      }
    }
  }
}

Teuchos::RCP<AdaptationData>
ExtrudedDiscretization::
checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                    const Teuchos::RCP<const Thyra_MultiVector>& dxdp)
{
  return m_basal_disc->checkForAdaptation(solution,solution_dot,solution_dotdot,dxdp);
}

void ExtrudedDiscretization::
adapt (const Teuchos::RCP<AdaptationData>& adaptData)
{
  // Adapt the basal mesh. This also runs m_basal_disc->updateMesh(), so the basal
  // disc is consistent with the new basal mesh when we return from here.
  m_basal_disc->adapt(adaptData);

  // The 3d layered numbering is built on top of the basal entity counts, which the
  // adaptation just changed. Refresh them before rebuilding anything that uses them.
  m_extruded_mesh->updateHorizEntityCounts();

  // Rebuild the extruded dof managers, worksets, node/side sets, graphs, ... so that
  // this disc (and its vector spaces) describe the adapted mesh. Without this, the
  // extruded disc would still advertise the pre-adaptation vector spaces.
  updateMesh();
}

// NOTE: the inverse of the permutation applied by writeSolutionToMeshDatabase.
// Both directions live in ExtrudedMeshFieldAccessor and share one walk, so they
// cannot drift apart.
Teuchos::RCP<Thyra_Vector>
ExtrudedDiscretization::getSolutionField(bool overlapped) const
{
  auto soln = Thyra::createMember(overlapped ? getOverlapVectorSpace()
                                             : getVectorSpace());
  soln->assign(0.0);
  m_extruded_mesh->get_extruded_field_accessor()
      ->fillLayeredSolution(*soln,solution_dof_name(),overlapped);
  return soln;
}

void
ExtrudedDiscretization::getField(
    Thyra_Vector& result,
    const std::string& name) const
{
  auto mfa = m_extruded_mesh->get_field_accessor();
  auto st = mfa->getNodalSIS().find(name);
  auto dof_mgr = getDOFManager(name);
  mfa->fillVector(result, name, dof_mgr, false);
}

void
ExtrudedDiscretization::getSolutionMV(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  const std::string names[3] = {
    solution_dof_name(),
    solution_dof_name()+"_dot",
    solution_dof_name()+"_dotdot"
  };
  auto mfa = m_extruded_mesh->get_extruded_field_accessor();
  for (int icol=0; icol<result.domain()->dim(); ++icol) {
    mfa->fillLayeredSolution(*result.col(icol),names[icol],overlapped);
  }
}

void
ExtrudedDiscretization::getSolutionDxDp(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  m_basal_disc->getSolutionDxDp(result,overlapped);
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void
ExtrudedDiscretization::setField(
    const Thyra_Vector& result,
    const std::string&  name,
    bool                overlapped)
{
  m_basal_disc->setField(result,name,overlapped);
}

void ExtrudedDiscretization::computeCoordinates ()
{
  m_nodes_coordinates.resize(getNumDim() * getLocalSubdim(getOverlapNodeVectorSpace()));

  const auto& layers_data = m_extruded_mesh->layers_data;

  const int num_layers = m_extruded_mesh->layers_data.cell.gid->numLayers;

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elem_lids = basal_node_dof_mgr->elem_dof_lids().host();
  const auto& basal_elems = basal_node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const auto& basal_coords = sideSetDiscretizations["basalside"]->getCoordinates();
  const auto& basal_mesh = m_extruded_mesh->basal_mesh();
  
  std::string thickness_name = m_disc_params->get<std::string>("Thickness Field Name","thickness");
  std::string surface_height_name = m_disc_params->get<std::string>("Surface Height Field Name","surface_height");

  auto surf_height = Thyra::createMember(basal_node_dof_mgr->ov_vs());
  auto thickness   = Thyra::createMember(basal_node_dof_mgr->ov_vs());
  basal_mesh->get_field_accessor()->fillVector(*surf_height,surface_height_name,basal_node_dof_mgr, true);
  basal_mesh->get_field_accessor()->fillVector(*thickness,thickness_name,basal_node_dof_mgr, true);
  auto s_h = getLocalData(surf_height.getConst());
  auto H = getLocalData(thickness.getConst());

  const auto node_indexer = getNodeDOFManager()->ov_indexer();
  const int num_basal_elems = basal_elems.size();
  const int npe_basal = basal_elem_lids.extent(1); // nodes-per-element
  const int basal_dim = m_basal_disc->getNumDim();
  const int mesh_dim = getNumDim();
  auto ni_vs = node_indexer->getVectorSpace();
  auto my_gids = getGlobalElements(ni_vs);

  int invalid_node_lid_count = 0;
  for (int ielem=0; ielem<num_basal_elems; ++ielem) {
    const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ielem);
    for (int node=0; node<npe_basal; ++node) {
      const int basal_node_lid = basal_elem_lids(ielem,node);
      const GO basal_node_gid  = basal_node_gids[node];
      double* bcoords = &basal_coords[basal_dim*basal_node_lid];

      for (int ilev=0; ilev<=num_layers; ++ilev) {
        // if (ilev!=num_layers)V
        // int elem3d = cell_layers_lid->getId(ielem,min(ilev
        // const auto& node_gids = node_dof_mgr->getElementGIDs(ielem);
        const GO node_gid = layers_data.node.gid->getId(basal_node_gid, ilev);
        const int node_lid = node_indexer->getLocalElement(node_gid);
        if (node_lid < 0) { ++invalid_node_lid_count; continue; }
        double* coords = &m_nodes_coordinates[mesh_dim*node_lid];

        for (int idim=0; idim<basal_dim; ++idim) {
          coords[idim] = bcoords[idim];
        }
        coords[basal_dim] = s_h[basal_node_lid] - H[basal_node_lid] * (1. - layers_data.z_ref[ilev]);
      }
    }
  }

  // Guard against invalid node LIDs (can happen at partition boundaries in parallel)
  if (invalid_node_lid_count > 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::runtime_error,
        "[ExtrudedDiscretization::computeCoordinates] " << invalid_node_lid_count
        << " node GIDs could not be mapped to local LIDs.\n"
        "This likely indicates a mismatch between the extruded and basal mesh partitioning.\n");
  }

  // DIAGNOSTIC: verify the 3d column geometry against the basal fields it is built
  // from. By the formula above, for each basal node:
  //   z(0)          = s - H            (bed)
  //   z(num_layers) = s                (surface, since z_ref.back()==1)
  //   z(l+1) > z(l) for H > 0          (monotone, non-degenerate column)
  // A column whose height does not match H, or which is inverted/collapsed, corrupts
  // every velocity gradient in it while leaving the solution values themselves intact.
  {
    long long n_cols = 0, n_bad_height = 0, n_non_monotone = 0, n_bad_bed = 0;
    double worst_height_err = 0, worst_H = 0;
    double min_H = std::numeric_limits<double>::max(), max_H = -min_H;
    double min_dz = std::numeric_limits<double>::max();

    for (int ielem=0; ielem<num_basal_elems; ++ielem) {
      const auto& basal_node_gids = basal_node_dof_mgr->getElementGIDs(ielem);
      for (int node=0; node<npe_basal; ++node) {
        const int basal_node_lid = basal_elem_lids(ielem,node);
        const GO  basal_node_gid = basal_node_gids[node];

        const double Hi = H[basal_node_lid];
        const double si = s_h[basal_node_lid];
        min_H = std::min(min_H,Hi);
        max_H = std::max(max_H,Hi);

        // Gather the column's z coordinates
        std::vector<double> z(num_layers+1,0.0);
        bool complete = true;
        for (int ilev=0; ilev<=num_layers; ++ilev) {
          const GO node_gid = layers_data.node.gid->getId(basal_node_gid, ilev);
          const int node_lid = node_indexer->getLocalElement(node_gid);
          if (node_lid<0) { complete = false; break; }
          z[ilev] = m_nodes_coordinates[mesh_dim*node_lid + basal_dim];
        }
        if (not complete) continue;
        ++n_cols;

        const double height = z[num_layers] - z[0];
        const double herr = std::abs(height - Hi);
        if (herr > 1e-8*std::max(1.0,std::abs(Hi))) {
          ++n_bad_height;
          if (herr > worst_height_err) { worst_height_err = herr; worst_H = Hi; }
        }
        if (std::abs(z[0] - (si - Hi)) > 1e-8*std::max(1.0,std::abs(si))) {
          ++n_bad_bed;
        }
        for (int ilev=0; ilev<num_layers; ++ilev) {
          const double dz = z[ilev+1]-z[ilev];
          min_dz = std::min(min_dz,dz);
          if (dz <= 0 and Hi > 0) { ++n_non_monotone; break; }
        }
      }
    }

    auto out = Teuchos::VerboseObjectBase::getDefaultOStream();
    *out << "[coords] 3d column geometry check:\n"
         << "  - columns checked      : " << n_cols << "\n"
         << "  - wrong column height  : " << n_bad_height
         << " (worst |z_top-z_bot - H| = " << worst_height_err
         << " at H = " << worst_H << ")\n"
         << "  - wrong bed elevation  : " << n_bad_bed << "\n"
         << "  - non-monotone columns : " << n_non_monotone << "\n"
         << "  - thickness range      : [" << min_H << ", " << max_H << "]\n"
         << "  - min layer spacing dz : " << min_dz << "\n";
  }

#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif
}

void ExtrudedDiscretization::createDOFManagers()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization:createDOFManagers");
  // NOTE: in Albany we use the mesh part name "" to refer to the whole mesh.
  //       That's not the name that stk uses for the whole mesh. So if the
  //       dof part name is "", we get the part stored in the stk mesh struct
  //       for the element block, where we REQUIRE that there is only ONE element block.

  Teuchos::RCP<DOFManager> dof_mgr;

  // Solution dof mgr
  dof_mgr  = create_dof_mgr("",solution_dof_name(),FE_Type::HGRAD,1,m_neq);
  m_dof_managers[solution_dof_name()][""] = dof_mgr;

  // Nodes dof mgr
  dof_mgr = create_dof_mgr("",nodes_dof_name(),FE_Type::HGRAD,1,1);
  m_dof_managers[nodes_dof_name()][""]    = dof_mgr;
  m_node_dof_managers[""]                 = dof_mgr;

  for (const auto& sis : m_extruded_mesh->get_field_accessor()->getNodalParameterSIS()) {
    const auto& dims = sis->dim;
    int dof_dim = -1;
    switch (dims.size()) {
      case 2: dof_dim = 1;               break;
      case 3: dof_dim = dims[2];         break;
      case 4: dof_dim = dims[2]*dims[3]; break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Error! Unsupported layout for nodal parameter '" + sis->name + ".\n");
    }

    dof_mgr = create_dof_mgr(sis->meshPart,sis->name,FE_Type::HGRAD,1,dof_dim);
    m_dof_managers[sis->name][sis->meshPart] = dof_mgr;

    dof_mgr = create_dof_mgr(sis->meshPart,sis->name,FE_Type::HGRAD,1,1);
    m_node_dof_managers[sis->meshPart] = dof_mgr;
  }
}

void
ExtrudedDiscretization::computeGraphs()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeGraphs");
  const auto vs = getVectorSpace();
  const auto ov_vs = getOverlapVectorSpace();
  m_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, vs, ov_vs, ov_vs));

  // Determine which equations are defined on the whole domain,
  // as well as what eqn are on each sideset
  std::vector<int> volumeEqns;
  std::map<std::string,std::vector<int>> ss_to_eqns;
  for (int k=0; k < m_neq; ++k) {
    if (m_sideSetEquations.find(k) == m_sideSetEquations.end()) {
      volumeEqns.push_back(k);
    }
  }
  const int numVolumeEqns = volumeEqns.size();

  // The global solution dof manager
  const auto sol_dof_mgr = getDOFManager();
  const int num_elems = sol_dof_mgr->cell_indexer()->getNumLocalElements();

  // Handle the simple case, and return immediately
  if (numVolumeEqns==m_neq) {
    // This is the easy case: couple everything with everything
    for (int icell=0; icell<num_elems; ++icell) {
      const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);
      m_jac_factory->insertGlobalIndices(elem_gids,elem_gids,true);
    }
    m_jac_factory->fillComplete();
    return;
  }

  // Ok, if we're here there is at least 1 side equation
  Teuchos::Array<GO> rows,cols;

  // First, couple global eqn (row) with global eqn (col)
  for (int icell=0; icell<num_elems; ++icell) {
    const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);

    for (int ieq=0; ieq<numVolumeEqns; ++ieq) {

      // Couple eqn=ieq with itself
      const auto& row_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(volumeEqns[ieq]);
      const int num_row_gids = row_gids_offsets.size();
      rows.resize(num_row_gids);
      for (int idof=0; idof<num_row_gids; ++idof) {
        rows[idof] = elem_gids[row_gids_offsets[idof]];
      }
      m_jac_factory->insertGlobalIndices(rows(),rows(),false);

      // Couple eqn=ieq with eqn=jeq!=ieq
      for (int jeq=0; jeq<numVolumeEqns; ++jeq) {
        const auto& col_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(jeq);
        const int num_col_gids = col_gids_offsets.size();
        cols.resize(num_col_gids);
        for (int jdof=0; jdof<num_col_gids; ++jdof) {
          cols[jdof] = elem_gids[col_gids_offsets[jdof]];
        }
        m_jac_factory->insertGlobalIndices(rows(),cols(),true);
      }
    }

    // While at it, for side set equations, set the diag entry, so that jac pattern
    // is for sure non-singular in the volume.
    for (const auto& it : m_sideSetEquations) {
      int eq = it.first;
      const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
      for (auto o : eq_offsets) {
        GO row = elem_gids[o];
        m_jac_factory->insertGlobalIndices(row,row,false);
      }
    }
  }

  // Now, process rows/cols corresponding to ss equations
  const auto& layers_data = m_extruded_mesh->layers_data;
  for (const auto& it : m_sideSetEquations) {
    const int side_eq = it.first;

    // If the side eqn is column-coupled, it needs special treatment.
    // A side eqn can be coupled to the whole column if
    //   1) the mesh is layered, AND
    //   2) all sidesets where it's defined are on the top or bottom
    int allowColumnCoupling = 1;
    for (const auto& ss_name : it.second) {
      SideStruct* side = nullptr;
      for (int ws=0; ws<getNumWorksets(); ++ws) {
        if (m_sideSets[ws].at(ss_name).size()>0) {
          side = &m_sideSets[ws].at(ss_name)[0];
        }
      }
      if (side==nullptr) {
        // This rank owns 0 sides on this sideset
        continue;
      }

      // Given any side of this sideSet, check layerId and pos within element,
      // to determine if we are on the top/bot of the mesh
      const auto pos = side->side_pos;
      const auto layer = layers_data.cell.gid->getLayerId(side->elem_GID);

      if (layer==(layers_data.cell.lid->numLayers-1)) {
        allowColumnCoupling = pos==layers_data.top_side_pos;
      } else if (layer==0) {
        allowColumnCoupling = pos==layers_data.bot_side_pos;
      } else {
        // This sideset is niether top nor bottom
        allowColumnCoupling = 0;
      }
    }
    // NOTE: Teuchos::reduceAll does not accept bool Packet, despite offerint REDUCE_AND as reduction op, so use int
    int globalAllowColumnCoupling = allowColumnCoupling;
    Teuchos::reduceAll(*m_comm,Teuchos::REDUCE_AND,1,&allowColumnCoupling,&globalAllowColumnCoupling);

    // Loop over all side sets where this eqn is defined
    for (const auto& ss_name : it.second) {
      for (int ws=0; ws<getNumWorksets(); ++ws) {
        const auto& elem_lids = getElementLIDs_host(ws);
        const auto& ss = m_sideSets[ws].at(ss_name);

        // Loop over all sides in this side set
        for (const auto& side : ss) {
          const LO ws_elem_idx = side.ws_elem_idx;
          const LO elem_LID = elem_lids(ws_elem_idx);
          const auto& side_elem_gids = sol_dof_mgr->getElementGIDs(elem_LID);
          const int side_pos = side.side_pos;
          const auto& side_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(side_eq,side_pos);

          // Compute row GIDs
          const int num_row_gids = side_eq_offsets.size();
          rows.resize(num_row_gids);
          for (int idof=0; idof<num_row_gids; ++idof) {
            rows[idof] = side_elem_gids[side_eq_offsets[idof]];
          }

          if (globalAllowColumnCoupling) {
            // Assume the worst, and couple with all eqns over the whole column
            const int numLayers = layers_data.cell.lid->numLayers;
            const LO basal_elem_LID = layers_data.cell.lid->getColumnId(elem_LID);
            for (int eq=0; eq<m_neq; ++eq) {
              const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
              const int num_col_gids = eq_offsets.size();
              cols.resize(num_col_gids);
              for (int il=0; il<numLayers; ++il) {
                const LO layer_elem_lid = layers_data.cell.lid->getId(basal_elem_LID,il);
                const auto& elem_gids = sol_dof_mgr->getElementGIDs(layer_elem_lid);

                for (int jdof=0; jdof<num_col_gids; ++jdof) {
                  cols[jdof] = elem_gids[eq_offsets[jdof]];
                }
                m_jac_factory->insertGlobalIndices(rows(),cols(),true);
              }
            }
          } else {

            // Add local coupling (on this side) with all eqns
            // NOTE: we could be fancier, and couple only with volume eqn or side eqn that are defined
            //       on this side set. However, if a sideset is a subset of another, we might miss the
            //       coupling since the side sets have different names. We'd have to inspect if a ss is
            //       contained in the other, but that starts to get too involved. Given that it's not
            //       a common scenario (need 2+ ss eqn defined on 2 different sidesets), and that we
            //       might have to redo this when we assemble by blocks, we just don't bother.
            for (int col_eq=0; col_eq<m_neq; ++col_eq) {
              const auto& col_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(col_eq,side_pos);
              const int num_col_gids = col_eq_offsets.size();
              cols.resize(num_col_gids);
              for (int jdof=0; jdof<num_col_gids; ++jdof) {
                cols[jdof] = side_elem_gids[col_eq_offsets[jdof]];
              }

              m_jac_factory->insertGlobalIndices(rows(),cols(),true);
            }
          }
        }
      }
    }
  }

  m_jac_factory->fillComplete();
}

void
ExtrudedDiscretization::computeWorksetInfo()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeWorksetInfo");

  const int num_elems = m_extruded_mesh->get_num_local_elements();
  const int ws_size = m_extruded_mesh->meshSpecs[0]->worksetSize;
  const int num_ws  = (num_elems + ws_size - 1) / ws_size;

  m_workset_sizes.resize(num_ws);
  m_workset_elements = DualView<int**>("ws_elem",num_ws,ws_size);
  for (int ws=0,lid=0; ws<num_ws; ++ws) {
    // For the last ws, we may have less elems.
    int this_ws_size = ws==(num_ws-1) ? num_elems-ws*ws_size : ws_size;
    m_workset_sizes[ws] = this_ws_size;
    for (int ie=0; ie<this_ws_size; ++ie, ++lid) {
      m_workset_elements.host()(ws,ie) = lid;
    }
    // Fill the remainder (if any) with very invalid numbers
    for (int ie=this_ws_size; ie<ws_size; ++ie) {
      m_workset_elements.host()(ws,ie) = -1;
    }
  }
  m_workset_elements.sync_to_dev();

  // For now, everything has the same element block name, and same phys index
  m_wsEBNames.resize(num_ws,m_extruded_mesh->meshSpecs[0]->ebName);
  m_wsPhysIndex.resize(num_ws,0);

  // Clear elem_LID->wsIdx index map if remeshing
  m_elem_ws_idx.clear();
  auto cell_indexer = getDOFManager()->cell_indexer();
  m_elem_ws_idx.resize(num_elems);

  m_ws_elem_coords.resize(num_ws);
  const auto& elem_gids = getDOFManager()->getAlbanyConnManager()->getElementsInBlock();
  const auto node_dof_mgr = getNodeDOFManager();
  const auto elem_node_lids = node_dof_mgr->elem_dof_lids().host();
  const int num_nodes = elem_node_lids.extent(1);
  const int num_dim = getNumDim();
  for (int ws = 0; ws < num_ws; ws++) {
    m_ws_elem_coords[ws].resize(m_workset_sizes[ws]);

    for (int ie=0; ie<m_workset_sizes[ws]; ++ie) {
      const int elem_lid = m_workset_elements.host()(ws,ie);
      m_elem_ws_idx[elem_lid].ws = ws;
      m_elem_ws_idx[elem_lid].idx = ie;

      m_ws_elem_coords[ws][ie].resize(num_nodes);
      for (int in=0; in<num_nodes; ++in) {
        const int node_lid = elem_node_lids(elem_lid,in);
        TEUCHOS_TEST_FOR_EXCEPTION(node_lid < 0 || node_lid >= (int)(m_nodes_coordinates.size()/num_dim),
            std::runtime_error,
            "[ExtrudedDiscretization::computeWorksetInfo] Invalid node_lid=" << node_lid
            << " for elem_lid=" << elem_lid << ", node=" << in
            << " (valid range [0," << m_nodes_coordinates.size()/num_dim-1 << "])\n");
        m_ws_elem_coords[ws][ie][in] = &m_nodes_coordinates[num_dim*node_lid];
      }
    }
  }

  // TODO: tell field accessor to init states
  m_extruded_mesh->get_field_accessor()->createStateArrays(m_workset_sizes);
  m_extruded_mesh->get_field_accessor()->transferNodeStatesToElemStates();
  m_extruded_mesh->get_extruded_field_accessor()->setWorksetElements(m_workset_elements);
  m_extruded_mesh->get_extruded_field_accessor()->setElemWorksetIdx(m_elem_ws_idx);

  // Give the accessor what it needs to permute the 3d solution into the basal
  // per-column layout. This must happen HERE (not in setFieldData): the dof managers
  // below are rebuilt by every updateMesh, including the one that follows adaptation,
  // so a pointer captured earlier would go stale exactly when the mesh changes.
  m_extruded_mesh->get_extruded_field_accessor()->setLayeredSolutionInfo(
      m_extruded_mesh->layers_data.node.lid,
      m_basal_disc->getNodeDOFManager(),
      m_basal_disc->getDOFManager(),
      getDOFManager(),
      m_neq);

  // Extrude/interpolate basal fields
  const auto& extrude_names = m_disc_params->get<Teuchos::Array<std::string>>("Extrude Basal Fields",{});
  const auto& interpolate_names = m_disc_params->get<Teuchos::Array<std::string>>("Interpolate Basal Layered Fields",{});
  auto emfa = Teuchos::rcp_dynamic_cast<ExtrudedMeshFieldAccessor>(m_extruded_mesh->get_field_accessor());
  emfa->extrudeBasalFields(extrude_names);
  emfa->interpolateBasalLayeredFields(interpolate_names);
}

void
ExtrudedDiscretization::computeSideSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeSideSets");

  // NOTE: the convention for ordering the mesh sides GIDs is
  // - basal side
  // - upper side
  // - lateral side

  // Clean up existing sideset structure (in case we are remeshing)
  m_sideSets.clear();

  int num_ws = getNumWorksets();
  m_sideSets.resize(num_ws);  // Need a sideset list per workset
  Teuchos::Array<GO> side_GIDs;

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_cell_indexer = m_basal_disc->getDOFManager()->cell_indexer();
  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& cell_indexer = node_dof_mgr->cell_indexer();
  const int num_glb_basal_elems = basal_cell_indexer->getNumGlobalElements();
  const auto& layers_data = m_extruded_mesh->layers_data;
  const auto& extr_conn_mgr = Teuchos::rcp_dynamic_cast<ExtrudedConnManager>(getNodeDOFManager()->getAlbanyConnManager(),true);
  for (const auto& ss : m_extruded_mesh->meshSpecs[0]->ssNames) {
    // Make sure the sideset exist even if no sides are owned on this process
    for (int i=0; i<num_ws; ++i) {
      m_sideSets[i][ss].resize(0);
    }

    if (ss=="basalside" or ss=="upperside") {
      // Side sets are just the basal mesh elems
      const int num_sides = basal_cell_indexer->getNumLocalElements();
      side_GIDs.reserve(side_GIDs.size()+num_sides);
      for (int iside=0; iside<num_sides; ++iside) {
        SideStruct sStruct;

        const GO basal_gid = basal_cell_indexer->getGlobalElement(iside);
        const int ilayer = ss=="basalside" ? 0 : layers_data.cell.gid->numLayers-1;

        sStruct.elem_GID = layers_data.cell.gid->getId(basal_gid,ilayer);
        sStruct.side_GID = basal_gid + (ss=="upperside" ? num_glb_basal_elems : 0);
        side_GIDs.push_back(sStruct.side_GID);

        auto elem_LID = cell_indexer->getLocalElement(sStruct.elem_GID);
        sStruct.ws_elem_idx = m_elem_ws_idx[elem_LID].idx;

        // Get the ws that this element lives in
        int workset = m_elem_ws_idx[elem_LID].ws;

        // Save the position of the side within element (0-based).
        sStruct.side_pos = ss=="basalside" ? layers_data.bot_side_pos : layers_data.top_side_pos;

        // Save the index of the element block that this elem lives in
        sStruct.elem_ebIndex = m_extruded_mesh->meshSpecs[0]->ebNameToIndex[m_wsEBNames[workset]];

        // Get or create the vector of side structs for this side set on this workset
        auto& ss_vec = m_sideSets[workset][ss];
        ss_vec.push_back(sStruct);
      }
    } else {
      std::vector<std::string> basal_ss_names;
      if (ss=="lateralside") {
        // Extrude all sideSets from the basal mesh
        basal_ss_names = m_extruded_mesh->basal_mesh()->meshSpecs[0]->ssNames;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION (ss.substr(0,9)!="extruded_", std::runtime_error,
            "Error! Unexpected value for side set name.\n"
            "  - ss name: " + ss + "\n"
            "  - supported values: basalside, upperside, lateralside, extruded_*\n");
        basal_ss_names.push_back(m_extruded_mesh->get_basal_part_name(ss));
      }

      // First, figure out the largest basal side GID (so we can build a proper LayeredMeshNumbering)
      GO max_basal_side_GID = -1;
      for (int ws=0; ws<m_basal_disc->getNumWorksets(); ++ws) {
        for (const auto& basal_ssn : basal_ss_names) {
          auto basal_ss = m_basal_disc->getSideSets(ws).at(basal_ssn);
          for (const auto& side : basal_ss) {
            max_basal_side_GID = std::max(max_basal_side_GID,side.side_GID);
          }
        }
      }

      LayeredMeshNumbering<GO> side_layers_gid (max_basal_side_GID,layers_data.cell.gid->numLayers,layers_data.cell.gid->ordering);
      auto get_basal_side_nodes = [&](const SideStruct& basal_side) {
        std::vector<GO> nodes;
        const int belem_LID = basal_cell_indexer->getLocalElement(basal_side.elem_GID);
        const auto& belem_nodes = basal_node_dof_mgr->getElementGIDs(belem_LID);
        const auto& offsets = basal_node_dof_mgr->getGIDFieldOffsetsSide(0,basal_side.side_pos);
        for (auto o : offsets) {
          nodes.push_back(belem_nodes[o]);
        }
        return nodes;
      };

      auto determine_side_pos = [&] (const GO elem_GID, std::vector<GO> basal_side_nodes) {
        const int num_sides = node_dof_mgr->get_topology().getSideCount();
        const int elem_LID = cell_indexer->getLocalElement(elem_GID);
        const auto& elem_nodes = node_dof_mgr->getElementGIDs(elem_LID);
        int pos = -1;
        std::vector<GO> side_nodes;
        GO ilay = layers_data.cell.gid->getLayerId(elem_GID);
        for (auto bn : basal_side_nodes) {
          side_nodes.push_back(layers_data.node.gid->getId(bn,ilay));
        }
        for (auto bn : basal_side_nodes) {
          side_nodes.push_back(layers_data.node.gid->getId(bn,ilay+1));
        }
        for (int iside=0; iside<num_sides and pos==-1; ++iside) {
          const auto& offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,iside);
          pos = iside;
          for (auto o : offsets) {
            if (std::find(side_nodes.begin(),side_nodes.end(),elem_nodes[o])==side_nodes.end()) {
              pos = -1; break;
            }
          }
        }
        TEUCHOS_TEST_FOR_EXCEPTION (pos==-1, std::runtime_error,
            "Error! Could not locate side inside an element.\n"
            " - side nodes gids: " + util::join(side_nodes,",") + "\n"
            " - elem nodes gids: " + util::join(elem_nodes,",") + "\n");
        return pos;
      };
      // Track added side GIDs to avoid duplicates. This matters for "lateralside",
      // which is built from all basal side sets. If those side sets overlap (e.g.,
      // boundary_side_set contains boundary_side_set_1 and boundary_side_set_2),
      // iterating all of them would add the same 3D face multiple times.
      std::set<GO> added_side_GIDs;
      for (int ws=0; ws<m_basal_disc->getNumWorksets(); ++ws) {
        for (const auto& basal_ssn : basal_ss_names) {
          auto basal_ss = m_basal_disc->getSideSets(ws).at(basal_ssn);
          for (const auto& basal_side : basal_ss) {
            const auto basal_elem_gid = basal_side.elem_GID;
            const auto basal_side_nodes_gids = get_basal_side_nodes(basal_side);
            for (int ilev=0; ilev<layers_data.cell.gid->numLayers; ++ilev) {
              SideStruct sStruct;
              sStruct.elem_GID = layers_data.cell.gid->getId(basal_elem_gid,ilev);
              sStruct.side_GID = 2*num_glb_basal_elems + side_layers_gid.getId(basal_side.side_GID,ilev);

              // Skip if this side was already added (can happen when basal side sets overlap)
              if (!added_side_GIDs.insert(sStruct.side_GID).second) continue;

              auto elem_LID = cell_indexer->getLocalElement(sStruct.elem_GID);
              sStruct.ws_elem_idx = m_elem_ws_idx[elem_LID].idx;
              side_GIDs.push_back(sStruct.side_GID);

              // Get the ws that this element lives in
              int workset = m_elem_ws_idx[elem_LID].ws;

              // Save the position of the side within element (0-based).
              sStruct.side_pos = determine_side_pos(sStruct.elem_GID,basal_side_nodes_gids);

              // Save the index of the element block that this elem lives in
              sStruct.elem_ebIndex = m_extruded_mesh->meshSpecs[0]->ebNameToIndex[m_wsEBNames[workset]];

              // Get or create the vector of side structs for this side set on this workset
              auto& ss_vec = m_sideSets[workset][ss];
              ss_vec.push_back(sStruct);
            }
          }
        }
      }
    }
  }
  auto vs = createVectorSpace(m_comm,side_GIDs);
  m_sides_indexer = createGlobalLocalIndexer(vs);

  buildSideSetsViews();
}

void
ExtrudedDiscretization::computeNodeSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: computeNodeSets");

  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& node_conn_mgr = node_dof_mgr->getAlbanyConnManager();
  const auto& node_dof_lids = node_dof_mgr->elem_dof_lids().host();
  const int num_elems = m_extruded_mesh->get_num_local_elements();
  const int mesh_dim = getNumDim();

  // Loop over all node sets
  for (const auto& ns : m_extruded_mesh->meshSpecs[0]->nsNames) {
    auto& ns_gids     = m_nodeSetGIDs[ns];
    auto& ns_elem_pos = m_nodeSets[ns];
    auto& ns_coords   = m_nodeSetCoords[ns];

    ns_gids.clear();
    ns_elem_pos.clear();
    ns_coords.clear();

    // Get the mask for this nodeset from the conn mgr, and count how many nodes are in it
    auto mask = node_conn_mgr->getConnectivityMask(ns);

    std::set<GO> gids_found;
    for (int ie=0; ie<num_elems; ++ie) {
      const auto& node_gids = node_dof_mgr->getElementGIDs(ie);
      const int conn_start = node_conn_mgr->getConnectivityStart(ie);
      const int conn_size  = node_conn_mgr->getConnectivitySize(ie);
      const auto ownership = node_conn_mgr->getOwnership(ie);
      for (int in=0; in<conn_size; ++in) {
        if (mask[conn_start+in]==1 and ownership[in]==Owned) {
          auto it_bool = gids_found.insert(node_gids[in]);
          if (it_bool.second) {
            // Newly processed node
            ns_gids.push_back(node_gids[in]);
            ns_elem_pos.push_back(std::make_pair(ie,in));
            const int node_lid = node_dof_lids(ie,in);
            ns_coords.push_back(&m_nodes_coordinates[mesh_dim*node_lid]);
          }
        }
      }
    }
  }
}

void
ExtrudedDiscretization::buildSideSetProjectors()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: buildSideSetProjectors");
  std::cout << "WARNING! ExtrudedDiscretization::buildSideSetProjectors not yet implemented!\n";
  return;
}

void ExtrudedDiscretization::printCoords() const 
{
  const int nnodes = m_extruded_mesh->get_num_local_nodes();
  const int ndim   = getNumDim();
  std::cout << "coordinates on processor " << m_comm->getRank() << "/" << m_comm->getSize() << "\n";
  for (int inode=0; inode<nnodes; ++inode) {
    GO node_gid = getNodeDOFManager()->ov_indexer()->getGlobalElement(inode);
    std::cout << "  node_GID=" << node_gid << ", coords=";
    for (int idim=0; idim<ndim; ++idim) {
      std::cout << " " << m_nodes_coordinates[inode*3+idim];
    }
    std::cout << "\n";
  }
}

void
ExtrudedDiscretization::updateMesh()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: updateMesh");

  // Make sure we don't reuse old dof mgrs (if adapting). create_dof_mgr returns any
  // cached dof mgr matching the requested specs, so without this the extruded disc
  // would keep the ones built for the pre-adaptation mesh.
  m_key_to_dof_mgr.clear();
  m_dof_managers.clear();
  m_node_dof_managers.clear();

  // First, make sure the basal disc is updated
  m_basal_disc->updateMesh();

  createDOFManagers();

  computeCoordinates();

  setupMLCoords();

  computeWorksetInfo();

  computeSideSets();

  computeNodeSets();

  computeGraphs();

  buildCellSideNodeNumerationMaps();

  if (sideSetDiscretizations.size()>0) {
    buildSideSetProjectors();
  }
}

void ExtrudedDiscretization::
buildCellSideNodeNumerationMaps()
{
  // The numeration is simple, since we decided the side gids
  const auto& node_dof_mgr = getNodeDOFManager();

  const auto& basal_node_dof_mgr = m_basal_disc->getNodeDOFManager();
  const auto& basal_elem_gids = m_basal_disc->getNodeDOFManager()->getAlbanyConnManager()->getElementsInBlock();

  const auto& layers_data = m_extruded_mesh->layers_data;

  std::vector<GO> side_nodes;

  // ONLY for basalside and upperside, since that's where we are likely to load data from mesh
  for (int ws=0; ws<getNumWorksets(); ++ws) {
    for (std::string ssn : {"basalside","upperside"}) {
      auto& s2ssc = m_side_to_ss_cell[ssn];
      auto& s2nn = m_side_nodes_to_ss_cell_nodes[ssn];

      for (const auto& s : m_sideSets[ws][ssn]) {
        const GO basal_elem_GID = s2ssc[s.side_GID] = layers_data.cell.gid->getColumnId(s.elem_GID);
        const LO basal_elem_LID = basal_node_dof_mgr->cell_indexer()->getLocalElement(basal_elem_GID);
        const auto elem_LID = node_dof_mgr->cell_indexer()->getLocalElement(s.elem_GID);
        const auto& elem_nodes = node_dof_mgr->getElementGIDs(elem_LID);
        const auto& basal_nodes = basal_node_dof_mgr->getElementGIDs(basal_elem_LID);
        const auto& offsets = node_dof_mgr->getGIDFieldOffsetsSide(0,s.side_pos);

        // Retrieve the gids of the basal mesh nodes that generated the gids of this side
        // NOTE: if ordering==LAYER, they are the same
        side_nodes.resize(offsets.size());
        for (size_t i=0; i<offsets.size(); ++i) {
          auto gid3d = elem_nodes[offsets[i]];
          side_nodes[i] = layers_data.node.gid->getColumnId(gid3d);
        }
        s2nn[s.side_GID].resize(offsets.size());
        for (size_t i=0; i<offsets.size(); ++i) {
          auto it = std::find(side_nodes.begin(),side_nodes.end(),basal_nodes[i]);
          TEUCHOS_TEST_FOR_EXCEPTION (it==side_nodes.end(), std::runtime_error,
              "Error! Could not locate node in the basal mesh.\n");

          s2nn[s.side_GID][i] = std::distance(side_nodes.begin(),it);
        }
      }
    }
  }
}

void ExtrudedDiscretization::setFieldData()
{
  TEUCHOS_FUNC_TIME_MONITOR("ExtrudedDiscretization: setFieldData");

  m_basal_disc->setFieldData();

  const auto basal_sol_mfa = m_basal_disc->get_solution_mesh_field_accessor();
  const auto elem_numbering_lid = m_extruded_mesh->layers_data.cell.lid;
  m_solution_mfa = Teuchos::rcp(new ExtrudedMeshFieldAccessor(basal_sol_mfa,elem_numbering_lid));
}

Teuchos::RCP<ConnManager>
ExtrudedDiscretization::create_conn_mgr (const std::string& part_name)
{
  auto basal_part_name = m_extruded_mesh->get_basal_part_name(part_name);
  auto conn_mgr_h = m_basal_disc->create_conn_mgr(basal_part_name);
  return Teuchos::rcp(new ExtrudedConnManager(conn_mgr_h,m_extruded_mesh));
}

Teuchos::RCP<DOFManager>
ExtrudedDiscretization::
create_dof_mgr (const std::string& part_name,
                const std::string& field_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim)
{
  auto& dof_mgr = get_dof_mgr(part_name,fe_type,order,dof_dim);
  if (Teuchos::nonnull(dof_mgr)) {
    // Not the first time we build a DOFManager for a field with these specs
    return dof_mgr;
  }

  const auto& ebn = m_extruded_mesh->meshSpecs()[0]->ebName;;
  std::vector<std::string> elem_blocks =  {ebn};

  // Create conn manager
  auto conn_mgr = create_conn_mgr(part_name);
  auto conn_mgr_h = Teuchos::rcp_dynamic_cast<ExtrudedConnManager>(conn_mgr)->get_basal_conn_mgr();

  // Create dof mgr
  dof_mgr  = Teuchos::rcp(new DOFManager(conn_mgr,m_comm,part_name));

  const auto& topo = conn_mgr->get_topology();
  Teuchos::RCP<panzer::FieldPattern> fp;
  if (topo.getName()==std::string("Particle")) {
    // ODE equations are defined on a Particle geometry, where Intrepid2 doesn't work.
    fp = Teuchos::rcp(new panzer::ElemFieldPattern(shards::CellTopology(topo)));
  } else {
    // For space-dependent equations, we rely on Intrepid2 for patterns
    const auto basis = getIntrepid2TensorBasis(*conn_mgr_h->get_topology().getCellTopologyData(),1);
    fp = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  }
  // NOTE: we add $dof_dim copies of the field pattern to the dof mgr,
  //       and call the fields cmp_n, n=0,..,$dof_dim-1
  for (int i=0; i<dof_dim; ++i) {
    dof_mgr->addField("cmp " + std::to_string(i),fp);
  }

  dof_mgr->build();

  return dof_mgr;
}

}  // namespace Albany
