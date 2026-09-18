# Diverging `||F||` after Omega_h mesh adaptation — investigation notes

Branch: `cws/adaptive-humboldt-test`
Test: `tests/landIce/FO_GIS`, `input_fo_humboldt_transient_omegah_refine.yaml`
Config: `NumLayers: 5` (6 node levels), 3 equations, Humboldt, serial.

Claude Code, using Opus 5, were used to debug the the problem and apply fixes.

## Refine-only vs. general adapt

The general adapt path drives Omega_h from the MeshFields SPR error indicator:
`isos_from_lengths` on the `tgtLength` vertex tag, clamped by "Minimum/Maximum Edge
Length", then `grade_fix_adapt`. That both refines and coarsens, so field values are
genuinely lost where elements collapse, and a wrong post-adaptation solution cannot be
distinguished from information legitimately discarded by coarsening.

`Refine Only` (`a892dd955`) exists to remove that ambiguity. It ignores the SPR size
field and calls `refine_by_size` against the mesh's implied metric, with the threshold
from "Refine Only Max Length". Refinement only *adds* vertices: every vertex of the old
mesh survives carrying its exact values, and each new vertex is linearly interpolated
along the edge it splits. A correct field transfer must therefore leave the solution
essentially unchanged and the residual close to its pre-adaptation value — so any large
jump in `||F||` is a transfer or rebuild bug, not lost information.

All results below are from `Refine Only`, which is why the residual jump is a defect
rather than an expected consequence of coarsening.

## Symptom

The pre-adaptation transient step converges normally:

```
-- Nonlinear Solver Step 8 --
||F|| = 1.666e-06  step = 1.000e+00  dx = 2.914e-04 (Converged!)
```

Immediately after `disc->adapt(...)` (2611 -> 4918 tris), the next step starts at

```
-- Nonlinear Solver Step 0 --
||F|| = 5.581e+05  step = 0.000e+00  dx = 0.000e+00
```

and Newton then **diverges monotonically** while accepting full steps:

| step | `\|\|F\|\|` | `dx` |
|---|---|---|
| 0 | 5.58e5 | 0 |
| 4 | 1.06e6 | 2.12e4 |
| 5 | 1.29e6 | 4.11e4 |
| 6 | 1.64e6 | — |

`||F||` at step 0 is evaluated at the transferred solution, before any Newton step, so it
is the only residual in the sequence that is independent of the bad search directions.

## Summary

Instrumentation confirmed that everything feeding the residual is correct after
adaptation: the solution transfer is bitwise exact on all 1425 surviving vertices, the
2D→3D read-back writes every dof, the node and side sets rebuild with consistent counts
and topology, the 3D column geometry matches the basal thickness and surface exactly,
the extruded and interpolated 3D element states are consistent down every column, the
Jacobian is well formed with dominant off-diagonal coupling, and no evaluator holds
stale cached state. Despite that, the residual is already ~1e6 coming out of the volume
fill — the Dirichlet BCs only reduce it — and the error is confined to the two velocity
equations and spread uniformly over the mesh rather than concentrated on a few dofs.

Next step: rather than test more candidate inputs, localize the error spatially by
dumping per-element residual contributions with element coordinates after adapt — if the
large contributions cluster near refined regions, workset boundaries, or the basal
boundary, that identifies the cause without having to guess it first.

## Commits

Fixes (keep):

| Commit | Summary |
|---|---|
| `bb73e05a3` | fix evaluator caches that break on field manager re-setup |
| `c94c7d78d` | fix overflow in `checkDerivatives` perturbation |
| `b1e4c54bf` | re-setup all field managers after adapt, not just the response ones |
| `0c72c6a38` | clear state arrays before resizing on re-mesh, warn on dropped tags |
| `a892dd955` | add `Refine Only` adapt mode |

Debug instrumentation (drop or gate before merging):

| Commit | Summary |
|---|---|
| `666476bf5` | per-phase residual and jacobian breakdown |
| `a83bca36f` | verify field transfer and set rebuild across adapt |
| `a224bce21` | verify extruded column geometry and 3d state assembly |

## Bugs found and fixed

These were real defects that blocked progress; all are unrelated to the residual itself.

1. **`LandIce_GatherVerticallyContractedSolution_Def.hpp:115`** (`bb73e05a3`) — `quadWeights.resize()`
   aborted on the second `postRegistrationSetup`. `refreshFieldManagers()` re-runs
   `postRegistrationSetup` on the *same* evaluator objects, and
   `Albany::DualView::resize` asserts `d_view.size()==0`. Guarded with a `size()==0`
   check, matching the existing pattern at `PHAL_GatherSolution_Def.hpp:235` and
   `PHAL_ScatterResidual_Def.hpp:129`.
   `numLayers` and `dz_ref` are invariant across adaptation (set once from the
   `NumLayers` input parameter in the `ExtrudedMesh` ctor / `setBulkData`), so the
   cached weights stay valid — no invalidation needed.

2. **`PHAL_SDirichletField_Def.hpp:124`** (`bb73e05a3`) — ASan heap-buffer-overflow. The
   `if (not col_is_dbc_.is_null()) return;` cache was keyed on nothing, so the
   post-adapt Jacobian (larger column space) was indexed with a vector built for the
   old one. Now also requires `isCompatible` on both the column and range spaces.
   Note the two sibling evaluators (`PHAL_SDirichlet_Def.hpp:127`,
   `PHAL_ExprEvalSDBC_Def.hpp:107`) rebuild unconditionally — the cache was the anomaly.

3. **`Albany_Application.cpp:1064`** (`checkDerivatives`, `c94c7d78d`) — pre-existing bug, unrelated
   to adaptation. The perturbation was `randomize(-rmax, +rmax)`, then rescaled with
   `xd = 2*xd - 1`, which overflows to `±inf`. Reported `reldif = -inf` /
   "should be on the order of inf". Changed to `randomize(0.0, 1.0)`, which is what the
   rescale expects. **Caveat:** even fixed, this check is not usable here — it is called
   at the end of `computeGlobalJacobianImpl` (post-Dirichlet) but computes its finite
   difference via `computeGlobalResidual`, which with SDBCs modifies `x` through
   `dfm->preEvaluate`. It reports `reldif ~0.2` on the *converged* pre-adapt state, so
   it is measuring something other than `J` vs `dF`.

## What was tested and cleared

Every item below was verified with instrumentation, with a pre-adapt baseline for
comparison where possible.

| Area | Method | Result |
|---|---|---|
| Solution transfer | Marker tag + coordinate match to identify true survivors | **1425/1425 survivors, 0 changed, worst diff exactly 0** |
| `fillVector` (2D→3D solution) | Count distinct LIDs written vs vector size | 47808/47808 distinct, max lid 47807, writes = 14754×18 = valence×ncomps |
| Component packing | `ncomps` vs dof manager | 18 = 3 eq × 6 node levels, consistent |
| Solution norms | `\|x\|`, `\|xdot\|` across adapt | ratio 1.3550 vs √(dof ratio) 1.3652; RMS/dof 46.37→46.03; `xdot/x` preserved to 7 digits |
| Node/side sets | Marked-entity counts before/after rebuild | `bss_1+bss_2 == bss` edges both sides; closed loop (V==E) and open arcs (V==E+1) preserved; uniform 1.63–1.69× growth |
| Side-set projectors | Traced consumers | `projectors`/`ov_projectors` are used **only** in `writeSolutionToMeshDatabase` (`Albany_STKDiscretization.cpp:617,646,678,707`). The `ExtrudedDiscretization` stub is irrelevant to assembly. |
| `sliding_velocity` zeroing | Input file + evaluator | `Field Usage: Output` (yaml:160); recomputed as `PHAL::FieldFrobeniusNorm` (`LandIce_StokesFOBase.hpp:1500`). Harmless. |
| Jacobian structure | Per-phase nnz / diag vs offdiag | `\|\|offdiag\|\|` 1.32e6 vs `\|\|diag\|\|` 8.56e5; 0 all-zero rows; ~33 nnz/row; graph bit-identical across Newton steps |
| 1-iteration GMRES | Compared pre-adapt | **Also 1 iteration pre-adapt** (where Newton converges) — normal for this config, not a signal |
| 3D column geometry | `z_top-z_bot` vs `H`, `z_bot` vs `s-H`, monotonicity | 0 bad height, 0 bad bed, 0 non-monotone; thickness range `[0.001, 2.53262]` **identical** pre/post |
| Basal input fields | VTK before/after (by CWS) | `temperature` (8 cmp), `velocity`, `basal_friction`, `ice_thickness`, `surface_height`, `bed_topography` all present and matching |
| 3D state derivation | Stacked-face column continuity | **0 mismatched** of 31332 (pre) / 59016 (post) per scalar field; 62664/118032 for `velocity`. Coverage = ntri×4 pairs×3 nodes exactly. |
| Evaluator stale caches | Swept all one-shot cache patterns | `Neumann`/`GatherSolution`/`ScatterResidual` offsets are reference-element topology only; `MDFieldMemoizer` **disabled** (`Use MDField Memoization` not set); `PHAL_Setup` field bookkeeping all gated on `_enableMemoization`; `computeSideDOFOffsets` declared but never defined/called |

### Key result: the residual breakdown

Added a per-phase, per-equation residual report (`Albany_Application.cpp`, in
`computeGlobalResidualImpl`). At step 0 post-adapt:

```
[resid] after volume+neumann fill: ||F|| = 9.90467009e+05 over 47808 local dofs
    eq 0 (cmp 0): ||F_eq|| = 6.27473589e+05 over 15936 dofs
    eq 1 (cmp 1): ||F_eq|| = 7.66356178e+05 over 15936 dofs
    eq 2 (cmp 2): ||F_eq|| = 6.52862794e-01 over 15936 dofs
[resid] after dirichlet bcs: ||F|| = 5.58056253e+05 over 47808 local dofs
```

Three things follow:

- **The Dirichlet BCs are not the cause** — they *reduce* `||F||` (9.90e5 → 5.58e5), as
  expected from zeroing rows. The residual is already ~1e6 out of the volume fill.
- **Only the velocity equations are affected.** eq 2 is ~6 orders of magnitude smaller.
  Whatever is wrong is specific to eqs 0/1 and hits them roughly equally.
- **The error is spread, not localized.** Worst single entry 1.4e5 against a 9.9e5 total
  over 15936 dofs; RMS/dof ~5e3 uniformly. Not a handful of bad nodes.

`rows diag-only: 13280` in the Jacobian report is *structure*, not damage:
13280 = 2656 basal verts × 5 = eq 2 having a diagonal-only row at 5 of 6 levels.

## Where this leaves it

Newton accepting full steps into monotonic divergence is consistent with `J` being a
correct derivative of an `F` that itself encodes wrong physics — i.e. `J` and `F` agree
with each other and are both wrong in the same way, which points at a shared input the
checks above do not reach.

**Untested surface (in suggested priority order):**

1. **A spatial localization pass** — dump per-element residual contributions with
   element coordinates after adapt. If the large contributions cluster (refined regions,
   partition/workset boundaries, basal boundary) that identifies the cause without
   needing to guess it first. Hypothesis-free, and the recommended next step.
2. **Workset connectivity/coordinate arrays** the Phalanx evaluators index —
   `loadWorksetBucketInfo`, `m_ws_elem_coords`, `wsElNodeEqID`. The column-continuity
   check validated the *state arrays* but not these.
3. **Side-set views** — `sideSetViews` / `localDOFViews` from `buildSideSetsViews`, and
   `allLocalDOFViews` in `Albany_AbstractDiscretization.cpp:220-250` (sized on
   `numLayers` and side-set counts, rebuilt each adapt). Internally consistent but
   possibly mismatched against the new worksets. Relevant because the basal friction
   BC for Stokes-FO evaluates through these.

Also worth noting: `ExtrudedDiscretization::buildSideSetProjectors` and
`fillSolnVector`/`fillSolnMultiVector`/`fillSolnSensitivity` in
`ExtrudedMeshFieldAccessor` are `NotYetImplemented`. None is on the residual path
(`OmegahDiscretization::getSolutionMV` bypasses `fillSolnMultiVector` and calls
`fillVector` per column), but they will matter for side-set output and restart.

## Diagnostics added (all debug-only, intended to be removed or gated)

| Commit | File | What |
|---|---|---|
| `666476bf5` | `Albany_Application.cpp` | Per-phase/per-equation residual breakdown + worst entries; Jacobian structure report (nnz, diag vs offdiag, empty/diag-only rows). Gated on `res_debug_count`, re-opened by `Albany::reset_residual_debug()` (declared in the header) which the observer calls after `adapt`. A plain absolute cap does not work — the pre-adapt solve exhausts it. |
| `666476bf5` | `Albany_PiroTempusObserver.cpp` | Solution norm + dof count across the adapt; calls `reset_residual_debug()`. |
| `a83bca36f` | `Albany_OmegahDiscretization.cpp` | `adapt_probe_id` vertex marker (stamped **before** the `before_adapt` VTK write so it appears in both files, registered `OMEGA_H_LINEAR_INTERP`); node/side set entity counts before and after rebuild; extra `after_adapt_sets<N>.vtk` written **after** `createNodeSets`/`createSideSets` (the existing `after_adapt<N>.vtk` is written earlier and structurally cannot contain the set tags). |
| `a83bca36f` | `Albany_OmegahMeshFieldAccessor.cpp/.hpp` | `fillVector` write-coverage counter; `probe_record`/`probe_compare` survivor round-trip check (marker + coordinates). |
| `a224bce21` | `Albany_ExtrudedDiscretization.cpp` | 3D column geometry check at the end of `computeCoordinates`. |
| `a224bce21` | `Albany_ExtrudedMeshFieldAccessor.cpp/.hpp` | `checkColumnContinuity` — stacked-face consistency, called from both `extrudeBasalFields` and `interpolateBasalLayeredFields`. |

Two of these are cheap and worth keeping (behind a flag): the per-equation residual
breakdown, and `checkColumnContinuity`.

## Methodological notes / traps hit

- **`Omega_h::GO` global ids are renumbered by adaptation.** They cannot be used to
  follow a vertex across an adapt. A marker tag transferred with the mesh is required.
- **A `LINEAR_INTERP` marker is not injective.** A vertex splitting an edge between
  markers `k-1` and `k+1` gets exactly `k`, colliding with the survivor that legitimately
  carries `k`. In this mesh **250 of 1425** vertices needed the coordinate cross-check to
  break such a tie. An early "confirmed transfer bug" was exactly this artifact — worth
  knowing if anyone repeats the technique.
- `class_id`/`class_dim` are hardcoded to always inherit in Omega_h
  (`Omega_h_transfer.cpp:24-26`), independent of `xfer_opts`, which is why rebuilding the
  sets from `mark_by_class` after adapt is sound.
- `basal_offset` in `extrudeBasalFields`/`interpolateBasalLayeredFields` assumes basal
  workset `ws` occupies a contiguous, in-order range of global basal element LIDs. This
  is undocumented and is the kind of assumption adaptation could break. The column
  continuity check passing is indirect evidence it holds, but it was not tested directly.
- The `temperature` file supplies 8 node levels **including both endpoints**
  (`_NLC = [0, 0.053, 0.120, 0.207, 0.320, 0.473, 0.688, 1]`), while the mesh has 6
  uniformly spaced levels. Only `z=0` and `z=1` coincide, so genuine interpolation is
  required. `z_ref`/`dz_ref` and `_NLC` are all invariant across adaptation, so the
  interpolation *weights* are provably unchanged — only the addressing changes.
