# Passive native Yee energy instrument

Enable with `warpx.native_yee_energy_balance = 1`. The optional
`warpx.native_yee_energy_balance_path` defaults to
`diags/native_yee_energy_balance.csv`. The instrument refuses an existing output
path: each restart segment needs a fresh path. Accumulators start at zero for
each process invocation, including checkpoint restarts. The first row describes
the first completed step, with its actual starting inventories; there is no
invented pre-restart flux or work.

This is a passive explicit-evolution hook, not a ReducedDiag. It does not enter
`DoDiags`, synchronize momenta, alter diagnostic cadence, mutate evolving fields,
or filter deposited current. It copies native fields/current and reduces scalars.
The supported contract is Cartesian 3D, one level, vacuum explicit staggered Yee,
periodic or exterior-PML field faces and periodic or zero-reflection absorbing
particle faces. Unsupported solvers, moving/boosted/Galilean frames, refinement,
subcycling, divergence cleaning, EB, fluids, lasers, current centering and
in-domain/particle PML are rejected. Dynamic grid decomposition is rejected.

The products sum physical cell volumes, averaging **products** on four native E
edges or two native B faces. Curl E is forward; curl B is backward. Each face
uses E at its native tangential edge and B averaged over the two normal cell
centers to that edge, then trapezoidal integration in the nodal tangent. All six
outward signed faces are retained, including cancelling periodic faces. Physical
guard values come from the actual stages. Only the private midpoint E copy gets
periodic/interbox `FillBoundary`; no physical BC or PML operation is applied to
that copy.

Raw J is copied after deposition and its deposition guards are summed on the
copy with `WarpXSumGuardCells` and `get_ng_depos_J()`. It is never filtered. The
E-update midpoint is used for both `W` (actual used J) and `Wraw` (unfiltered J).
`W_processing = W-Wraw` explains current processing; it is not a second ledger
term. Positive W removes field energy. Native particle push work is compared
with W descriptively; their different time centering is not an exact particle
energy theorem.

The CSV has a named header sorted lexically. All `_J` quantities are joules,
`*_s` quantities seconds, `weight_*` quantities sums of physical-particle
weights, and `step` the completed one-based step. Each transfer/residual/absolute
scale has a `cum_` counterpart, summed since this segment started. Inventories,
timestamps, `domain_cells` and `real_epsilon` have no cumulative counterpart.

| Columns | Definition |
|---|---|
| `UE_<stage>_J`, `UB_<stage>_J`, `U_<stage>_J` | Native E/B/total energy at `start`, `after_B_first`, `after_B_first_exchange`, `after_E`, `after_E_exchange`, `after_B_second`, `end` |
| `CB_first_J`, `CB_second_J` | `-dt/2 <Bbar,curl E>/mu0` for the actual first/second B kicks |
| `CE_J`, `A_J` | `dt <Ebar,curl Bhalf>/mu0`, `dt <Bhalf,curl Ebar>/mu0` |
| `W_J`, `Wraw_J`, `W_processing_J` | Used-current work, unfiltered-current work, their difference |
| `F_xlo_J` through `F_zhi_J`, `F_J` | Independently evaluated six outward face transfers and their sum |
| `T_J` | Explicit split-stage correction `A+CB_first+CB_second` |
| `G_B_first_exchange_J`, `G_E_exchange_J`, `G_final_exchange_J` | Changes of domain field energy across the named guard/PML exchange stages |
| `G_gap_J` | Current field start minus preceding recorded field end; zero on the first row |
| `G_J` | Sum of the three within-step exchanges and `G_gap` |
| `delta_U_solve_J` | `U_end-U_start` |
| `delta_U_J` | `U_end-previous_U_end`; first row uses `U_start` instead of previous end |
| `delta_U_B_first_J`, `delta_U_E_J`, `delta_U_B_second_J` | Actual energy changes across the individual field pushes, before exchanges |
| `residual_field_update_J` | `delta_U+W-CE-CB_first-CB_second-G` |
| `residual_face_identity_J` | `CE-A+F` |
| `residual_combined_J` | `delta_U+W+F-T-G`; equals the preceding two residuals summed up to floating-point roundoff |
| `residual_B_first_J`, `residual_E_J`, `residual_B_second_J` | Individual actual-update changes minus their directly measured update products |
| `<product>_abs_product_J` | Absolute elementary-product sum for CB, CE, A, W, Wraw and every face; curl input operands are counted **before** cancellation |
| `field_update_absolute_scale_J` | Sum of all positive U snapshots, previous end U when available, and absolute W/CE/CB product scales |
| `face_identity_absolute_scale_J` | Sum of absolute CE/A/F product scales |
| `K_<stage>_J`, `weight_<stage>` | Native particle KE and weight before/after push and before/after boundary handling |
| `K_push_work_J` | Post-push minus pre-push native particle KE |
| `K_gap_J` | Pre-push KE minus previous post-boundary KE; zero on first row; includes intervening synchronization/unsynchronization |
| `K_push_to_boundary_gap_J` | Pre-boundary minus post-push KE |
| `K_outgoing_J`, `weight_outgoing` | Independent pre-deletion sum over particles outside any absorbing face; corner exits counted once, periodic faces excluded |
| `residual_particle_boundary_J` | Pre-boundary minus post-boundary KE minus independent outgoing KE |
| `residual_particle_weight_boundary` | Analogous particle-weight boundary residual |
| `particle_grid_mismatch_J` | `K_push_work-W`; descriptive coupling/time-centering comparison |

Particle K inventories/work/gaps/outgoing/boundary residuals additionally have
species columns with the species name before `_J`. Inventory and outgoing
weights also have species columns. The outgoing predicate matches
`ParticleBoundaries_K.H`: position strictly below the lower or above the upper
physical face. K uses the same native momenta, mass and stable relativistic
kinetic-energy routine as the existing particle-energy reduction, but the
outgoing sum is selected by positions independently of inventory differences.

CSV output flushes after boundary handling and before existing end-step
momentum synchronization. No exterior PML-field inventory is claimed: G records
exchange into the declared physical-domain inventory. A closed field budget
does not establish correct absorption or correct particle/grid coupling.

The repository regression is `Examples/Tests/native_yee_energy_balance`: a
16-step, periodic, source-free 16-cubed Yee wave with eight 8-cubed boxes. It
uses two MPI ranks when MPI is enabled, otherwise one process. No OpenPMD or
field checksum is required. The standalone NumPy analyzer reconstructs the
stage/exchange/face identities from primitive columns, retains the nonzero
split correction, checks cumulative columns, and compares original energy and
Poynting diagnostic cadence. Floating-point checks use a conservative
`gamma_(500*Ncells)` envelope on absolute primitive operands and positive
inventory scales, not relative errors of cancelled terms.

After configuring with `BUILD_TESTING=ON` and a Python interpreter containing
NumPy, run:

```sh
ctest --test-dir build-energy-cpu -R '^test_3d_native_yee_energy_balance\.' --output-on-failure
```

The CTest setup removes only its own previous native CSV in its dedicated
test directory, allowing repeat test invocations without weakening production
segment overwrite protection. Disabled/enabled passivity, restart equivalence,
and particle-exit behavior remain separate qualification checks.
