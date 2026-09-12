# Parameters — authoritative table

**Generated from `src/plantsim/config.py` by `src/scripts/07_parameter_table.py`. Do not edit by hand.**

Any parameter value quoted anywhere else — in the paper, in slides, in `Physics_and_Code_Description.md` — must match this table. If it does not, this table is right, because it is what the code actually ran.

| Symbol | Value | Unit | Basis |
|---|---:|---|---|
| g | 0, -9.81, 0 | m/s² | standard gravity |
| L | 0.85 | m | row crop at working height |
| n | 16 | – | converged at this resolution; see VALIDATION.md |
| r_base | 5.0 | mm | calliper-measurable; sets EI at the clamp |
| r_tip | 2.0 | mm | taper gives EI ∝ r⁴ falloff with height |
| E_stem | 1.00 | GPa | green herbaceous stem, published range ≈0.3–5 GPa; values below ≈0.2 GPa cannot support the stem's own weight (see material sanity check) |
| ρ_stem | 700 | kg/m³ | fresh tissue, mostly water |
| Cd_stem | 1.10 | – | bluff cylinder in cross-flow |
| E_leaf | 0.10 | GPa | lamina ≈10× more compliant than stem |
| L_leaf | 0.140 | m | blade length |
| w_leaf | 0.050 | m | sets drag area |
| Cd_leaf | 1.40 | – | flat plate normal to flow |
| U | 4.0 | m/s | moderate breeze at reference height |
| α | 0.20 | – | power-law boundary layer over cropland |
| ρ_air | 1.225 | kg/m³ | ISA sea level |
| Δt | 0.01667 | s | 60 Hz frame |
| n_sub | 32 | – | converged; see convergence study |
| n_iter | 4 | – | converged; see convergence study |
| c | 1.20 | 1/s | ζ≈0.1 at ~1.5 Hz; rate, not per-frame factor |

## Derived quantities

These are not inputs — they follow from the table above.

| Quantity | Value | How |
|---|---:|---|
| Flexural rigidity at base | 0.4909 N·m² | EI, I = πr⁴/4 |
| Flexural rigidity at tip | 0.0126 N·m² | EI, I = πr⁴/4 |
| Base/tip rigidity ratio | 23× | from the r⁴ taper |
| Plant mass | 34.3 g | ρ·A·L, lumped at nodes |
| Particles per plant | 25 | stem + leaves + soil anchor |
| Constraints per plant | 46 | axial + bending |
| Solver substep h | 0.521 ms | Δt / n_sub |
