# Selective suppression of the lift-up and of the Orr energy transfer

The production filter ([production_filter.md](production_filter.md)) multiplies the production of
the columns with `k_h < prod_kc` by a weight `G(k_h)`.  The production of a Fourier column has two
mechanisms, and the filter acts on both with the same weight.  This extension gives each its own:

| mechanism | in the equations | weight |
| --- | --- | --- |
| lift-up: the tilting of the mean vorticity by `v`, the source of `eta` | `(S2 i alfa - S i beta) D0 v` in the `eta` equation | `glu = prod_lu * G(k_h)` |
| Orr energy transfer: the tilting of the wavevector, the part of the `d2v` equation that is left when the lift-up is gone | `-(1 - gorr) (i alfa S + i beta S2) D1 v` added to the `d2v` equation | `gorr = prod_orr * G(k_h)` |

With the defaults (`prod_lu = prod_orr = 1`) the code is the production filter as it was.  With
`prod_kc = 0` (`G = 1`), `prod_lu = 0` removes the lift-up everywhere (classical SS-HST otherwise),
`prod_orr = 0` the Orr transfer.  The study this serves (switch-off runs from a saved field of the
reference run, ensembles, spectral maps) is the run protocol of sections 6 and 7 below; what is
implemented and tested is sections 1 to 5, and the ensemble tool of 7.1.

Numbering follows the implementation guide this was written from; the code refers to the sections
below.

## 1. Deck parameters

`&production`, in addition to `prod_kc` and `prod_edge`:

| parameter | meaning | default |
| --- | --- | --- |
| `prod_lu` | `lambda_LU`, weight of the lift-up | 1 |
| `prod_orr` | `lambda_Orr`, weight of the Orr energy transfer | 1 |

Values outside `[0, 1]` are accepted (negative reverses a transfer, above 1 amplifies it) with a
warning.  The run header prints the weights when either differs from 1, and the number of columns
with a weight below 1 (over all ranks, of `(nx + 1)(2 nz + 1) - 1`, the `(0, 0)` column not counted).

```fortran
! no lift-up, classical SS-HST otherwise
&production  prod_kc = 0.0, prod_edge = 0.0, prod_lu = 0.0, prod_orr = 1.0 /

! no Orr energy transfer, in the production-filtered box
&production  prod_kc = 3.6276, prod_edge = 0.3, prod_lu = 1.0, prod_orr = 0.0 /
```

## 2. Weight arrays

`hst_production.f90` builds `glu = prod_lu*gprod` and `gorr = prod_orr*gprod` (module `hst_params`,
same shape and index convention as `gprod`) once, and maps them to the device with `gprod`.
`gprod` stays: the statistics of the production filter use it for the forced and the unforced
range.

## 3. Equations

In `buildrhs_prepare` (`hst_equations.f90`), per column:

* `eta`: the lift-up source is multiplied by `glu(iz, ix)` (it was `gprod`);
* `d2v`: `expl = -g*unkn - (1 - gorr(iz, ix)) (S i alfa + S2 i beta) D1 v` (it was `1 - gprod`).
  `expl` is stored in `oldrhs`, so the next substep carries it with `ODE(3)`.  The large-scale
  damping term `-g*unkn` stays as it was; both terms use the `S2` of the substep.

The `(0, 0)` column is untouched (`alfa = beta = 0` makes both terms vanish).

## 4. Run-time statistics: `statistics/split_runtime.dat`

Written at `dt_stat` for every run without a Stokes layer (a new file starts with a `#` header line;
a restart appends).  Per unit volume, as `Runtimedata`:

| column | quantity |
| --- | --- |
| 1 | time |
| 2, 3 | `E_tor`, `E_pol` |
| 4, 5 | `P_LU`, `P_Orr`: the full mechanisms (unweighted) |
| 6, 7 | `gP_LU = sum glu P_LU`, `gP_Orr = sum gorr P_Orr`: what enters the budget |
| 8 | `diss` (as `Runtimedata`) |
| 9, 10 | `P_LU` over the columns with `k_par = 0`, `P_Orr` over those with `k_perp = 0` |
| 11 | `E_00`, the part of column 2 that is the `(0, 0)` column |

With `a = S alfa + S2 beta`, `b = S beta - S2 alfa` (`|(S, S2)| k_par` and `|(S, S2)| k_perp`),
`kh^2 = alfa^2 + beta^2`, and the amplitudes `u, v, w` of a column with `k_h > 0`:

```
P_Orr = -(a/kh^2) Re <(alfa u + beta w) v*>      P_LU = -(b/kh^2) Re <(beta u - alfa w) v*>
E_tor = <|beta u - alfa w|^2>/(2 kh^2)           E_pol = <|u|^2 + |v|^2 + |w|^2>/2 - E_tor
```

(continuity gives `D v = -i alfa u - i beta w`, so no derivative is needed), summed with the weights
of the other statistics (`ix > 0` twice, `dyl` in `y`).  Then

```
P_LU + P_Orr = -S <uv> - S2 <vw>                   (the production of Runtimedata)
d/dt (E_tor + E_pol) = gP_LU + gP_Orr - diss       (no large-scale damping, no Stokes layer)
```

Two choices the guide does not make:

* **The `(0, 0)` column.**  Its horizontal velocity has no `k_h` to split with.  It is the `k_h -> 0`
  limit of the toroidal part and enters `E_tor` with its fluctuation about the box mean (the
  fluctuation of `Runtimedata`), so that `E_tor + E_pol = energy` holds to round-off; column 11 is
  that part.
* **`k_par = 0`, `k_perp = 0` (columns 9, 10).**  Columns where one mechanism acts alone
  (`P_Orr` vanishes at `a = 0`, `P_LU` at `b = 0`).  The condition is `|a| <= 1e-9 |(S, S2)| k_h`
  (`|b|` for the other): for `S2 = 0` the columns `ix = 0` and `iz = 0`; with `S2 /= 0` only the
  columns that are exactly aligned with the shear (a lattice coincidence), so these two columns
  carry information for `S2 = 0` only.

The sums are made by one separate kernel (`split_sums`), so the existing statistics are not touched.
Spectral maps of `P_LU`, `P_Orr`, `E_tor`, `E_pol` over `(kx, kz)` (guide section 4.3) are not
written: they are to be made from saved fields in post-processing.

## 5. Tests

All of them in `tests/run_tests.sh`: 58 runs (the 34 that existed and 24 new), all passing on the
CPU build (gfortran, 4 min) and on the GPU build (nvfortran 25.9, GTX 1650, `cc75`, 1 rank);
`tests/regression.sh` has two more decks (five, at 1e-10, on both).  The multi-GPU and NCCL builds are
not tested.

### 5.1 Kelvin modes

`test_kelvin_s2` takes the weights as `prod_lu G`, `prod_orr G` **computed from the deck** (the
smoothstep written out in the test), not read from `glu` and `gorr`, and fails if the solver's arrays
differ from them.  With the arrays as the reference the test could not see a construction error in
`hst_production.f90` (the filter is left out of `glu`: all 19 decks passed until this was found by
the seeded-bug audit, 5.5).  For any `S2(t)`

```
v(t)   = v0 (k0^2/k^2)^((1 + g_Orr)/2) E(t)
eta(t) = E(t) [eta0 - i g_LU int (S kz - S2 kx) (k0^2/k^2)^((1 + g_Orr)/2) dt]
```

`tests/kelvin_ref.py` is the independent reference: `Mode` has `g_lu` and `g_orr`; `Mode.ode` (and
`reference_split`) integrates the `(v, eta, ky)` equations of the guide with DOP853 instead of the
closed form (`--selftest`: agreement 1e-10 or better for all the weights below).

Decks `tests/decks/kelvin_split_<mode>_<weights>_<S2>.in`, mode `a` (`kx = 0`) and `b`
(`kx = 2 pi/15`, `kz = 2 pi/5`), weights `(g_LU, g_Orr)` = `lu0orr1` (0, 1), `lu1orr0` (1, 0), `mix`
(0.5, 0.3), `S2` oscillating (`S T = 5.25`, 10 and 4 periods, 8 samples per period) or constant
(`6` and `8` time units): 12 decks.  Also:

| deck | what |
| --- | --- |
| `kelvin_split_a_mix_filter`, `b_mix_filter` | the weights 0.5, 0.3 times a smooth filter (`G` = 0.417 and 0.222): `g` is `lambda G`, not `lambda` |
| `kelvin_split_b_outside` | `prod_lu = -0.5`, `prod_orr = 1.4` (the warning; a reversed lift-up, an amplified Orr transfer) |
| `kelvin_split_b_zero_osc` | `prod_lu = prod_orr = 0` without the filter: no energy gain |

All 19 pass at `tol = 1e-5` (`5e-5` for the three order decks of 5.2; largest error 2.7e-6, in the energy of
`a_lu1orr0_osc`); the four decks
with `csv` (`a_mix_osc`, `b_mix_osc`, `a_mix_filter`, `b_mix_filter`) are also checked by
`kelvin_ref.py --compare`.

### 5.2 Order of accuracy

`order_a_orr0`: `kx = 0`, `prod_orr = 0` (the extra `d2v` term alone), `v`; `order_b_eta`: `kx /= 0`,
weights (1, 0.5), `eta`; `order_b_eta_g547`: (0.547, 0.547), `eta`.  `nu = 1e-12`, `dt` = `T/32`, `T/64`,
`T/128` (`S T = 1.75`, two periods), `ny = 256`.  New `&kelvin` options of `test_kelvin_s2`:
`order_qty = 'v' | 'eta'` and `order_fine_min` (the criterion on the finest pair only, 2.8).
Observed order of the finest pair: 3.01, 2.95, 2.96.

`ny = 256`, not 128, for the `eta` decks: at `ny = 128` the finest pair gives 2.48 (a spatial error of
about 1e-8 at the finest level); at 256 it is 2.95 and at 512 2.99.  `kelvin_checks.py order` also
takes `--glu`, `--gorr`, `--qty eta` and `--eta0` for runs made outside the suite.

### 5.3 Budget in a nonlinear run

`split_budget_ny64.in` and `_ny128.in` (`15 x ny x 15`, `Re = 200`, `S2 = 0.3 + 0.5 sin(pi t)`,
weights 0.5, 0.3 times the filter `kc = 6`, `edge = 0.3`, `exact_shift`), checked by
`tests/split_budget.py`:

| check | result |
| --- | --- |
| `E_tor + E_pol = energy` | 4e-14 (relative) |
| `P_LU + P_Orr = -S uv - S2 vw` | 3e-14 (relative to the peak) |
| budget residual / max `|dE/dt|` | 7.9e-4 at `ny = 64`, 1.4e-4 at `ny = 128` (ratio 5.7) |
| the budget without the weights (`sum P_LU + sum P_Orr - diss`) misses by | 165 and 943 times as much |

Two things the guide did not expect:

* **`exact_shift = .true.` is needed.**  In a linear run with the default shift the residual is
  `O(dy^2)` (9.2e-3, 2.2e-3, 5.5e-4 at `ny` = 32, 64, 128) and is the same for the unweighted,
  unfiltered solver (8.3e-3, 2.0e-3, 4.9e-4): it is the `D0` weighting error documented at
  `shear_shift`, not the weights.  With `exact_shift` it is 2.6e-4, 1.7e-5, 3e-6 (weights 0.5, 0.3).
* **The residual does not depend on `dt`.**  At `ny = 64` it is 1.68e-4, 1.81e-4, 1.85e-4 for
  `deltat` = 0.004, 0.002, 0.001 (`dt_stat` = 5 steps): it is the discretisation in `y`, which is why the
  test requires it to fall with `ny`, and not with the time step.

### 5.4 Regression

* **GPU: bit for bit.**  The final field of `small_prod.in` (the production filter with the weights
  at 1) is identical, byte for byte, to that of the binary built from `production_filter` before the
  change.  The run-time statistics differ at 1e-16, as between two runs of the old binary (the
  reductions on the device are not deterministic).
* **CPU: not bit for bit, and cannot be.**  The FFTW plans are made with `FFTW_MEASURE`: two pairs of
  runs of the old binary differ by 0 and 7.4e-15 (relative), the old binary against the new by
  7.5e-15 to 8.6e-15, the same level.  `tests/regression.sh` compares at 1e-10.
* `tests/reference/small_prod.fld` is from the **old** binary (the same deck without `prod_lu` and
  `prod_orr`); `small_split.fld` is from this code and guards later changes only.
* **`prod_kc = 0`, default weights, against `main`** (binary built from `main`; decks `small` and
  `small_s2x`, no `&production` group): the final fields are bitwise identical on the GPU and differ by
  5e-15 (relative) on the CPU, the planning noise.  The 34 runs that existed pass unchanged.

`tests/test_split_stats.f90` checks every column of `split_runtime.dat` against a reference summed
the other way round (box mean subtracted from the field first, plain loops, `S2 = 0` and `S2 = 0.3`) to
1e-14, and on 1, 2 and 4 ranks (x-z pencils and y slabs) the file agrees with one rank to 3e-13.

### 5.5 Seeded bugs

`tests/split_bug_audit.py` plants each bug by exact text replacement in a scratch copy, builds,
runs the 19 Kelvin decks (or the budget run), and gives the margin = largest error / tolerance:

| bug | detected by | margin |
| --- | --- | --- |
| `glu`, `gorr` swapped | 17 of 19 decks (`a_lu0orr1_osc`: eta) | 1.6e7 |
| `(1 - gorr)` written as `gorr` | 18 of 19 | 5.4e5 |
| extra term not stored in `oldrhs` | 15 of 19 and the order decks | 1.0e5; order 0.0 |
| `der(iy, 0, j)` for `der(iy, 1, j)` | 15 of 19 | 5.4e5 |
| `s2now` missing in the extra `d2v` term | 15 of 19 | 2.7e5 |
| `s2now` missing in the `eta` source | 8 of 19 (`b_lu1orr0_const`) | 9.6e4 |
| `glu = prod_lu` (no `G`) | the two filter decks | 9.8e5 |
| `gorr = prod_orr` (no `G`) | the two filter decks | 3.8e4 |
| `eta` source uses `gorr` | 17 of 19 | 2.5e7 |
| statistics: `gP_LU` with `gorr` | budget | 12.6 |
| statistics: `P_Orr` sign | identity of `P` | 1e11 |
| statistics: `(0, 0)` column left out of `E_tor` | identity of `E` | 3e9 |
| statistics: `P_LU` with `a` for `b` | identity of `P` | 5e10 |

* A term missing from `oldrhs` is not a first-order scheme, as the guide expects: the term is then
  integrated with the wrong total weight, the scheme is inconsistent, and the observed order is 0.
* `S1` (a wrong weight in the sum `gP_LU`, `gorr` for `glu`, 0.3 against 0.5) is the least visible
  bug of all, with 12.6 over the budget tolerance.

The table is from the CPU build, and is the audit run on the final tree.  The device mapping of `glu` and `gorr` (the last row of the guide's
table) needs the GPU build; two planted variants, run on the GTX 1650:

| variant | result |
| --- | --- |
| `glu`, `gorr` not mapped at all | **passes**: the arrays are mapped implicitly at every kernel launch, so the numbers are right and only the speed would suffer; no test can see it |
| mapped before they are filled (a stale device copy) | fails on every deck tried: errors 0.06 to 1 in `v`, `eta`, energy (the CPU build passes) |

so a stale mapping is caught by running the suite on the GPU, as the guide expects, and a missing one is
not a correctness problem with this compiler.

## 6. Run protocol

Not part of the code: reference run (3:2:1, `Re_z = 1000`, `S2 = 0`, to `S t = 2000`, a field every
`Delta (S t) = 50` after 200), switch-off runs restarted from it with only `&production` changed
(first RK substep does not use `oldrhs`: changing the weights at a restart is clean; during a run
only at a step boundary), `split_runtime.dat` and fields at `Delta (S t)` = 0, 5, 10, 20, 50, 100, 200,
300 after the switch; one pilot realisation per set, 8 and then 16.  Nothing of it has been run.

## 7. Post-processing

`python3 -m post.python ensemble RUN1 RUN2 ... --tau-max 300 --dtau 1 -o ens.dat` aligns runs on
`tau = t - t_switch`, interpolates onto a common grid and writes the mean and the standard error of
`q2`, `diss`, `uv`, `vw`, the anisotropy `b_ij` and the split quantities `E_tor`, `E_pol`, `P_LU`,
`P_Orr`, `gP_LU`, `gP_Orr` (`--selftest`: three synthetic runs of known mean).  `S*` is not
written (its normalisation is a convention).  The maps of 7.2 (over `(k_par, k_perp)`) need the snapshots
and are not made.

## 8. Extensions (not done)

Band-shaped weights (replace the scalars in `production_setup`; the kernels do not change), a
ramp `lambda(t)` (the weights become scalars times `gprod` in the kernels), separate weights on the
`S` and `S2` parts of each term, the Stokes layer (not combined with the weights).
