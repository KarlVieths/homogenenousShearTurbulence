# Kelvin mode in an oscillating spanwise shear, with the production filter

Test of the linearised solver, `tests/test_kelvin_s2.f90` (2026-11-16, branch
`production_filter`).  It checks, column by column and against exact
solutions, that the code

1. tilts the wavevector correctly when S2 depends on time (the shear-periodic
   shift with the displacement gamma_z = int S2 dt);
2. applies the filtered source terms (the eta lift-up times G, and the extra
   term -(1 - G)(i alfa S + i beta S2) D1 v in the d2v equation) with the S2 of
   every Runge-Kutta substep;
3. lets a fully masked column (G = 0) gain no energy from the shear.

It extends `test_kelvin` (one end time, v only for S2(t), no eta for
0 < G < 1) to time series over many S2 periods.  The test specification is
`kelvin_test_unsteady_S2.md` (not in the repository); everything it needs is
below.

## The exact solution

U = (S y, 0, S2(t) y), y from the plane U = 0 (in the code y runs over
[0, L_y) and U = S y, so no offset).  For one column (k_x, k_z) with
k_h^2 = k_x^2 + k_z^2 and production weight G (constant along the trajectory
since k_h is):

    k_y(t)  = k_y0 - k_x S t - k_z gamma_z(t),            gamma_z = int S2 dt
    v(t)    = v0 (k0^2/k^2)^((1 + G)/2) E(t),             E = exp(-nu int k^2 dt)
    eta(t)  = E [eta0 - i G v0 int (S k_z - S2 k_x)(k0^2/k^2)^((1 + G)/2) dt]
    |u|^2   = (k^2 |v|^2 + |eta|^2)/k_h^2                 (= |u0|^2 E^2 for G = 0)

for any S2(t).  It follows from d(u)/dt = -G v (S, 0, S2) - i k p - nu k^2 u,
k.u = 0 (so i p = -(1 + G) a v/k^2 with a = S k_x + S2 k_z and a k_y = -(1/2)
dk^2/dt).  For k_x = 0 and S2 = A sin(omega t) the k^2 integral is closed
(b = k_y0 - c, c = k_z A/omega):
int k^2 = (k_z^2 + b^2) t + 2 c b sin(omega t)/omega + c^2 (t/2 + sin(2 omega t)/(4 omega)).

## What is in the repository

| file | |
|---|---|
| `tests/test_kelvin_s2.f90` | the test (one rank, linear).  Reads the deck, plus a `&kelvin` group: `ix_mode`, `iz_mode`, `iy_mode` (k_y0 = 2 pi iy_mode/L_y at t = 0), `v0`/`eta0` as (re, im), `dt_out` (default S2 period/8), `refine`, `order_min`, `order_fine_min` (the criterion on the finest pair only), `order_qty` (`'v'` or `'eta'`: the quantity whose order is taken), `tol`, `csv`.  The weights of the column are `prod_lu G` and `prod_orr G`, computed from the deck, and the solver's arrays must equal them (the lift-up / Orr split: [../doc.production/selective_lu_orr.md](../doc.production/selective_lu_orr.md), decks `kelvin_split_*`) |
| `tests/kelvin_ref.py` | independent reference (scipy ODE integration instead of the quadrature of the Fortran test); `--selftest` checks the closed forms against a direct integration of the three velocity components (4e-12); `--compare run.csv` checks a table of the test; without options it writes the reference curves |
| `tests/kelvin_s2_study.py` | the full matrix on demand, with the constant-S2 controls |
| `tests/decks/kelvin_s2_*.in` | the 13 decks of the suite |
| `tests/run_tests.sh` | runs the 13 decks, four of them with `kelvin_ref.py --compare` (skipped without numpy and scipy) |

`make test` builds it, `tests/run_tests.sh build-cpu` runs it (+23 s).

## What each run checks

The column holds v0 = 1 + 0.3 i, eta0 = 0.4 - 0.2 i (the nonzero eta0 tests the
G = 0 decoupling), k_z = 2 pi/5, k_y0 = pi, nu = 1e-3 (or 1e-12), one column of
the 15 x 2 x 5 box (nx = nz = 1, ny = 128).  For k_x = 0 the Hermitian partner
(0, -k_z) is set.  At each output time, with the analytic k_y(t):

* amplitudes of v and eta projected on exp(i k_y y), against the closed form;
* the projection residual |v - vhat exp(i k_y y)|/|v| (and for eta): a wrong
  phase or wavenumber in the shift shows up here;
* the energy of the column from the three velocity components against
  (k^2 |v|^2 + |eta|^2)/(2 k_h^2) and, for G = 0, against |u0|^2 E^2;
* gamma_y of the code (the value written to the field files) against its
  closed form, to 1e-10.

Errors in v, eta, energy are relative to the largest reference amplitude of
the run; the pass limit is `tol` (1e-5), the same for the projection residual.
`refine = r` repeats the run at deltat/2, ..., deltat/2^r and prints the
observed order; `order_min` makes it a criterion (1.6).

## The decks of the suite

| deck | column | G | S T | duration | largest error |
|---|---|---|---|---|---|
| `a_G0` | k_x = 0 | 0 | 5.25 | 10 periods | 2.6e-6 (energy) |
| `a_G1` | k_x = 0 | 1 | 5.25 | 10 periods | 1.2e-7 |
| `a_Gh_fast` | k_x = 0 | 0.417 | 1.75 | 10 periods | 1.5e-7 |
| `a_G0_slow` | k_x = 0 | 0 | 14 | 5 periods | 2.1e-6 |
| `a_G0_inviscid` | k_x = 0 | 0 | 5.25 | 4 periods, nu = 1e-12 | 7.1e-7 |
| `b_G0` | k_x = 2 pi/15 | 0 | 5.25 | 4 periods | 2.9e-6 |
| `b_G1` | k_x = 2 pi/15 | 1 | 5.25 | 4 periods | 3.6e-7 |
| `b_Gh` | k_x = 2 pi/15 | 0.222 | 5.25 | 4 periods | 9.8e-7 |
| `b_G0_fast` | k_x = 2 pi/15 | 0 | 1.75 | 8 periods | 4.0e-7 |
| `b_Gh_start` | k_x = 2 pi/15 | 0.222 | 5.25 | S2 switched on at t = 1.3 | 4.8e-7 |
| `order_a_G1`, `order_b_Gh` | both | 1, 0.222 | 5.25 | 3.15, deltat 0.105 to 0.013 | order 1.9-2.0, 2.4-2.7 |
| `order_a_G1_const` | k_x = 0 | 1 | constant S2 = 1 | the same: the control | order 2.0-2.1 |

Mode (b) is limited by resolution: max |k_y| dy stays below 0.12 (ny = 128),
as in the existing decks; 10 periods would need ny = 512.

## What the test found out about itself

* **Sampling at the end of the period is blind for k_x = 0.**  The mode is
  back at k_y0 after every period, and v = v0 E there for every G: G = 0 and
  G = 1 have the same reference at t = mT.  A first version that compared at
  period ends only passed the code with the whole extra d2v term removed.
  The test now samples eight times per period; the gain peaks at T/2.
* **The time error is third order only without viscosity.**  Observed order
  3.00 with nu = 1e-12 (G = 0 and 0.547, both k_x), and 2.0 (G = 1) to 2.5
  (G < 1, G = 0) with nu = 1e-3, equal for constant and oscillating S2.  The
  explicit filtered terms with S2(t) are therefore integrated at the order of
  the Runge-Kutta scheme; what lowers it is the viscous term, which the scheme
  treats with Crank-Nicolson (not investigated further).  The older text in
  `test_kelvin.f90` says O(deltat^3).  A first-order time error, as from S2
  taken at the start of a step, shows as order 1.0.
* **Bugs in the solver that the decks catch** (copies of the code with one
  change each; "X" = the deck fails):

| change | caught by |
|---|---|
| explicit S2 of the step start, not of the substep (specification pitfall 1) | every deck with k_x != 0 or G < 1 (S2 enters no explicit term for k_x = 0, G = 1); observed order 1.0-1.1 |
| shear-periodic shift with -gamma_z (pitfall 3) | all but `a_G0_inviscid` |
| displacement of a substep S2(t_i) dt instead of the integral | all but the constant control; observed order 1.0 |
| S2 frequency of gamma_z 1 % off | all but the constant control |
| extra d2v term without its S2 part | all with G < 1 |
| extra d2v term missing | all with G < 1 |
| extra d2v term with S2 of half a step earlier (pitfall 2) | all with G < 1 |
| eta lift-up: sign of the S2 part reversed | decks with k_x != 0 and G > 0 (`b_G1`, `b_Gh`, `b_Gh_start`, `order_b_Gh`); G multiplies it, and for k_x = 0 it vanishes |

## The full matrix

`tests/kelvin_s2_study.py [--build build-cpu] [--jobs 4] [--quick]` runs
modes (a) k_x = 0 and (b) k_x = 2 pi/15, G = 0, 0.547, 1, S T = 1.75, 5.25, 14,
nu = 1e-3 and 1e-12, at ny = 256 with deltat = min(0.1, T/50), deltat/2,
deltat/4, and the constant-S2 controls (run to |k_y| dy = 0.1, about 10
time units).  4.5 minutes on 8 processes of the CPU build.

Result (2026-11-16, `tests/kelvin_s2_study_result.md`, 36 oscillating cases
and 6 controls, all pass): the error at the finest deltat is 3e-6 or less in
v, eta and the energy, the projection residual 1e-8 or less, gamma_y within
1e-12; the finest table agrees with `kelvin_ref.py`.  Observed order in v
and eta: 3.0 for nu = 0 (G = 0, 0.547), 2.0 (G = 1) to 2.8 with nu = 1e-3,
against 2.0 to 2.6 in the constant-S2 controls of the same G.  Where the
error is already below 5e-7 at the second deltat the order is not shown
(`nan`): G = 1 with nu = 0 sits at the spatial floor of 1e-9 already at the
coarsest step.

## Limits

* One column and linear: the nonlinear transfer, the (0, 0) column and the
  statistics are not touched.  (The specification also allows a nonlinear
  run with amplitude about 1e-8; not done.)
* The suite runs on one rank (the test stops otherwise).  CPU (gfortran)
  only so far; the GPU and NCCL builds have not run it.
* 5e-9, the error of the existing 1-time-unit decks, is not reached over 10
  periods at deltat = 0.025: with nu = 1e-3 the time error is second order.
  The decks guard against bugs (all of the above are at 1e-3 or more), not
  against discretisation changes below 1e-5.

## Order test and seeded-bug audit (2026-11-17)

`tests/kelvin_checks.py` (`order`, `margin`).  Seeded-bug audit: the 9 bugs
above plus B9 (S2 at the step midpoint, an order-2 error), each against all
26 Kelvin decks at the suite's own time step; the correct solver fails none.
Every bug fails at least 5 decks.  Margin = largest error/tolerance:

| bug | margin | detecting deck |
|---|---|---|
| gamma_period, gamma_sign, extra_v_missing, extra_v_no_s2, shift_s2_lag, inconsistent_s2, B1 (S2 at step start), eta_s2_sign | 1e4 and up | various |
| **B9 (S2 at step midpoint)** | **5.0, fragile** | order_b_Gh; 4.8e-5 against 1e-5 in `a_G0`, 5.9e-6 (missed) in `a_G0_slow`, 1.2e-7 (invisible) in `a_G1` |

Headroom of the correct solver (tolerance/error) is 3.4 to 6 in the G = 0
decks (energy) and larger elsewhere; the old `kelvin.in` has 2.8.  Per-deck
tolerances of 3 x error would give B9 a margin of 17-22 in `a_G0_inviscid`,
`a_Gh_fast`, `b_G0_fast`, `b_Gh_start` and leave 5-6 or less elsewhere.  Results
and conclusions in `tests/KELVIN_S2_AUDIT.md`.

The order test with the case of the addendum (k_x = 0, G = 1, S T = 1.75,
nu = 1e-12, v only) does not work for this code: the error of v does not depend
on deltat (2.1e-8 at deltat = T/32, T/64, T/128: the spatial floor; 1.4e-8 at
ny = 128 and T/8 to T/128 alike), because for k_x = 0 and G = 1 S2 enters v only
through the exact shear shift, with no explicit term.  The correct solver fails
(order 0.00) and so do B1 and B9 with the same numbers.  With G = 0 (the extra d2v
term makes v depend on S2 explicitly) it works: correct 3.23 (PASS), B1 0.99,
B9 2.00 (FAIL).  eta and the energy of the G = 1 case converge at order 3.0.
