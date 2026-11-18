# Order test and seeded-bug audit: results and conclusions

*Branch `production_filter`, 17 November 2026.  Runs of the two checks proposed in
`kelvin_order_and_bug_audit.md` (Option 1: order-of-accuracy test; Option 2:
seeded-bug audit), against the Kelvin-mode tests of `tests/KELVIN_S2.md`.
CPU build (gfortran, FFTW), one rank.  The GPU and NCCL builds were not run.*

## 1. Summary

1. **The seeded-bug audit shows the suite detects all nine bugs.**  Every bug fails
   at least 5 of the 26 Kelvin decks and the correct solver fails none.  Eight bugs
   exceed the tolerance by a factor of 10^4 or more.  One, B9 (S2 taken at the
   step midpoint, a second-order error), is only fragile: best margin 5.0, and
   three decks miss or barely see it.
2. **The order test, as specified, cannot be used.**  In the proposed case
   (k_x = 0, G = 1, v only) the correct solver gives observed order 0.00 and
   the seeded bugs B1 and B9 give the same numbers as the correct solver.  The
   error in v does not depend on the time step in this column.
3. **With G = 0 the order test works**: the correct solver passes (order 3.23),
   B1 and B9 fail (0.99 and 2.00).  It is the only check that detects B9 with a
   large margin and without a hand-tuned tolerance.
4. **Recommendation**: add the G = 0 order test to the suite, and tighten the
   tolerances per deck (section 5).

## 2. Setup

| | |
| --- | --- |
| Solver | the current branch, and 9 copies with one seeded bug each (below) |
| Decks | 13 `kelvin_s2_*.in` (this branch) and the 13 older `kelvin*.in` of `test_kelvin` |
| Time step | the decks' own (suite) step for the audit; Δt₀ = T/32 and its halvings for the order test |
| Order-test column | k_x = 0, k_z = 2π/5, k_y0 = π, A/S = 1, S T = 1.75, ν = 10⁻¹², two periods, 8 outputs per period, ny = 128 |
| Order-test reference | `kelvin_ref.closed_form`, errors in v relative to |v₀|, evaluated at the run's own output times |

Seeded bugs (each one change in a copy of `src/`):

| name | change |
| --- | --- |
| B1 `B1_S2_step_start` | S2 of the explicit terms taken at the start of the step instead of each substep (order 1) |
| B9 `B9_S2_step_midpoint` | S2 of the explicit terms taken at t + Δt/2 of the step in every substep (order 2) |
| `shift_s2_lag` | displacement of a substep S2(t_i) Δt instead of the integral of S2 |
| `gamma_sign` | shear-periodic shift with -γ_z |
| `gamma_period` | frequency in the closed form of γ_z 1 % off |
| `extra_v_missing` | extra ∇²v term, -(1 - G)(iαS + iβS2) D1 v, removed |
| `extra_v_no_s2` | extra ∇²v term without its S2 part |
| `inconsistent_s2` | extra ∇²v term sees S2 half a step earlier than the η term |
| `eta_s2_sign` | sign of the S2 part of the η lift-up reversed |

## 3. Option 1: the order test

Three runs at Δt₀ = T/32 = 0.0547, Δt₀/2, Δt₀/4 (exact halving), `kelvin_checks.py order`.

### 3.1 The case as specified (G = 1): invalid

| solver | max |v - v_ref|/|v₀| at Δt₀, Δt₀/2, Δt₀/4 | observed order, finest pair | verdict |
| --- | --- | --- | --- |
| correct | 2.1e-8, 2.1e-8, 2.1e-8 | 0.00 | FAIL |
| B1 | 2.1e-8, 2.1e-8, 2.1e-8 | 0.00 | FAIL |
| B9 | 2.1e-8, 2.1e-8, 2.1e-8 | 0.00 | FAIL |

The error is the same to all printed digits for all three solvers and does not
depend on Δt.  A scan from T/8 to T/128 gives 1.39e-8 at ny = 128 and 8.7e-10 at
ny = 256 over the whole range: this is the spatial error of the y stencils.

The reason is in the equations.  For k_x = 0 the extra ∇²v term is
-(1 - G)(iβS2) D1 v and vanishes for G = 1; S2 then acts on v only through the exact
shear-periodic shift of the right-hand sides, which has no time error.  The time
error of the G = 1 column sits in η and the energy, which converge at order 3.0
(η: 7.5e-5, 9.2e-6, 1.15e-6, 1.5e-7 for T/8 ... T/64), but the proposed check
reads v only.  The script's guard against a meaningless test (finest error below
1e-11) does not catch this, because the spatial floor is far above 1e-11; it
reports FAIL, not INVALID.

### 3.2 The G = 0 variant: works

Same column with the filter off (sharp filter above the column's k_h: G = 0).  The extra term is
then -iβS2 D1 v and v depends on S2 explicitly.

| solver | error at Δt₀, Δt₀/2, Δt₀/4 | order (coarse pair, fine pair) | verdict |
| --- | --- | --- | --- |
| correct | 5.05e-6, 6.18e-7, 6.57e-8 | 3.03, 3.23 | PASS |
| B1 (S2 at step start) | 1.60e-2, 8.13e-3, 4.10e-3 | 0.98, 0.99 | FAIL (first order) |
| B9 (S2 at step midpoint) | 4.18e-4, 1.04e-4, 2.60e-5 | 2.00, 2.00 | FAIL (second order) |

The finest correct error is 6.6e-8.  The spatial floor of this column was not
measured; for G = 1 it is 1.4e-8 at ny = 128 (section 3.1), so the fine pair is
probably already slightly contaminated.  A smaller Δt₀ would put it into the floor; a larger one is not needed.

## 4. Option 2: seeded-bug audit

### 4.1 Which decks detect which bug (X = deck fails)

| bug | a_G0 | a_G0_inviscid | a_G0_slow | a_G1 | a_Gh_fast | b_G0 | b_G0_fast | b_G1 | b_Gh | b_Gh_start | order_a_G1 | order_a_G1_const | order_b_Gh | old decks (of 13) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | X | X | X | . | X | X | X | X | X | X | . | . | X | 0 |
| B9 | X | X | . | . | . | X | X | . | X | X | . | . | X | 0 |
| shift_s2_lag | X | X | X | X | X | X | X | X | X | X | X | . | X | 2 |
| gamma_sign | X | X | X | X | X | X | X | X | X | X | X | X | X | 4 |
| gamma_period | X | X | X | X | X | X | X | X | X | X | X | . | X | 0 |
| extra_v_missing | X | X | X | . | X | X | X | . | X | X | . | . | X | 3 |
| extra_v_no_s2 | X | X | X | . | X | X | X | . | X | X | . | . | X | 1 |
| inconsistent_s2 | X | X | X | . | X | X | X | . | X | X | . | . | X | 0 |
| eta_s2_sign | . | . | . | . | . | . | . | X | X | X | . | . | X | 1 |

The pattern follows the equations: the explicit S2 terms act only where
G < 1 (extra ∇²v term) or k_x ≠ 0 (η lift-up); the shift bugs act everywhere
except in the constant-S2 control.  The older `test_kelvin` decks run one time
unit and detect only the shift and extra-term bugs (0 to 4 of 13 each); none
of them sees B1, B9, `gamma_period` or `inconsistent_s2`.

### 4.2 Margins (largest error/tolerance over all decks and quantities)

| bug | best deck | quantity | error | tolerance | margin | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `gamma_period` | a_G0_slow | γ_y | 0.67 | 1e-10 | 7e9 | robust |
| `extra_v_missing` | a_G0_slow | energy | 5.4 | 1e-5 | 5e5 | robust |
| `extra_v_no_s2` | a_G0_slow | energy | 5.4 | 1e-5 | 5e5 | robust |
| `gamma_sign` | a_Gh_fast | energy | 1.9 | 1e-5 | 2e5 | robust |
| `shift_s2_lag` | a_G0 | projection | 0.20 | 1e-5 | 2e4 | robust |
| B1 | a_G0_slow | energy | 0.10 | 1e-5 | 1e4 | robust |
| `inconsistent_s2` | a_G0_slow | energy | 0.10 | 1e-5 | 1e4 | robust |
| `eta_s2_sign` | b_Gh_start | η | 0.10 | 1e-5 | 1e4 | robust |
| **B9** | order_b_Gh | energy | 2.5e-4 | 5e-5 | **5.0** | **caught, fragile** |

### 4.3 B9 in detail

| deck | correct solver | B9 | tolerance | B9 / tolerance |
| --- | --- | --- | --- | --- |
| a_G0 | 2.6e-6 | 4.8e-5 | 1e-5 | 4.8 |
| a_G0_inviscid | 7.1e-7 | 5.0e-5 | 1e-5 | 5.0 |
| a_G0_slow | 2.1e-6 | 5.9e-6 | 1e-5 | 0.6 (missed) |
| a_Gh_fast | 1.0e-7 | 9.7e-6 | 1e-5 | 0.97 (missed) |
| b_G0 | 2.9e-6 | 4.9e-5 | 1e-5 | 4.9 |
| b_G0_fast | 4.0e-7 | 3.4e-5 | 1e-5 | 3.4 |
| b_G1 | 1.8e-7 | 2.1e-6 | 1e-5 | 0.2 (missed) |
| b_Gh | 9.8e-7 | 1.5e-5 | 1e-5 | 1.5 |
| b_Gh_start | 4.8e-7 | 4.4e-5 | 1e-5 | 4.4 |
| order_b_Gh | 1.4e-5 | 2.5e-4 | 5e-5 | 5.0 |
| a_G1, order_a_G1, order_a_G1_const | 1e-7 to 4e-6 | same as correct | | 0 (blind) |

(The quantity is the largest of v, η, energy and projection residual in each deck.)

### 4.4 Headroom of the correct solver (tolerance/error)

The tightest values are 2.8 (old `kelvin.in`), 3.4 (`b_G0`, energy), 3.6
(`order_b_Gh`), 3.8 (`a_G0`), 4.7 (`a_G0_slow`); the others are 5 to
10^5.  Where the bugs' errors are 10^4 times the tolerance this headroom does not matter; for B9 it does.

With one tolerance per deck equal to 3 × its largest correct error, rounded up to
one digit, the B9 margin becomes (decks not listed are blind to B9 or unchanged):

| deck | tolerance now | tolerance at 3 × error | B9 margin now → then |
| --- | --- | --- | --- |
| a_G0_inviscid | 1e-5 | 3e-6 | 5.0 → 17 |
| a_Gh_fast | 1e-5 | 5e-7 | 1.0 → 19 |
| b_G0_fast | 1e-5 | 2e-6 | 3.4 → 17 |
| b_Gh_start | 1e-5 | 2e-6 | 4.4 → 22 |
| b_Gh | 1e-5 | 3e-6 | 1.4 → 4.8 |
| a_G0 | 1e-5 | 8e-6 | 4.8 → 6.0 |
| b_G0 | 1e-5 | 9e-6 | 4.9 → 5.4 |
| a_G0_slow | 1e-5 | 7e-6 | 0.6 → 0.8 (still missed) |
| b_G1 | 1e-5 | 2e-6 | 0.2 → 1.1 |
| order_b_Gh | 5e-5 | 5e-5 | 5.0 → 5.0 |
| a_G1, order_a_G1, order_a_G1_const | | | 0.0 to 0.3: blind to B9 |

Four decks reach a margin of 17 to 22; in the others the energy error of the
correct solver is itself dominated by the time error (2e-6 to 1e-5), so the
tolerance cannot be tightened without a smaller Δt, and B9 stays at 5 to 6 or
below.  Tighter tolerances help, but do not remove the need for the order test.

## 5. Conclusions

1. **Bugs of order 1 and gross errors in S2 handling are detected by many decks
   with huge margins.**  Nothing in the audit suggests that the S2 handling of
   the production filter is wrong: the correct solver converges at order 3.0
   (ν ≈ 0) and 2.0 to 2.6 (ν = 10⁻³), for oscillating S2 as for constant S2.
2. **The amplitude checks are weak against second-order and smaller errors.**
   B9 is the example: a second-order error has margin 5 at best, at the
   suite's Δt, and is missed by 3 of the 13 new decks.  With a smaller time
   step or a weaker bug it would pass.  A fixed tolerance is a poor test for
   this kind of bug; the order test is not.
3. **The order test must use a quantity that depends on S2 explicitly.**  For
   k_x = 0 and G = 1 the time error of v is zero and the check is meaningless;
   G = 0 (or k_x ≠ 0 with η or the energy) does depend on it.  The test
   would have caught nothing and failed on the correct solver, had it been added as specified.
   `kelvin_checks.py` should check this assumption before running: if the error
   of the coarsest and the finest run agree to 10 %, the case is limited by the
   spatial floor and the result should be INVALID, not FAIL.
4. **The older `test_kelvin` decks (one time unit) add little for the S2
   terms**: they detect none of B1, B9, `gamma_period`, `inconsistent_s2`.
5. **Two limits of this audit.**  The bugs are seeded one at a time and each is a
   one-line change; combinations were not run.  The audit is for this Δt, these
   decks and the CPU build (redo it when they change).

## 6. Recommended actions

1. Add the G = 0 order test (three decks at Δt₀, Δt₀/2, Δt₀/4, G = 0, ν = 10⁻¹², S T = 1.75,
   two periods, `kelvin_checks.py order`) to `tests/run_tests.sh`.  Cost: three runs of
   about 1 s.  Its observed orders on the current code: 3.03 and 3.23.
2. Make the check in `kelvin_checks.py order` return INVALID when the error
   does not decrease between the coarsest and the finest run (the floor case of section 3.1).
3. Decide on per-deck tolerances (section 4.4; four decks gain a B9 margin of 17 to 22) after running the suite on the GPU
   build: CPU-GPU differences are about 1e-13 for these runs, far below any of them.
4. Add B9 to the list of seeded bugs in `tests/KELVIN_S2.md` (done in section
   "Order test and seeded-bug audit") and keep the audit table with the code.

## 7. Files

| file | |
| --- | --- |
| `tests/kelvin_checks.py` | the two checks (`order`, `margin`); `read_run` reads the tables through `kelvin_ref.read_csv` |
| `tests/kelvin_ref.py` | `closed_form` added for the checker |
| `tests/test_kelvin_s2.f90` | the table has a `dt` column |
| `tests/kelvin_s2_bug_errors.csv` | the audit data: bug, deck, quantity, error, tolerance (1010 rows) |
| `tests/kelvin_s2_bug_audit.md` | the output of `kelvin_checks.py margin` (headroom of every deck and quantity) |
