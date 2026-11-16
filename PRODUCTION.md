# Production filter (branch `production_filter`, from main)

Session 17 (2026-11-13).  Fourth route to a turbulence whose statistics do
not depend on the angle between the mean shear and the box.  The
large-scale damping (DAMPING.md) achieves it at n = 10, but (i) already at
theta = 0 the flow differs from the small box (w'w' close to u'u'; a
circular hard mask gives the statistics of the 1:1 box), and (ii) the
ellipse must turn with an unsteady shear, which acts on the flow.

## 1. Idea

No mode is removed.  The scale is prescribed through the energy
injection: the mean shear produces only in the columns with

    k_h = sqrt(kx^2 + kz^2) >= prod_kc

(isotropic in the x-z plane, so nothing turns with the shear).  The mean
shear enters the fluctuation equations twice:

* the advection S y d/dx (+ S2 y d/dz), which tilts the wavevectors,
  ky(t) = ky0 - S kx t - ...; with its share of the pressure it conserves
  energy.  It must act on every mode (otherwise the lattices of the large
  and small scales differ and triads do not close) and is untouched;
* the production term -v (S, 0, S2) of the momentum equation, the only
  energy source.  It is multiplied, column by column, by the filter
  G(k_h): 0 below the cutoff, 1 above, optionally a smoothstep over
  prod_kc (1 -+ prod_edge).

The tilting changes ky only: k_h of a column is invariant, so the columns
below the cutoff are fed by the nonlinear transfer alone (backscatter),
are tilted, and dissipate.  The energy budget is
d(q2/2)/dt = -S <G u v> - S2 <G w v> - eps.

## 2. Implementation

Deck group `&damping`: `prod_kc` (0 = off), `prod_edge` (0 = step; the
columns on k_h = prod_kc within 1e-6 are forced).  Independent of the
damping parameters.

* `hst_damping`: `gprod(iz, ix)` = G, built once (1 everywhere when off).
* `hst_equations` (`buildrhs_prepare`): the lift-up term of eta is
  multiplied by G; d2v gets the explicit term
  -(1 - G) (i alfa S + i beta S2) D1 v.  (Unfiltered, the production and
  the advection cancel in the d2v equation, which is why it has no shear
  term; removing the production leaves the advection's half.)  With
  G = 1 the added terms are exact zeros: the regression (three decks at
  1e-10 against the references) is unchanged.
* `hst_stats`: `production_runtime.dat`: time, number of columns with
  G < 1, ly/2 <(1 - G) uu>, <.. vv>, <.. ww>, ly/2 <(1 - G) grad u : grad u>
  (the unforced range: its energy by component and its dissipation, which
  in the mean equals the nonlinear flux into it), ly/2 <G u v>,
  ly/2 <G w v> (the stresses that produce), ly/2 <(1 - G) u w> (to turn
  the unforced stresses into the frame of the shear).

## 3. Verification

A Kelvin mode in a column with filter value G has the closed form

    v(t) = v0 (k0^2/k(t)^2)^((1 + G)/2) exp(-nu int k^2 dt)

(G = 1: Kelvin's k^-2; G = 0: |v| k constant, and for kz = 0 the energy
is conserved exactly), and for G = 0 eta keeps its initial value.
`tests/test_kelvin` with four new decks: `kelvin_unforced.in` (G = 0:
v 5e-9, eta 3e-10 relative), `kelvin_unforced_s2.in` (with a constant
S2: 4e-9, 2e-9), `kelvin_halfforced.in` (G = 0.547 in a smooth edge:
5e-9), `kelvin_forced.in` (above the cutoff: unchanged).  The suite (19
decks) and the regression pass on the CPU and GPU builds.

## 4. Runs

`~/Codes/hst/damping_campaign` (`tools/make_runs.py --prod KC`, binary
`bin/hst-a100-pf` on HoreKA).  First experiment: n = 3 (nx 144, nz 72,
ny 95, Re 1000, |S| = 1), theta = 0, 15, 30, 45, sharp cutoff at

    prod_kc = 2 pi/sqrt(Lx Lz) = 2 pi/sqrt(3) = 3.6276

(the circle with the area of the small box's ellipse in the wavenumber
plane; 28 unforced columns), to S t = 1400.  Job 5238014.  The cutoff is
to be corrected once so that L_eps = q^3/eps matches the small box (0.40)
if it does not.

### First experiment: n = 3, k_c = 2 pi/sqrt(3), sharp edge (job 5238014, 2026-11-14)

Four angles to S t = 1400 (1 h 49 min on four A100, 0.030 s/step), averages
over t > 400 (`tools/analyse.py`, `tools/prod.py` of the campaign).  Frame
of the shear: par along it, perp across it in the plane.

| case | theta | q2 | S* | -<u_par v>/q2 | L_eps | Re_lambda | b_par | b_v | b_perp |
|---|---|---|---|---|---|---|---|---|---|
| small box | 0 | 0.1085 | 6.23 | 0.163 | 0.395 | 33.6 | 0.112 | -0.049 | -0.064 |
| n3 hard ellipse | 0 | 0.1192 | 6.31 | 0.159 | 0.419 | 35.4 | 0.066 | -0.102 | 0.037 |
| n3 filter | 0 | 0.1120 | 6.82 | 0.148 | 0.439 | 35.7 | 0.148 | -0.117 | -0.031 |
| n3 filter | 15 | 0.1050 | 6.80 | 0.149 | 0.424 | 34.5 | 0.141 | -0.117 | -0.024 |
| n3 filter | 30 | 0.1015 | 6.80 | 0.149 | 0.417 | 33.9 | 0.132 | -0.116 | -0.016 |
| n3 filter | 45 | 0.0966 | 6.78 | 0.150 | 0.406 | 33.0 | 0.126 | -0.116 | -0.010 |

Block errors: q2 0.001-0.0015, S* 0.01-0.02, b 0.001.

| theta | E_low/q2 | eps_low/eps | P/eps | uvG/uv | b_low (par, v, perp) | b_high (par, v, perp) |
|---|---|---|---|---|---|---|
| 0 | 0.165 | 0.058 | 1.003 | 0.995 | 0.174 -0.224 0.050 | 0.143 -0.096 -0.047 |
| 15 | 0.163 | 0.058 | 1.003 | 0.989 | 0.166 -0.221 0.055 | 0.137 -0.097 -0.039 |
| 30 | 0.163 | 0.058 | 1.003 | 0.988 | 0.159 -0.220 0.061 | 0.127 -0.096 -0.030 |
| 45 | 0.161 | 0.058 | 1.003 | 0.983 | 0.160 -0.220 0.060 | 0.119 -0.097 -0.023 |

- The flow is stationary and the budget closes (P/eps = 1.003).  The
  unforced range holds 16 % of the energy at every angle, constant in time
  from S t = 100 on (0.154-0.166 per window, rms 0.015), and receives 5.8 %
  of the dissipation as nonlinear flux; it carries 1-2 % of the shear
  stress.  No accumulation at the box scale.
- L_eps (0.41-0.44) and Re_lambda (33-36) are within 3-11 % of the small
  box (0.395, 33.6): the equal-area cutoff reproduces the scale.
- S* (0.6 % spread), -<u_par v>/q2 (1.8 %) and b_v (0.001) are angle
  independent.  q2 falls monotonically by 15 % from theta 0 to 45, and
  energy moves from par to perp (b_par 0.148 -> 0.126, b_perp -0.031 ->
  -0.010; in the box frame at 45 degrees uu = ww).  The trend is many
  block errors; the hard ellipse at n = 3 had the same size of spread
  (q2 24 %, b 0.02) but irregular in theta.
- The structure differs from both references: S* 6.8 (6.2-6.3), stress
  ratio 0.148 (0.16), b_par 0.13-0.15.  The ordering is par > perp > v:
  the par/perp pancake of the hard mask (0.066/0.037) is gone, b_perp is
  between the small box and the hard mask, but v is as weak as with the
  hard mask (-0.117; small box -0.049).  The unforced range itself is
  strongly two-component (b_v = -0.22).
- Spectra (last three, `spectrum_n3_prod3.63_*.png`): the most energetic
  columns are the forced ones just outside the circle with k_par near 0:
  (k_par, k_perp) = (0.7, 4.19), (1.4, 4.19), (0, 4.19) at theta = 0,
  3.9-5.0 at the other angles.  The streaks take the widest spacing that
  is forced, 2 pi/4.19 = 1.5, where the small box allows 1.  Which k_perp
  the lattice offers just above k_c changes with theta (kz step 2.09 at
  n = 3): the likely origin of the residual angle trend.

Consequences: (i) the lattice near the cutoff matters as with the hard
mask, so n = 5 at the same k_c is the next run (job 5238353, 6 h limit);
(ii) the equal-area circle forces spanwise wavelengths up to sqrt(3) and
the flow uses them; a cutoff k_c = 2 pi/Lz = 6.28 would restrict the
streak spacing to the small box's and unforce only streamwise-short,
spanwise-wide modes in addition (candidate, not run).


### n = 5 at the same cutoff (job 5238353, 2026-11-15; 4 h 51 min)

| theta | q2 | S* | -<u_par v>/q2 | b_par | b_v | b_perp | E_low/q2 | eps_low/eps |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1237 | 6.93 | 0.147 | 0.146 | -0.116 | -0.030 | 0.169 | 0.056 |
| 15 | 0.1140 | 6.88 | 0.148 | 0.139 | -0.116 | -0.023 | 0.166 | 0.056 |
| 30 | 0.1094 | 6.87 | 0.148 | 0.131 | -0.115 | -0.016 | 0.164 | 0.056 |
| 45 | 0.1055 | 6.87 | 0.149 | 0.129 | -0.116 | -0.013 | 0.163 | 0.056 |

- The angle trend does not shrink with n: q2 falls by 16 % from 0 to 45
  degrees (15 % at n = 3), b_par and b_perp move by the same 0.02.  It is
  therefore not the lattice sampling near the cutoff (the hypothesis of
  the n = 3 section is wrong; the streak columns now sit at k_perp
  3.8-4.3 at all angles).  The level rises by 9-10 % from n = 3 to 5 at
  every angle; the structure (S*, stress ratio, b_ij) is unchanged.
- Energy per ring of k_h (share of q2; `exz` files 3-7): 0-1.2: 0.014,
  1.2-2.4: 0.06, 2.4-k_c: 0.09, k_c-5: 0.21, 5-7: 0.25; the same at all
  angles within 0.005.  The unforced range is filled evenly per unit
  area (no pile-up at the box scale), the (0,0) column holds 0.1 %.
- Open: what sets the monotone theta trend.  What still depends on theta
  is the box (3:1 in x:z) seen by the unforced modes, which the hard
  ellipse removed.  Tests: theta = 60, 90 at n = 3 (does the trend
  continue?), a square box of the same area.

### k_c = 2 pi/Lz = 6.2832, n = 3, Re = 1000 (job 5238576): laminar

All four angles decay monotonically from the initial field (ly/2 q2:
0.114 -> 0.011 at S t = 50 -> 1e-3 at 120; the k_c = 3.63 run recovers
from 0.057 at t = 5 to 0.13 at t = 50) and are laminar by S t = 130.
With production only at wavelengths <= 1 in the plane, Re = 1000 does
not sustain turbulence: S l_f^2/nu = 1000 here against 3000 for the
equal-area cutoff.  (Scaling: k_c = 6.28 at Re = 3000, Ly = 2 is the k_c
= 3.63, Re = 1000 flow in a box with Ly = 2 sqrt(3) in units of l_f; so
the Re-corrected run also tests Ly/l_f = 2 against 1.15.)

Re-corrected runs (Re = 2000, 3000 on finer grids; jobs 5239964, 5239965)
were cancelled before they started (2026-11-15); their rescaled
equivalent is the taller box below.

### The cutoff edge is the source of the angle dependence (2026-11-16)

Two tests at n = 5, four angles, S t = 1400 (jobs 5240018, 5240020):

| case | q2 at 0 / 15 / 30 / 45 deg | spread | S* | -<u_par v>/q2 | b_par | b_v | b_perp | L_eps |
|---|---|---|---|---|---|---|---|---|
| k_c 3.63, sharp | 0.1237 / 0.1140 / 0.1094 / 0.1055 | 16 %, monotone | 6.87-6.93 | 0.147-0.149 | 0.146 -> 0.129 | -0.116 | -0.030 -> -0.013 | 0.469 -> 0.429 |
| k_c 3.80, sharp | 0.0962 / 0.1027 / 0.0988 / 0.0972 | 6.6 %, theta = 0 lowest | 6.85-6.91 | 0.148-0.149 | 0.136-0.143 | -0.119 | -0.017..-0.023 | 0.409-0.426 |
| k_c 3.63, edge 0.3 | 0.0886 / 0.0883 / 0.0905 / 0.0875 | 3.4 % | 6.92-6.94 | 0.153 | 0.141-0.143 | -0.122 | -0.018..-0.021 | 0.394-0.402 |

(block errors of q2: 0.7-1 %.)  Moving the sharp cutoff from 3.63 to 3.80
takes the theta = 0 row k_z = 3.77 out of the forced set and turns
theta = 0 from the highest into the lowest level: the trend of the sharp
filter is which lattice rows of long modes fall just inside the circle.
The smoothstep edge (G from 0 at 2.54 to 1 at 4.72) removes it: q2 within
3.4 % (30 deg 2-3 sigma above the others), b_ij within 0.002, every
normalised quantity flat.  Its level is lower (q2 0.088, uv_G/uv 0.94)
and L_eps = 0.395 equals the small box (0.395); S* 6.93 and the stress
ratio 0.153 against 6.23 and 0.163 there.

Where the sharp filter's dependence sits (`tools/streak_bands.py`, bands
of the shear frame with the columns spread over their lattice cells): the
small box's streak family (|k_perp| >= 2 pi, |k_par| < 2 pi/3) is angle
independent in energy and production; the long modes wider than it lose
16 % of their energy and carry 14-15 of the 13-18 points of production
lost at 45 deg.  The k_x spectrum at theta = 0 still rises at the lowest
non-zero k_x of n = 5: the streak length is not box independent.

### Square, taller box (job 5240934, pending)

`tools/make_square.py`: 10.5 x 2 sqrt 3 x 10.5 (nx 168, ny 165, nz 252),
k_c = 2 pi/sqrt 3 with edge 0.3, theta 0/15/30/45.  Square because the
3:1 shape has no role once k_c sets the scale and theta, 90 - theta
become equivalent; Ly/l_f = 2 (Sekimoto et al.: Ly/Lz < 1 is vertically
constrained) instead of 1.15, the rescaled k_c = 2 pi, Re = 3000 flow.
