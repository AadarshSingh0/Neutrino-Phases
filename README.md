# neutrino-phases

Code for two papers by Amol Dighe and Aadarsh Singh on geometric phases in neutrino oscillations:

- **P6:** *Enhanced geometric phases in neutrino oscillations from heavy sterile neutrinos: gedanken experiments and detection strategies*
- **P7:** *Geometric phases in neutrino oscillations beyond the Standard Model: time-varying neutrino masses, clockwork, extra dimensions and pseudo-Dirac neutrinos*

## Contents

| File | What it does | Figure |
|---|---|---|
| `GP/Phases.nb` | Dynamical (Θ), total (φ) and geometric (Φ = φ − Θ) phases for the SM, a heavy sterile state, clockwork, extra dimensions and pseudo-Dirac pairs | P6 Fig. 1 |
| `GP/Geometric_base.nb` | Dynamical and total phases for a general n-state system; SM and clockwork cases | – |
| `BP/BP_analytical.nb` | Analytic Berry connection for three active and one sterile neutrino in matter | P6 Sec. 4 |
| `BP/BP_Num.nb` | Berry phases on closed loops in the (ρ_e, ρ_n) plane, from the line integral and from the Bargmann product of eigenvectors | P6 Fig. 3 |
| `python-codes/sterile_bands/Ue4_future.py` | Sterile-neutrino regions with a 0.1–10 rad phase shift, \|U_e4\|², future limits | P6 Fig. 2 |
| `python-codes/sterile_bands/Ue4_current.py` | Same, \|U_e4\|², current limits | P6 Fig. 6 |
| `python-codes/sterile_bands/Umu4_*.py` | Same, \|U_μ4\|², current and future limits | P6 Fig. 7 |
| `python-codes/sterile_bands/Utau4_*.py` | Same, \|U_τ4\|², current and future limits | P6 Fig. 8 |
| `python-codes/clockwork_plot.py` | Clockwork (m, y) regions with a 0.1–10 rad phase shift | P7 Fig. 1 |
| `python-codes/pseudo_dirac.py` | (L, E) plane with lines of constant Δm² for pseudo-Dirac splittings | – |

`python-codes/data/` holds the digitised experimental limits read by each band script (one folder per script; see the papers for the original references). `figures/` holds the figures as they appear in the papers.

## Conventions

- A single-state phase is m²L/(2E) = 2.534 m²[eV²] L[km] / E[GeV]. The coefficient 1.27 appears only in oscillation (half-angle) arguments such as sin²(1.27 Δm²L/E).
- A band covers phase shifts between 0.1 and 10 rad over the energy range of each source, for states that remain coherent over the baseline, Δm² < 2E√(E/L).

## Running

Python 3 with `numpy`, `scipy`, `pandas`, `matplotlib` and `colour`:

```bash
pip install numpy scipy pandas matplotlib colour
python python-codes/sterile_bands/Ue4_current.py   # -> figures/Ue4_current_sterile.png
python python-codes/clockwork_plot.py              # -> figures/clockwork_eV-MeV.pdf/.png
```

Paths are relative to the repository, so the scripts run from any folder. The notebooks need Mathematica.
