"""Clockwork neutrinos: regions of the (m, y) plane where the geometric-phase shift
lies between 0.1 and 10 rad (P7, Fig. 1).

Physics (leading order in p = y v / m):
    delta Theta_cw = 2.534 * y^2 v^2 * L / E        (y^2 v^2 in eV^2, L in km, E in GeV)
because sum_j C_j = N + 1 (up to q^{-2N}) and the flavour weights sum to 1.
A source with energies E_min..E_max and baseline L covers
    y from  y(0.1 rad, E_min)  to  y(10 rad, E_max),
for gear scales m below the coherence limit of the heaviest gear,
    m * sqrt(lambda_N) <= sqrt(dm2_max(E)),   dm2_max = 2 E sqrt(E / L)   (natural units).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon

# ----------------------------------------------------------------------------- physics
V_EV = 246e9          # Higgs vev [eV]
HBARC = 0.197e-6      # eV m
N_GEARS, Q = 15, 3
k = np.arange(1, N_GEARS + 1)
lam = 1 + Q**2 - 2 * Q * np.cos(k * np.pi / (N_GEARS + 1))
C = 2 * Q**2 / lam * np.sin(N_GEARS * k * np.pi / (N_GEARS + 1))**2
SUM_C = C.sum() / (N_GEARS + 1)          # = 1 to 1e-14 for N = 15, q = 3
HEAVIEST = np.sqrt(lam.max())            # heaviest gear mass / m  (~3.99)


def y_of(dphi, E_GeV, L_km):
    """Yukawa coupling giving a phase shift dphi [rad]."""
    return np.sqrt(dphi * E_GeV / (2.534 * L_km * V_EV**2 * SUM_C))


def m_coh(E_GeV, L_km):
    """Largest gear scale m [eV] for which the heaviest gear stays coherent."""
    E, L = E_GeV * 1e9, L_km * 1e3 / HBARC          # eV, eV^-1
    return np.sqrt(2 * E * np.sqrt(E / L)) / HEAVIEST


def band(Emin, Emax, L, mleft=0.1):
    """Vertices of the region covered by one source (log-log polygon)."""
    ylo, yhi = y_of(0.1, Emin, L), y_of(10, Emax, L)
    return [(mleft, ylo), (m_coh(Emin, L), ylo), (m_coh(Emax, L), y_of(0.1, Emax, L)),
            (m_coh(Emax, L), yhi), (mleft, yhi)]


# name, E_min [GeV], E_max [GeV], L [km], colour  (four overlapping bands: validated all-pairs)
SOURCES = [
    ("FASER",  100,   1000,  0.48,  "#eda100"),
    ("SBND",   0.1,   10,    0.11,  "#008300"),
    ("DANSS",  1e-3,  1e-2,  0.01,  "#e87ba4"),
    ("DUNE",   0.5,   10,    1300,  "#2a78d6"),
    ("Solar",  1e-4,  1.8e-2, 1e8,  "#eb6834"),
    ("Astrophysical", 1e3, 1e6, 1e22, "#6250d6"),
]
DETAIL = {"FASER": r"100 GeV$-$1 TeV, 480 m", "SBND": r"0.1$-$10 GeV, 110 m", "DANSS": r"1$-$10 MeV, 10 m",
          "DUNE": r"0.5$-$10 GeV, 1300 km", "Solar": r"0.1$-$18 MeV, $10^{8}$ km",
          "Astrophysical": r"1 TeV$-$1 PeV, $10^{22}$ km"}

# ----------------------------------------------------------------------------- style
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.family": ["cmr10", "DejaVu Serif"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
    "font.size": 10, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.edgecolor": INK2, "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
})

XMIN, XMAX, YMIN, YMAX = 0.1, 1e7, 1e-23, 1e-5
fig, ax = plt.subplots(figsize=(6.4, 5.0))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
ax.set_yticks([10.0**e for e in range(-23, -4, 3)])

halo = [pe.withStroke(linewidth=2.5, foreground="white")]
for i, (name, Emin, Emax, L, col) in enumerate(SOURCES):
    verts = band(Emin, Emax, L)
    ax.add_patch(Polygon(verts, closed=True, facecolor=col, alpha=0.22, edgecolor="none", zorder=2 + i))
    ax.add_patch(Polygon(verts, closed=True, facecolor="none", edgecolor=col, linewidth=1.4, zorder=10 + i,
                         label=f"{name} ({DETAIL[name]})"))
    # short direct label beside the band's right edge
    xr = m_coh(Emax, L)
    ymid = np.sqrt(y_of(0.1, Emax, L) * y_of(10, Emax, L))
    if name == "FASER":   # no room right of the band: label above its top edge instead
        ax.text(xr / 1.3, y_of(10, Emax, L) * 1.8, name, color=INK, fontsize=10,
                va="bottom", ha="right", path_effects=halo, zorder=30)
    else:
        ax.text(xr * 1.25, ymid, name if name != "Astrophysical" else "Astro", color=INK, fontsize=10,
                va="center", ha="left", path_effects=halo, zorder=30)

# non-perturbative region  y > m / v
m = np.logspace(np.log10(XMIN), np.log10(XMAX), 200)
ax.fill_between(m, m / V_EV, YMAX, facecolor="#f0efec", edgecolor="#c3c2b7", hatch="////",
                linewidth=0, zorder=20)
ax.plot(m, m / V_EV, color="#8a8984", linewidth=1.0, zorder=21)
ax.set_xlabel(r"clockwork mass scale $m$ [eV]", color=INK)
ax.set_ylabel(r"Yukawa coupling $y$", color=INK)
from matplotlib.patches import Patch
handles = [Patch(facecolor=(*plt.matplotlib.colors.to_rgb(col), 0.22), edgecolor=col, linewidth=1.4)
           for _, _, _, _, col in SOURCES]
labels = [f"{name} ({DETAIL[name]})" for name, *_ in SOURCES]
leg = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False,
                 handlelength=1.6, columnspacing=1.6, labelcolor=INK, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.12, 1, 1))
# rotate the label along the boundary
p1, p2 = ax.transData.transform([(1e1, 1e1 / V_EV), (1e3, 1e3 / V_EV)])
angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
ax.text(3e1, 3e1 / V_EV * 30, r"non-perturbative ($y > m/v$)", rotation=angle, rotation_mode="anchor",
        color=INK2, fontsize=10, ha="left", va="bottom", path_effects=[pe.withStroke(linewidth=3, foreground="#f0efec")],
        zorder=22)

fig.savefig(str(Path(__file__).resolve().parents[1] / "figures") + "/clockwork_eV-MeV.pdf")
fig.savefig(str(Path(__file__).resolve().parents[1] / "figures") + "/clockwork_eV-MeV.png", dpi=300)
print("sum C/(N+1) =", SUM_C, " heaviest gear / m =", HEAVIEST)
