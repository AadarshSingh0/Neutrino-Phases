import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ============================================================================
# CONSTANTS AND PARAMETERS
# ============================================================================
v = 246e9  # eV
etolog10 = 2.31
N1 = 15
q = 3
p = 1
n = N1

# Calculate lambda array
lambda_array = np.array([1 + q**2 - 2*q*np.cos(i*np.pi/(n + 1)) for i in range(1, n + 1)])

# Calculate Ccomp array
Ccomp = np.array([2*q**2/lambda_array[i-1] * np.sin(n*i*np.pi/(n + 1))**2 
                  for i in range(1, n + 1)])

# Calculate masscw array
masscw = np.array([np.sqrt(lambda_array[i-1] * (1 + p**2*Ccomp[i-1]/(2*(n + 1)*lambda_array[i-1]))) 
                   for i in range(1, n + 1)])

# Calculate ratio array and sumratio
ratio = np.array([(Ccomp[i-1] * masscw[i-1]**2) / lambda_array[i-1] 
                  for i in range(1, n + 1)])
sumratio = np.sum(ratio)

print(f"Calculated sumratio: {sumratio}")
print(lambda_array)
# Color definitions (matching your Mathematica colorList)
colorList = [
    (0.1, 0.2, 0.8),    # Blue [0]
    (0.8, 0.1, 0.2),    # Red [1]
    (0.2, 0.8, 0.2),    # Green [2]
    (0.9, 0.6, 0.2),    # Orange [3]
    (0.6, 0.2, 0.7),    # Purple [4]
    (0.2, 0.8, 0.8),    # Cyan [5]
    (0.9, 0.4, 0.6),    # Pink [6]
    (0.7, 0.7, 0.2),    # Olive [7]
    (0.9, 0.5, 0.0),    # Amber [8]
    (0.1, 0.9, 0.1),    # Lime [9]
    (0.5, 0.3, 0.0),    # Brown [10]
    (0.8, 0.0, 0.8),    # Magenta [11]
    (0.0, 0.5, 0.5),    # Teal [12]
    (1.0, 0.84, 0.0)    # Gold [13]
]

# ============================================================================
# FUNCTIONS
# ============================================================================
def delm2(E, L):
    """E in eV, L in m"""
    return np.sqrt(4 * 0.197e-6 * E**3 / L)

def pert(m):
    """m in eV"""
    return m / v

def ycwlow(E, L):
    """E in GeV, L in Km"""
    return np.sqrt((0.1 / (3 * 1.27)) * (E / L) * (1 / (sumratio * v**2)))

def ycwhigh(E, L):
    """E in GeV, L in Km"""
    return np.sqrt((10 / (3 * 1.27)) * (E / L) * (1 / (sumratio * v**2)))

# ============================================================================
# EXPERIMENTAL PARAMETERS
# ============================================================================

# DANSS: L = 10m, 1 MeV < E < 10 MeV
mindanss = np.sqrt(delm2(1e6, 10))      # E = 1 MeV = 10^-3 GeV
maxdanss = np.sqrt(delm2(1e7, 10))      # E = 10 MeV = 10^-2 GeV

# SBL (Short Baseline): L = 110m, 100 MeV < E < 10 GeV
minsbl = np.sqrt(delm2(1e8, 110))       # E = 0.1 GeV
maxsbl = np.sqrt(delm2(1e10, 110))      # E = 10 GeV

# LBL (Long Baseline): L = 1300km, 1 MeV < E < 10 GeV
minlbl = np.sqrt(delm2(1e6, 1.3e6))     # E = 10^-3 GeV
maxlbl = np.sqrt(delm2(1e10, 1.3e6))    # E = 10 GeV

# Solar: L = 10^8 km, 100 keV < E < 18 MeV
minsolar = np.sqrt(delm2(1e5, 1e11))    # E = 10^-4 GeV (100 keV)
maxsolar = np.sqrt(delm2(1.8e7, 1e11))  # E = 1.8*10^-2 GeV (18 MeV)

# Astro: L = 10^22 m, 1 TeV < E < 1 PeV
minastro = np.sqrt(delm2(1e12, 1e22))   # E = 10^3 GeV
maxastro = np.sqrt(delm2(1e15, 1e22))   # E = 10^6 GeV

# SND/FASER: L = 480m, 100 GeV < E < 1 TeV
minsnd = np.sqrt(delm2(1e11, 480))      # E = 100 GeV
maxsnd = np.sqrt(delm2(1e12, 480))      # E = 1 TeV

print(f"mindanss: {mindanss:.2e}, maxdanss: {maxdanss:.2e}")
print(f"minsbl: {minsbl:.2e}, maxsbl: {maxsbl:.2e}")
print(f"minlbl: {minlbl:.2e}, maxlbl: {maxlbl:.2e}")
print(f"minsolar: {minsolar:.2e}, maxsolar: {maxsolar:.2e}")
print(f"minastro: {minastro:.2e}, maxastro: {maxastro:.2e}")
print(f"minsnd: {minsnd:.2e}, maxsnd: {maxsnd:.2e}")

# ============================================================================
# CREATE FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# ============================================================================
# PLOT DANSS
# ============================================================================
x_danss_min = np.logspace(np.log10(0.1), np.log10(mindanss), 500)
y_danss_min = ycwlow(1e-3, 0.01) * np.ones_like(x_danss_min)
ax.loglog(x_danss_min, y_danss_min, 
         linestyle='--', 
         color=colorList[1], 
         linewidth=2,
         label='DANSS E = 1 MeV')

x_danss_max = np.logspace(np.log10(0.1), np.log10(maxdanss), 500)
y_danss_max = ycwhigh(1e-2, 0.01) * np.ones_like(x_danss_max)
ax.loglog(x_danss_max, y_danss_max, 
         linestyle='-', 
         color=colorList[1], 
         linewidth=2,
         label='DANSS E = 10 MeV')

# DANSS shaded region
danss_polygon_x = [0.1, maxdanss, mindanss, 0.1]
danss_polygon_y = [ycwhigh(1e-2, 0.01), ycwhigh(1e-2, 0.01), 
                   ycwlow(1e-3, 0.01), ycwlow(1e-3, 0.01)]
danss_poly = Polygon(list(zip(danss_polygon_x, danss_polygon_y)), 
                     facecolor=colorList[1], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(danss_poly)

# ============================================================================
# PLOT SBL (Short Baseline)
# ============================================================================
x_sbl_min = np.logspace(np.log10(0.1), np.log10(minsbl), 500)
y_sbl_min = ycwlow(0.1, 0.110) * np.ones_like(x_sbl_min)
ax.loglog(x_sbl_min, y_sbl_min, 
         linestyle='--', 
         color=colorList[2], 
         linewidth=2,
         label='SBND E = 100 MeV')

x_sbl_max = np.logspace(np.log10(0.1), np.log10(maxsbl), 500)
y_sbl_max = ycwhigh(10, 0.110) * np.ones_like(x_sbl_max)
ax.loglog(x_sbl_max, y_sbl_max, 
         linestyle='-', 
         color=colorList[2], 
         linewidth=2,
         label='SBND E = 10 GeV')

# SBL shaded region
sbl_polygon_x = [0.1, maxsbl, minsbl, 0.1]
sbl_polygon_y = [ycwhigh(10, 0.110), ycwhigh(10, 0.110), 
                 ycwlow(0.1, 0.110), ycwlow(0.1, 0.110)]
sbl_poly = Polygon(list(zip(sbl_polygon_x, sbl_polygon_y)), 
                   facecolor=colorList[2], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(sbl_poly)

# ============================================================================
# PLOT LBL (Long Baseline) - DUNE
# ============================================================================
x_lbl_min = np.logspace(np.log10(0.1), np.log10(minlbl), 500)
y_lbl_min = ycwlow(1e-3, 1300) * np.ones_like(x_lbl_min)
ax.loglog(x_lbl_min, y_lbl_min, 
         linestyle='--', 
         color=colorList[7], 
         linewidth=2,
         label='DUNE E = 1 MeV')

x_lbl_max = np.logspace(np.log10(0.1), np.log10(maxlbl), 500)
y_lbl_max = ycwhigh(10, 1300) * np.ones_like(x_lbl_max)
ax.loglog(x_lbl_max, y_lbl_max, 
         linestyle='-', 
         color=colorList[7], 
         linewidth=2,
         label='DUNE E = 10 GeV')

# LBL shaded region
lbl_polygon_x = [0.1, maxlbl, minlbl, 0.1]
lbl_polygon_y = [ycwhigh(10, 1300), ycwhigh(10, 1300), 
                 ycwlow(1e-3, 1300), ycwlow(1e-3, 1300)]
lbl_poly = Polygon(list(zip(lbl_polygon_x, lbl_polygon_y)), 
                   facecolor=colorList[7], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(lbl_poly)

# ============================================================================
# PLOT SOLAR
# ============================================================================
x_solar_min = np.logspace(np.log10(0.1), np.log10(minsolar), 500)
y_solar_min = ycwlow(1e-4, 1e8) * np.ones_like(x_solar_min)
ax.loglog(x_solar_min, y_solar_min, 
         linestyle='--', 
         color=colorList[10], 
         linewidth=2,
         label='Solar E = 100 keV')

x_solar_max = np.logspace(np.log10(0.1), np.log10(maxsolar), 500)
y_solar_max = ycwhigh(1.8e-2, 1e8) * np.ones_like(x_solar_max)
ax.loglog(x_solar_max, y_solar_max, 
         linestyle='-', 
         color=colorList[10], 
         linewidth=2,
         label='Solar E = 18 MeV')

# Solar shaded region
solar_polygon_x = [0.1, maxsolar, minsolar, 0.1]
solar_polygon_y = [ycwhigh(1.8e-2, 1e8), ycwhigh(1.8e-2, 1e8), 
                   ycwlow(1e-4, 1e8), ycwlow(1e-4, 1e8)]
solar_poly = Polygon(list(zip(solar_polygon_x, solar_polygon_y)), 
                     facecolor=colorList[10], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(solar_poly)

# ============================================================================
# PLOT ASTRO
# ============================================================================
x_astro_min = np.logspace(np.log10(0.1), np.log10(minastro), 500)
y_astro_min = ycwlow(1e3, 1e22) * np.ones_like(x_astro_min)
ax.loglog(x_astro_min, y_astro_min, 
         linestyle='--', 
         color=colorList[4], 
         linewidth=2,
         label='Astro E = 1 TeV')

x_astro_max = np.logspace(np.log10(0.1), np.log10(maxastro), 500)
y_astro_max = ycwhigh(1e6, 1e22) * np.ones_like(x_astro_max)
ax.loglog(x_astro_max, y_astro_max, 
         linestyle='-', 
         color=colorList[4], 
         linewidth=2,
         label='Astro E = 1 PeV')

# Astro shaded region
astro_polygon_x = [0.1, maxastro, minastro, 0.1]
astro_polygon_y = [ycwhigh(1e6, 1e22), ycwhigh(1e6, 1e22), 
                   ycwlow(1e3, 1e22), ycwlow(1e3, 1e22)]
astro_poly = Polygon(list(zip(astro_polygon_x, astro_polygon_y)), 
                     facecolor=colorList[4], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(astro_poly)

# ============================================================================
# PLOT SND
# ============================================================================
x_snd_min = np.logspace(np.log10(0.1), np.log10(minsnd), 500)
y_snd_min = ycwlow(100, 0.48) * np.ones_like(x_snd_min)
ax.loglog(x_snd_min, y_snd_min, 
         linestyle='--', 
         color=colorList[3], 
         linewidth=2,
         label='SND E = 100 GeV')

x_snd_max = np.logspace(np.log10(0.1), np.log10(maxsnd), 500)
y_snd_max = ycwhigh(2000, 0.48) * np.ones_like(x_snd_max)
ax.loglog(x_snd_max, y_snd_max, 
         linestyle='-', 
         color=colorList[3], 
         linewidth=2,
         label='SND E = 1 TeV')

# SND shaded region
snd_polygon_x = [0.1, maxsnd, minsnd, 0.1]
snd_polygon_y = [ycwhigh(2000, 0.48), ycwhigh(2000, 0.48), 
                 ycwlow(100, 0.48), ycwlow(100, 0.48)]
snd_poly = Polygon(list(zip(snd_polygon_x, snd_polygon_y)), 
                   facecolor=colorList[3], alpha=0.3, edgecolor='none', zorder=5)
ax.add_patch(snd_poly)

# ============================================================================
# PLOT PERTURBATIVE BOUNDARY (GRAY REGION) - ON TOP
# ============================================================================
x_pert = np.logspace(np.log10(0.1), np.log10(maxsnd), 500)
y_pert = pert(x_pert)
ax.loglog(x_pert, y_pert, color='gray', linewidth=3, zorder=10)

# Fill gray "Non-Perturbative" region (on top of everything)
ax.fill_between(x_pert, y_pert, 1e-5, color='gray', alpha=1.0, zorder=10)

# Add "Non-Perturbative" label
ax.text(1, 1e-8, 'Non-Perturbative', 
        fontsize=11, fontweight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', edgecolor='none'),
        zorder=11)

# ============================================================================
# CONFIGURE PLOT
# ============================================================================
ax.set_xlabel('m (eV)', fontsize=16, fontweight='bold')
ax.set_ylabel('y', fontsize=16, fontweight='bold')
ax.set_xlim(0.1, 1e7)
ax.set_ylim(1e-25, 1e-5)
ax.tick_params(labelsize=14)
ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.5)

# Add legend INSIDE the plot at upper right
ax.legend(loc='upper right', fontsize=10, framealpha=0.9, 
          bbox_to_anchor=(0.98, 0.38), ncol=1)

plt.tight_layout()
plt.show()