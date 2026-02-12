import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# ============================================================================
# CREATE FIGURE
# ============================================================================
fig, ax = plt.subplots(figsize=(12, 10))

# ============================================================================
# EXPERIMENTAL REGIONS (Colored Boxes)
# ============================================================================

# Reactor experiments (purple/magenta)
reactor_box = Rectangle((1e0, 1.5e6), width=1e5-1e0, height=2.1e7-2e6, 
                        facecolor='mediumpurple', edgecolor='black', 
                        linewidth=0.2, alpha=0.6, zorder=5)
ax.add_patch(reactor_box)
ax.text(5e2, 3e6, 'Reactor', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white', zorder=6)

# Accelerator experiments (green)
acc_box = Rectangle((1e1, 2e7), width=1.5e6-1e4, height=1e10-1e8, 
                    facecolor='green', edgecolor='black', 
                    linewidth=0.2, alpha=0.6, zorder=5)
ax.add_patch(acc_box)
ax.text(3e3, 3e8, 'Accelerator', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white', zorder=6)

# Atmospheric experiments (blue)
atm_box = Rectangle((1e4, 2e7), width=1e7-1e5, height=1e11-1e9, 
                    facecolor='royalblue', edgecolor='black', 
                    linewidth=0.2, alpha=0.6, zorder=5)
ax.add_patch(atm_box)
ax.text(3e5, 3e10, 'Atmospheric', fontsize=10, fontweight='bold', 
        ha='center', va='center', color='white', zorder=6)

# High Energy (HE) - dark blue/purple box upper right
he_box = Rectangle((1e22, 1e13), width=3e25-1e24, height=1e16-1e14, 
                   facecolor='darkslateblue', edgecolor='black', 
                   linewidth=0.2, alpha=0.7, zorder=5)
ax.add_patch(he_box)
ax.text(1e24, 3e14, 'HE', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='white', zorder=6)

# Solar (yellow vertical bar)
sun_box = Rectangle((1.5e11, 2e0), width=1e11-1e10, height=1e7-1e5, 
                    facecolor='yellow', edgecolor='black', 
                    linewidth=0.2, alpha=0.7, zorder=5)
ax.add_patch(sun_box)
ax.text(3e10, 3e3, 'Sun', fontsize=12, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

# Supernova (SN) - cyan box
sn_box = Rectangle((1e18, 1.5e6), width=1e21-1e19, height=8e7-1e7, 
                   facecolor='cyan', edgecolor='black', 
                   linewidth=0, alpha=0.6, zorder=5)
ax.add_patch(sn_box)
ax.text(3e19, 8e6, 'SN 1987', fontsize=11, fontweight='bold', 
        ha='center', va='center', color='black', zorder=6)

# DSNB (Diffuse Supernova Neutrino Background) - dark purple
dsnb_box = Rectangle((1e23, 1e7), width=3e25-1e24, height=6e7-1e6, 
                     facecolor='indigo', edgecolor='black', 
                     linewidth=0.2, alpha=0.7, zorder=5)
ax.add_patch(dsnb_box)
ax.text(2e24, 2e7, 'DSNB', fontsize=10, fontweight='bold', 
        ha='center', va='center', color='white', zorder=6)

# # CνB (Cosmic neutrino Background) - light blue vertical bar on right
# cvb_box = Rectangle((1e26, 1e-5), width=1e27-1e26, height=1e-2-1e-5, 
#                     facecolor='deepskyblue', edgecolor='black', 
#                     linewidth=1.5, alpha=0.7, zorder=5)
# ax.add_patch(cvb_box)
# ax.text(3e26, 1e-3, 'CνB', fontsize=11, fontweight='bold', 
#         ha='center', va='center', rotation=90, color='black', zorder=6)

# ============================================================================
# OBSERVABLE UNIVERSE (Gray region on right)
# ============================================================================
obs_universe = Rectangle((2e26, 1e-6), width=1e29-1e27, height=1e17-1e-6, 
                        facecolor='gray', edgecolor='black', 
                        linewidth=2, alpha=0.5, zorder=3)
ax.add_patch(obs_universe)
ax.text(5e27, 1e5, 'Observable Universe', fontsize=11, fontweight='bold', 
        ha='center', va='center', rotation=90, color='white', zorder=6)

# ============================================================================
# DIAGONAL CONTOUR LINES - Log(Δm² [eV²])
# ============================================================================
# These represent constant Δm² lines
# The relation is: Δm² ≈ 2.48 × E_ν [eV] / L [m]
# So: log₁₀(Δm²) = log₁₀(E) - log₁₀(L) + log₁₀(2.48)
# For constant Δm²: log₁₀(E) = log₁₀(L) + log₁₀(Δm²) - log₁₀(2.48)
triangle_coords = np.array([[1e0, 8e6], [1.4e9, 1e16], [1e0, 1e16]])
ax.fill(triangle_coords[:,0], triangle_coords[:,1], 'lightgray', alpha=0.5)

ax.text(5e3, 1e12, r'$m_{\nu}> 1~\mathrm{eV}$', fontsize=11, fontweight='bold', 
        ha='center', va='center', rotation=45, color='black', zorder=6)


# Red dashed lines (main contours)
log_dm2_values_red = [-3]
L_range = np.logspace(0, 28, 1000)

for log_dm2 in log_dm2_values_red:
    dm2 = 2.5*10**log_dm2
    # E = Δm² × L / 1.27
    E_line = 1e6 * dm2 * L_range / 1.27
    ax.loglog(L_range, E_line, 'b--', linewidth=1.5, alpha=0.7, zorder=2)
        
ax.text(4e9, 1e12, r'$|\Delta m_{31}^2| $', fontsize=11, fontweight='bold', 
        ha='center', va='center', rotation=45, color='blue', zorder=6)


# Black dotted lines (secondary grid)
log_dm2_values_black = [1, -2, -5, -8, -11, -14, -17, -20, -23, -26, -29, -32, -35]
temp=-3.3
for log_dm2 in log_dm2_values_black:
    dm2 = 10**log_dm2
    E_line = 1e6 * dm2 * L_range / 1.27
    ax.loglog(L_range, E_line, 'k:', linewidth=0.8, alpha=0.5, zorder=1)

    if log_dm2 >= -20:
        ypos=1e13
        xpos=10**temp * 5e9
    else:
        ypos=1e-3
        xpos=10**temp * 5e-8
    

    ax.text(xpos,ypos, f'{log_dm2}', fontsize=9, color='black', 
           fontweight='bold', rotation=45, ha='center', va='center', zorder=7)
    temp = temp + 3

# ============================================================================
# AXES FORMATTING
# ============================================================================
ax.set_xlabel('L [m]', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$E_\nu$ [eV]', fontsize=16, fontweight='bold')
ax.set_xlim(1e0, 1e28)
ax.set_ylim(1e-6, 1e16)
ax.tick_params(labelsize=13)
ax.grid(False)

# Set ticks
ax.set_xticks([1e0, 1e5, 1e10, 1e15, 1e20, 1e25])
ax.set_yticks([1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14, 1e16])

plt.tight_layout()
plt.show()

print("Neutrino oscillation parameter space plot created successfully!")