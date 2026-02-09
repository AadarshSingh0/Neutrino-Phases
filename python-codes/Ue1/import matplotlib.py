import numpy as np
import matplotlib.pyplot as plt

# Define constants and functions for each plot

# DANSS (*L = 10m , 1 MeV < E < 10 MeV*)
mindanss = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e6)**3 / 10))
maxdanss = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e7)**3 / 10))
c1danss = 0.1 * 0.01 / (1.27 * 0.01)
c2danss = 0.1 * 0.001 / (1.27 * 0.01)
ymaxdanss = lambda x: c1danss / x**2
ymindanss = lambda x: c2danss / x**2

# SND (*L = 480m , 100 GeV < E < 1 TeV*)
minsnd = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e11)**3 / 480))
maxsnd = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e12)**3 / 480))
c1snd = 0.1 * 1000 / (1.27 * 0.48)
c2snd = 0.1 * 100 / (1.27 * 0.48)
ymaxsnd = lambda x: c1snd / x**2
yminsnd = lambda x: c2snd / x**2

# SBL (*L = 110m , 100 MeV < E < 10 GeV*)
minsbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e8)**3 / 110))
maxsbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e10)**3 / 110))
c1sbl = 0.1 * 10 / (1.27 * 0.11)
c2sbl = 0.1 * 0.1 / (1.27 * 0.11)
ymaxsbl = lambda x: c1sbl / x**2
yminsbl = lambda x: c2sbl / x**2

# LBL (*L = 1300km , 1 MeV < E < 10 GeV*)
minlbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e6)**3 / (1300 * 1e3)))
maxlbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e10)**3 / (1300 * 1e3)))
c1lbl = 0.1 * 10 / (1.27 * 1300)
c2lbl = 0.1 * 0.001 / (1.27 * 1300)
ymaxlbl = lambda x: c1lbl / x**2
yminlbl = lambda x: c2lbl / x**2

# Solar (*L = 10**8km , 100 KeV < E < 18 MeV*)
minsolar = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e5)**3 / (1 * 1e11)))
maxsolar = np.sqrt(np.sqrt(4 * 0.197e-6 * (1.8*1e7)**3 / (1 * 1e11)))
c2solar = 0.1 * 1e-4 / (1.27 * 1e8)
c1solar = 0.1 * 1.8*1e-2 / (1.27 * 1e8)
ymaxsolar = lambda x: c1solar / x**2
yminsolar = lambda x: c2solar / x**2

# Astro (*L = 10**22km , 1 TeV < E < 1 PeV*)
minastro = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e12)**3 / (1e25)))
maxastro = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e15)**3 / (1e25)))
c2astro = 0.1 * 1e3 / (1.27 * 1e22)
c1astro = 0.1 * 1e6 / (1.27 * 1e22)
ymaxastro = lambda x: c1astro / x**2
yminastro = lambda x: c2astro / x**2

# Create a consolidated plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot DANSS
x_danss_min = np.logspace(-1, np.log10(mindanss), 500)
x_danss_max = np.logspace(-1, np.log10(maxdanss), 500)
ax.plot(x_danss_max, ymaxdanss(x_danss_max), label="DANSS E = 10 MeV", linestyle='-', color='blue', linewidth=2)
ax.plot(x_danss_min, ymindanss(x_danss_min), label="DANSS E = 1 MeV", linestyle='--', color='blue', linewidth=2)

# Plot SND
x_snd_min = np.logspace(-1, np.log10(minsnd), 500)
x_snd_max = np.logspace(-1, np.log10(maxsnd), 500)
ax.plot(x_snd_max, ymaxsnd(x_snd_max), label="SND E = 1 TeV", linestyle='-', color='orange', linewidth=2)
ax.plot(x_snd_min, yminsnd(x_snd_min), label="SND E = 100 GeV", linestyle='--', color='orange', linewidth=2)

# Plot SBL
x_sbl_min = np.logspace(-1, np.log10(minsbl), 500)
x_sbl_max = np.logspace(-1, np.log10(maxsbl), 500)
ax.plot(x_sbl_max, ymaxsbl(x_sbl_max), label="SBL E = 10 GeV", linestyle='-', color='green', linewidth=2)
ax.plot(x_sbl_min, yminsbl(x_sbl_min), label="SBL E = 100 MeV", linestyle='--', color='green', linewidth=2)

# Plot LBL
x_lbl_min = np.logspace(-1, np.log10(minlbl), 500)
x_lbl_max = np.logspace(-1, np.log10(maxlbl), 500)
ax.plot(x_lbl_max, ymaxlbl(x_lbl_max), label="LBL E = 10 GeV", linestyle='-', color='red', linewidth=2)
ax.plot(x_lbl_min, yminlbl(x_lbl_min), label="LBL E = 1 MeV", linestyle='--', color='red', linewidth=2)

# Plot Solar
x_solar_min = np.logspace(-1, np.log10(minsolar), 500)
x_solar_max = np.logspace(-1, np.log10(maxsolar), 500)
ax.plot(x_solar_max, ymaxsolar(x_solar_max), label="Solar E = 10 GeV", linestyle='-', color='purple', linewidth=2)
ax.plot(x_solar_min, yminsolar(x_solar_min), label="Solar E = 100 KeV", linestyle='--', color='purple', linewidth=2)

# Plot Astro
x_astro_min = np.logspace(-1, np.log10(minastro), 500)
x_astro_max = np.logspace(-1, np.log10(maxastro), 500)
ax.plot(x_astro_max, ymaxastro(x_astro_max), label="Astro E = 1 PeV", linestyle='-', color='brown', linewidth=2)
ax.plot(x_astro_min, yminastro(x_astro_min), label="Astro E = 1 TeV", linestyle='--', color='brown', linewidth=2)

# Customizing plot
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.1, 1e7)
ax.set_ylim(1e-30, 1)
ax.set_xlabel("Mass (eV)", fontsize=14)
ax.set_ylabel(r"$|U_e4|^2$", fontsize=14)
ax.legend(fontsize=10)

plt.title("Combined Plot of Mass vs $|U_e4|^2$", fontsize=16)
plt.tight_layout()
plt.show()
