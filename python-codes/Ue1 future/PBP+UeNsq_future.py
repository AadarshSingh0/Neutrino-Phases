"""
UeNsq_future.py - 31/03/2023

Summary: 
Code for plotting constraints on the mixing squared between
the electron neutrino and sterile neutrino |U_{eN}|^2 as 
a function of the sterile neutrino mass m_N

References for each individual constraint are compiled
on the 'Plots and Data' page of the website.

Here data with consistent log units are loaded and plotted.

Requires numpy, matplotlib, scipy and pandas.
"""

import numpy as np
from numpy import cos as Cos
from numpy import sin as Sin
from numpy import sqrt as Sqrt
from numpy import ma
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib import ticker, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy.ndimage
import pandas as pd
from colour import Color

### Load data frame for each data set ###
base_path = "/home/as/Downloads/neutrino data/python-codes/data_ue1_fut/data"

df_current_LNC = pd.read_csv(f"{base_path}/current_LNC_data.csv", header=None, sep=",", names=["X", "Y"])
df_current_LNV = pd.read_csv(f"{base_path}/current_LNV_data.csv", header=None, sep=",", names=["X", "Y"])

df_TRISTAN = pd.read_csv(f"{base_path}/TRISTAN_data.csv", header=None, sep=",", names=["X", "Y"])
df_KATRIN = pd.read_csv(f"{base_path}/KATRIN_data.csv", header=None, sep=",", names=["X", "Y"])
df_HUNTER_1 = pd.read_csv(f"{base_path}/HUNTER_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_HUNTER_2 = pd.read_csv(f"{base_path}/HUNTER_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_HUNTER_3 = pd.read_csv(f"{base_path}/HUNTER_3_data.csv", header=None, sep=",", names=["X", "Y"])
df_BeEST_1 = pd.read_csv(f"{base_path}/BeEST_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_BeEST_2 = pd.read_csv(f"{base_path}/BeEST_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_BeEST_3 = pd.read_csv(f"{base_path}/BeEST_3_data.csv", header=None, sep=",", names=["X", "Y"])
df_PIONEER = pd.read_csv(f"{base_path}/PIONEER_data.csv", header=None, sep=",", names=["X", "Y"])
df_SHiP = pd.read_csv(f"{base_path}/SHiP_data.csv", header=None, sep=",", names=["X", "Y"])
df_DUNE_1 = pd.read_csv(f"{base_path}/DUNE_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_DUNE_2 = pd.read_csv(f"{base_path}/DUNE_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_DUNE_Indirect = pd.read_csv(f"{base_path}/DUNE_Indirect_data.csv", header=None, sep=",", names=["X", "Y"])
df_FCC_ee = pd.read_csv(f"{base_path}/FCC_ee_data.csv", header=None, sep=",", names=["X", "Y"])
df_LHCb_disp = pd.read_csv(f"{base_path}/LHCb_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_disp = pd.read_csv(f"{base_path}/ATLAS_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMS_disp = pd.read_csv(f"{base_path}/CMS_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_lept_disp = pd.read_csv(f"{base_path}/lept_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_MATHUSLA_disp = pd.read_csv(f"{base_path}/MATHUSLA_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_FASER_disp = pd.read_csv(f"{base_path}/FASER_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_AL3X_disp = pd.read_csv(f"{base_path}/AL3X_disp_data.csv", header=None, sep=",", names=["X", "Y"])
df_CODEX_b_disp = pd.read_csv(f"{base_path}/CODEX_b_e_data.csv", header=None, sep=",", names=["X", "Y"])
df_ANUBIS_disp = pd.read_csv(f"{base_path}/ANUBIS_data.csv", header=None, sep=",", names=["X", "Y"])
df_NA62 = pd.read_csv(f"{base_path}/NA62_data.csv", header=None, sep=",", names=["X", "Y"])
df_ILC = pd.read_csv(f"{base_path}/ILC_data.csv", header=None, sep=",", names=["X", "Y"])
df_DUNE_osc = pd.read_csv(f"{base_path}/DUNE_osc_data.csv", header=None, sep=",", names=["X", "Y"])
df_JUNO = pd.read_csv(f"{base_path}/JUNO_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATHENA = pd.read_csv(f"{base_path}/ATHENA_data.csv", header=None, sep=",", names=["X", "Y"])
df_eROSITA = pd.read_csv(f"{base_path}/eROSITA_data.csv", header=None, sep=",", names=["X", "Y"])
df_collider_14TeV = pd.read_csv(f"{base_path}/collider_14TeV_data.csv", header=None, sep=",", names=["X", "Y"])
df_collider_100TeV = pd.read_csv(f"{base_path}/collider_100TeV_data.csv", header=None, sep=",", names=["X", "Y"])
df_FCC_he = pd.read_csv(f"{base_path}/FCC_he_data.csv", header=None, sep=",", names=["X", "Y"])
df_LHeC_1 = pd.read_csv(f"{base_path}/LHeC_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_LHeC_2 = pd.read_csv(f"{base_path}/LHeC_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_CLIC_1 = pd.read_csv(f"{base_path}/CLIC_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_CLIC_2 = pd.read_csv(f"{base_path}/CLIC_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_MesonDecays_LNV = pd.read_csv(f"{base_path}/MesonDecays_LNV_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB = pd.read_csv(f"{base_path}/CMB_data.csv", header=None, sep=",", names=["X", "Y"])
# df_CMB_Linear = pd.read_csv(f"{base_path}/CMB_linear_data.csv", header=None, sep=",", names=["X", "Y"])
df_ShiSigl_SN = pd.read_csv(f"{base_path}/ShiSigl_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_BAO_H = pd.read_csv(f"{base_path}/CMB_BAO_H_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_H_only = pd.read_csv(f"{base_path}/CMB_H_only_data.csv", header=None, sep=",", names=["X", "Y"])
df_BBN = pd.read_csv(f"{base_path}/BBN_data.csv", header=None, sep=",", names=["X", "Y"])


### Read to (x,y) = (m_N,|V_{eN}|^2)  ###

x_current_LNC, y_current_LNC = [], []
for i in range(len(df_current_LNC.index)):
    x_current_LNC.append(df_current_LNC.iloc[i]['X'])
    y_current_LNC.append(df_current_LNC.iloc[i]['Y'])

x_current_LNV, y_current_LNV = [], []
for i in range(len(df_current_LNV.index)):
    x_current_LNV.append(df_current_LNV.iloc[i]['X'])
    y_current_LNV.append(df_current_LNV.iloc[i]['Y'])

x_TRISTAN, y_TRISTAN = [], []
for i in range(len(df_TRISTAN.index)):
    x_TRISTAN.append(df_TRISTAN.iloc[i]['X'])
    y_TRISTAN.append(df_TRISTAN.iloc[i]['Y'])

x_KATRIN, y_KATRIN = [], []
for i in range(len(df_KATRIN.index)):
    x_KATRIN.append(df_KATRIN.iloc[i]['X'])
    y_KATRIN.append(df_KATRIN.iloc[i]['Y'])

x_HUNTER_1, y_HUNTER_1 = [], []
for i in range(len(df_HUNTER_1.index)):
    x_HUNTER_1.append(df_HUNTER_1.iloc[i]['X'])
    y_HUNTER_1.append(df_HUNTER_1.iloc[i]['Y'])

x_HUNTER_2, y_HUNTER_2 = [], []
for i in range(len(df_HUNTER_2.index)):
    x_HUNTER_2.append(df_HUNTER_2.iloc[i]['X'])
    y_HUNTER_2.append(df_HUNTER_2.iloc[i]['Y'])

x_HUNTER_3, y_HUNTER_3 = [], []
for i in range(len(df_HUNTER_3.index)):
    x_HUNTER_3.append(df_HUNTER_3.iloc[i]['X'])
    y_HUNTER_3.append(df_HUNTER_3.iloc[i]['Y'])

x_BeEST_1, y_BeEST_1 = [], []
for i in range(len(df_BeEST_1.index)):
    x_BeEST_1.append(df_BeEST_1.iloc[i]['X'])
    y_BeEST_1.append(df_BeEST_1.iloc[i]['Y'])

x_BeEST_2, y_BeEST_2 = [], []
for i in range(len(df_BeEST_2.index)):
    x_BeEST_2.append(df_BeEST_2.iloc[i]['X'])
    y_BeEST_2.append(df_BeEST_2.iloc[i]['Y'])

x_BeEST_3, y_BeEST_3 = [], []
for i in range(len(df_BeEST_3.index)):
    x_BeEST_3.append(df_BeEST_3.iloc[i]['X'])
    y_BeEST_3.append(df_BeEST_3.iloc[i]['Y'])

x_PIONEER, y_PIONEER = [], []
for i in range(len(df_PIONEER.index)):
    x_PIONEER.append(df_PIONEER.iloc[i]['X'])
    y_PIONEER.append(df_PIONEER.iloc[i]['Y'])

x_SHiP, y_SHiP = [], []
for i in range(len(df_SHiP.index)):
    x_SHiP.append(df_SHiP.iloc[i]['X'])
    y_SHiP.append(df_SHiP.iloc[i]['Y'])

x_DUNE_1, y_DUNE_1 = [], []
for i in range(len(df_DUNE_1.index)):
    x_DUNE_1.append(df_DUNE_1.iloc[i]['X'])
    y_DUNE_1.append(df_DUNE_1.iloc[i]['Y'])

x_DUNE_2, y_DUNE_2 = [], []
for i in range(len(df_DUNE_2.index)):
    x_DUNE_2.append(df_DUNE_2.iloc[i]['X'])
    y_DUNE_2.append(df_DUNE_2.iloc[i]['Y'])

x_DUNE_Indirect, y_DUNE_Indirect = [], []
for i in range(len(df_DUNE_Indirect.index)):
    x_DUNE_Indirect.append(df_DUNE_Indirect.iloc[i]['X'])
    y_DUNE_Indirect.append(df_DUNE_Indirect.iloc[i]['Y'])

x_FCC_ee, y_FCC_ee = [], []
for i in range(len(df_FCC_ee.index)):
    x_FCC_ee.append(df_FCC_ee.iloc[i]['X'])
    y_FCC_ee.append(df_FCC_ee.iloc[i]['Y'])

x_LHCb_disp, y_LHCb_disp = [], []
for i in range(len(df_LHCb_disp.index)):
    x_LHCb_disp.append(df_LHCb_disp.iloc[i]['X'])
    y_LHCb_disp.append(df_LHCb_disp.iloc[i]['Y'])

x_ATLAS_disp, y_ATLAS_disp = [], []
for i in range(len(df_ATLAS_disp.index)):
    x_ATLAS_disp.append(df_ATLAS_disp.iloc[i]['X'])
    y_ATLAS_disp.append(df_ATLAS_disp.iloc[i]['Y'])

x_CMS_disp, y_CMS_disp = [], []
for i in range(len(df_CMS_disp.index)):
    x_CMS_disp.append(df_CMS_disp.iloc[i]['X'])
    y_CMS_disp.append(df_CMS_disp.iloc[i]['Y'])

x_lept_disp, y_lept_disp = [], []
for i in range(len(df_lept_disp.index)):
    x_lept_disp.append(df_lept_disp.iloc[i]['X'])
    y_lept_disp.append(df_lept_disp.iloc[i]['Y'])

x_MATHUSLA_disp, y_MATHUSLA_disp = [], []
for i in range(len(df_MATHUSLA_disp.index)):
    x_MATHUSLA_disp.append(df_MATHUSLA_disp.iloc[i]['X'])
    y_MATHUSLA_disp.append(df_MATHUSLA_disp.iloc[i]['Y'])

x_FASER_disp, y_FASER_disp = [], []
for i in range(len(df_FASER_disp.index)):
    x_FASER_disp.append(df_FASER_disp.iloc[i]['X'])
    y_FASER_disp.append(df_FASER_disp.iloc[i]['Y'])

x_AL3X_disp, y_AL3X_disp = [], []
for i in range(len(df_AL3X_disp.index)):
    x_AL3X_disp.append(df_AL3X_disp.iloc[i]['X'])
    y_AL3X_disp.append(df_AL3X_disp.iloc[i]['Y'])

x_CODEX_b_disp, y_CODEX_b_disp = [], []
for i in range(len(df_CODEX_b_disp.index)):
    x_CODEX_b_disp.append(df_CODEX_b_disp.iloc[i]['X'])
    y_CODEX_b_disp.append(df_CODEX_b_disp.iloc[i]['Y'])

x_ANUBIS_disp, y_ANUBIS_disp = [], []
for i in range(len(df_ANUBIS_disp.index)):
    x_ANUBIS_disp.append(df_ANUBIS_disp.iloc[i]['X'])
    y_ANUBIS_disp.append(df_ANUBIS_disp.iloc[i]['Y'])

x_NA62, y_NA62 = [], []
for i in range(len(df_NA62.index)):
    x_NA62.append(df_NA62.iloc[i]['X'])
    y_NA62.append(df_NA62.iloc[i]['Y'])

x_ILC, y_ILC = [], []
for i in range(len(df_ILC.index)):
    x_ILC.append(df_ILC.iloc[i]['X'])
    y_ILC.append(df_ILC.iloc[i]['Y'])

x_DUNE_osc, y_DUNE_osc = [], []
for i in range(len(df_DUNE_osc.index)):
    x_DUNE_osc.append(df_DUNE_osc.iloc[i]['X'])
    y_DUNE_osc.append(df_DUNE_osc.iloc[i]['Y'])

x_JUNO, y_JUNO = [], []
for i in range(len(df_JUNO.index)):
    x_JUNO.append(df_JUNO.iloc[i]['X'])
    y_JUNO.append(df_JUNO.iloc[i]['Y'])

x_ATHENA, y_ATHENA = [], []
for i in range(len(df_ATHENA.index)):
    x_ATHENA.append(df_ATHENA.iloc[i]['X'])
    y_ATHENA.append(df_ATHENA.iloc[i]['Y'])

x_eROSITA, y_eROSITA = [], []
for i in range(len(df_eROSITA.index)):
    x_eROSITA.append(df_eROSITA.iloc[i]['X'])
    y_eROSITA.append(df_eROSITA.iloc[i]['Y'])

x_collider_14TeV, y_collider_14TeV = [], []
for i in range(len(df_collider_14TeV.index)):
    x_collider_14TeV.append(df_collider_14TeV.iloc[i]['X'])
    y_collider_14TeV.append(df_collider_14TeV.iloc[i]['Y'])

x_collider_100TeV, y_collider_100TeV = [], []
for i in range(len(df_collider_100TeV.index)):
    x_collider_100TeV.append(df_collider_100TeV.iloc[i]['X'])
    y_collider_100TeV.append(df_collider_100TeV.iloc[i]['Y'])

x_FCC_he, y_FCC_he = [], []
for i in range(len(df_FCC_he.index)):
    x_FCC_he.append(df_FCC_he.iloc[i]['X'])
    y_FCC_he.append(df_FCC_he.iloc[i]['Y'])

x_LHeC_1, y_LHeC_1 = [], []
for i in range(len(df_LHeC_1.index)):
    x_LHeC_1.append(df_LHeC_1.iloc[i]['X'])
    y_LHeC_1.append(df_LHeC_1.iloc[i]['Y'])

x_CLIC_1, y_CLIC_1 = [], []
for i in range(len(df_CLIC_1.index)):
    x_CLIC_1.append(df_CLIC_1.iloc[i]['X'])
    y_CLIC_1.append(df_CLIC_1.iloc[i]['Y'])

x_CLIC_2, y_CLIC_2 = [], []
for i in range(len(df_CLIC_2.index)):
    x_CLIC_2.append(df_CLIC_2.iloc[i]['X'])
    y_CLIC_2.append(df_CLIC_2.iloc[i]['Y'])

x_LHeC_2, y_LHeC_2 = [], []
for i in range(len(df_LHeC_2.index)):
    x_LHeC_2.append(df_LHeC_2.iloc[i]['X'])
    y_LHeC_2.append(df_LHeC_2.iloc[i]['Y'])

x_MesonDecays_LNV, y_MesonDecays_LNV = [], []
for i in range(len(df_MesonDecays_LNV.index)):
    x_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['X'])
    y_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['Y'])

x_CMB, y_CMB, z_CMB = [], [], []
for i in range(len(df_CMB.index)):
    x_CMB.append(df_CMB.iloc[i]['X'])
    y_CMB.append(df_CMB.iloc[i]['Y'])
    z_CMB.append(df_CMB.iloc[i]['Y']+0.7)

# x_CMB_Linear, y_CMB_Linear, z_CMB_Linear = [], [], []
# for i in range(len(df_CMB_Linear.index)):
#     x_CMB_Linear.append(df_CMB_Linear.iloc[i]['X'])
#     y_CMB_Linear.append(df_CMB_Linear.iloc[i]['Y'])
#     z_CMB_Linear.append(df_CMB_Linear.iloc[i]['Y']+0.7)

x_CMB_BAO_H, y_CMB_BAO_H  = [], []
for i in range(len(df_CMB_BAO_H.index)):
    x_CMB_BAO_H.append(df_CMB_BAO_H.iloc[i]['X'])
    y_CMB_BAO_H.append(df_CMB_BAO_H.iloc[i]['Y'])

x_CMB_BAO_H_2, y_CMB_BAO_H_2  = [], []
for i in range(len(df_CMB_BAO_H.index)):
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[i]['X'])
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[i]['Y'])
for i in range(1,len(df_CMB_BAO_H.index)+1):
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['X'] + 0.5)
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['Y'] + 0.5)

x_CMB_H_only, y_CMB_H_only  = [], []
for i in range(len(df_CMB_H_only.index)):
    x_CMB_H_only.append(df_CMB_H_only.iloc[i]['X'])
    y_CMB_H_only.append(df_CMB_H_only.iloc[i]['Y'])

x_BBN, y_BBN  = [], []
for i in range(len(df_BBN.index)):
    x_BBN.append(df_BBN.iloc[i]['X'])
    y_BBN.append(df_BBN.iloc[i]['Y'])
x_BBN.append(6.84-9)
y_BBN.append(0.1)

x_BBN_2, y_BBN_2  = [], []
x_BBN_2.append(6.84 - 0.5-9)
y_BBN_2.append(0.1)
for i in range(1,len(df_BBN.index)+1):
    x_BBN_2.append(df_BBN.iloc[-i]['X'] - 0.5)
    y_BBN_2.append(df_BBN.iloc[-i]['Y'] - 0.5)
for i in range(len(df_BBN.index)):
    x_BBN_2.append(df_BBN.iloc[i]['X'])
    y_BBN_2.append(df_BBN.iloc[i]['Y'])
x_BBN_2.append(6.84-9)
y_BBN_2.append(0.1)

x_ShiSigl_SN, y_ShiSigl_SN = [], []
for i in range(len(df_ShiSigl_SN.index)):
    x_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['X'])
    y_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['Y'])

# Define constants and functions for each plot


# DANSS (*L = 10m , 1 MeV < E < 10 MeV*)
mindanss = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e6)**3 / 10)) # The minimum mass is determined by the coherent 
#oscillation length being comparable to the baseline, which gives m_N ~ (4 * E^3 / L)^(1/4). Here we take E = 1 MeV and L = 10 m.
maxdanss = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e7)**3 / 10)) # The maximum mass is determined by the coherent 
#oscillation length being comparable to the baseline, which gives m_N ~ (4 * E^3 / L)^(1/4). Here we take E = 1 MeV and L = 10 m.
c1danss = 0.1 * 0.01 / (1.27 * 0.01)
c2danss = 0.1 * 0.001 / (1.27 * 0.01)
c1danss = 10 * 0.01 / (1.27 * 0.01)
c2danss = 0.1 * 0.001 / (1.27 * 0.01)
ymaxdanss = lambda x: c1danss / x**2
ymindanss = lambda x: c2danss / x**2

# FASER (*L = 480m , 100 GeV < E < 1 TeV*)
minsnd = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e11)**3 / 480))
maxsnd = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e12)**3 / 480))
c1snd = 10 * 1000 / (1.27 * 0.48)
c2snd = 0.1 * 100 / (1.27 * 0.48)
ymaxsnd = lambda x: c1snd / x**2
yminsnd = lambda x: c2snd / x**2

# SBL (*L = 110m , 100 MeV < E < 10 GeV*)
minsbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e8)**3 / 110))
maxsbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e10)**3 / 110))
c1sbl = 10 * 10 / (1.27 * 0.11)
c2sbl = 0.1 * 0.1 / (1.27 * 0.11)
ymaxsbl = lambda x: c1sbl / x**2
yminsbl = lambda x: c2sbl / x**2

# LBL (*L = 1300km , 1 MeV < E < 10 GeV*)
minlbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e6)**3 / (1300 * 1e3)))
maxlbl = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e10)**3 / (1300 * 1e3)))
c1lbl = 10 * 10 / (1.27 * 1300)
c2lbl = 0.1 * 0.001 / (1.27 * 1300)
ymaxlbl = lambda x: c1lbl / x**2
yminlbl = lambda x: c2lbl / x**2

# Solar (*L = 10**8km , 100 KeV < E < 18 MeV*)
minsolar = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e5)**3 / (1 * 1e11)))
maxsolar = np.sqrt(np.sqrt(4 * 0.197e-6 * (1.8*1e7)**3 / (1 * 1e11)))
c2solar = 0.1 * 1e-4 / (1.27 * 1e8)
c1solar = 10 * 1.8*1e-2 / (1.27 * 1e8)
ymaxsolar = lambda x: c1solar / x**2
yminsolar = lambda x: c2solar / x**2

# Astro (*L = 10**22km , 1 TeV < E < 1 PeV*)
minastro = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e12)**3 / (1e25)))
maxastro = np.sqrt(np.sqrt(4 * 0.197e-6 * (1e15)**3 / (1e25)))
c2astro = 0.1 * 1e3 / (1.27 * 1e22)
c1astro = 10 * 1e6 / (1.27 * 1e22)
ymaxastro = lambda x: c1astro / x**2
yminastro = lambda x: c2astro / x**2



# Set up the combined figure
fig, ax = plt.subplots(figsize=(20, 16))

axes =ax

spacing=0.2
m = np.arange(-12,6+spacing, spacing)
age_bound = np.log10(1.1 * 10**(-7) * ((50 * 10**(3))/10**(m))**5)
bbn_bound = np.log10(5.55007 * 10**(35) * (1/10**(m))**5)
seesaw_bound = np.log10(0.05*10**(-9)/10**(m))

### Current Constraints ###

axes.plot(m,seesaw_bound,linewidth=1,linestyle='dotted',color='black') # Seesaw line
axes.plot(x_current_LNC,y_current_LNC,linewidth=0.5,linestyle='-.',color='black')
axes.plot(x_current_LNV,y_current_LNV,linewidth=0.5,linestyle='-.',color='black') 

### Future Constraints ###

axes.plot(x_TRISTAN,y_TRISTAN,linewidth=1.5,linestyle='-.',color='mediumslateblue') # TRISTAN
axes.plot(x_KATRIN,y_KATRIN,linewidth=1.5,linestyle='-',color='tomato') # KATRIN
axes.plot(x_HUNTER_1,y_HUNTER_1,linewidth=1.5,linestyle='-',color='mediumvioletred') # HUNTER 1
axes.plot(x_HUNTER_2,y_HUNTER_2,linewidth=1.5,linestyle='--',color='mediumvioletred') # HUNTER 2
axes.plot(x_HUNTER_3,y_HUNTER_3,linewidth=1.5,linestyle='-.',color='mediumvioletred') # HUNTER 3
axes.plot(x_BeEST_1,y_BeEST_1,linewidth=1.5,linestyle='-',color='rosybrown') # BeEST Future 1
axes.plot(x_BeEST_2,y_BeEST_2,linewidth=1.5,linestyle='--',color='rosybrown') # BeEST Future 2
axes.plot(x_BeEST_3,y_BeEST_3,linewidth=1.5,linestyle='-.',color='rosybrown') # BeEST Future 3
axes.plot(x_LHCb_disp,y_LHCb_disp,linewidth=1.5,linestyle='--',color='darkgreen') # LHCb displaced
axes.plot(x_ATLAS_disp,y_ATLAS_disp,linewidth=1.5,linestyle='--',color='blue') # ATLAS displaced
axes.plot(x_CMS_disp,y_CMS_disp,linewidth=1.5,linestyle='--',color='red') # CMS displaced
# axes.plot(x_lept_disp,y_lept_disp,linewidth=1.5,linestyle='-',color='m') # Lepton displaced
axes.plot(x_MATHUSLA_disp,y_MATHUSLA_disp,linewidth=1.5,linestyle='-',color='gold') # MATHUSLA
axes.plot(x_FASER_disp,y_FASER_disp,linewidth=1.5,linestyle='--',color='c') # FASER
axes.plot(x_AL3X_disp,y_AL3X_disp,linewidth=1.5,linestyle='--',color='sienna') # AL3X
axes.plot(x_MesonDecays_LNV,y_MesonDecays_LNV,linewidth=1.5,linestyle='-',color='mediumseagreen') # Future LNV meson decays
axes.plot(x_NA62,y_NA62,linewidth=1.5,linestyle='--',color='teal') # NA62 forcast (beam dump)
axes.plot(x_ILC,y_ILC,linewidth=1.5,linestyle='-',color='fuchsia') # ILC
axes.plot(x_SHiP,y_SHiP,linewidth=1.5,linestyle='-',color='purple') # SHiP
# axes.plot(x_DUNE_1,y_DUNE_1,linewidth=1.5,linestyle='-',color='black') # DUNE ND
axes.plot(x_DUNE_2,y_DUNE_2,linewidth=1.5,linestyle='-',color='navy') # DUNE ND
axes.plot(x_DUNE_Indirect,y_DUNE_Indirect,linewidth=1.5,linestyle=':',color='navy') # DUNE Indirect
axes.plot(x_FCC_ee,y_FCC_ee,linewidth=1.5,linestyle='-',color='limegreen') # FCC-ee
# axes.plot(x_DUNE_osc,y_DUNE_osc,linewidth=1.5,linestyle='-',color='darkred') # DUNE
# axes.plot(x_FCC_he,y_FCC_he,linewidth=1.5,linestyle='-',color='lime') # FCC-he LFV
# axes.plot(x_LHeC_1,y_LHeC_1,linewidth=1.5,linestyle='-',color='indianred') # LHeC LFV
axes.plot(x_LHeC_2,y_LHeC_2,linewidth=1.5,linestyle='--',color='indianred') # LHeC Das
# axes.plot(x_CLIC_1,y_CLIC_1,linewidth=1.5,linestyle='-',color='darkslategrey') # CLIC Mitra
axes.plot(x_CLIC_2,y_CLIC_2,linewidth=1.5,linestyle='-',color='darkslategrey') # CLIC Das
# axes.plot(x_JUNO,y_JUNO,linewidth=1.5,linestyle='-',color='seagreen') # JUNO
axes.plot(x_ATHENA,y_ATHENA,linewidth=1.5,linestyle='-',color='orange') # ATHENA
axes.plot(x_collider_14TeV,y_collider_14TeV,linewidth=1.5,linestyle='--',color='cadetblue') # Future collider 14 TeV 3 inverse ab
axes.plot(x_collider_100TeV,y_collider_100TeV,linewidth=1.5,linestyle='-',color='y') # Future collider 100 TeV 30 inverse ab

axes.plot(x_CMB,y_CMB,linewidth=1,linestyle='-.',color='dimgrey') # Evans data
# axes.plot(x_CMB_Linear,y_CMB_Linear,linewidth=1,linestyle='-.',color='dimgrey') # Linear CMB
axes.plot(x_CMB_BAO_H,y_CMB_BAO_H,linewidth=0.5,linestyle='-',color='grey') # # Decay after BBN constraints
# axes.plot(x_CMB_H_only,y_CMB_H_only,linewidth=1.5,linestyle='--',color='red') # Decay after BBN, Hubble only
axes.plot(x_BBN,y_BBN,linewidth=0.5,linestyle='-',color='grey') # Decay before BBN constraints
axes.plot(x_ShiSigl_SN,y_ShiSigl_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.4) # Shi and Sigl Supernova constraints

# Convert x-axis to log10(GeV) and y-axis to log10 for all experiments

# DANSS
log_x_danss_min = np.log10(np.logspace(-1, np.log10(mindanss), 500) / 1e9)  # Convert x_danss_min to log10(GeV)
log_x_danss_max = np.log10(np.logspace(-1, np.log10(maxdanss), 500) / 1e9)  # Convert x_danss_max to log10(GeV)

log_ymax_danss = np.log10(ymaxdanss(10**(log_x_danss_max) * 1e9))
log_ymin_danss = np.log10(ymindanss(10**(log_x_danss_min) * 1e9))

# SND
log_x_snd_min = np.log10(np.logspace(-1, np.log10(minsnd), 500) / 1e9)  # Convert x_snd_min to log10(GeV)
log_x_snd_max = np.log10(np.logspace(-1, np.log10(maxsnd), 500) / 1e9)  # Convert x_snd_max to log10(GeV)

log_ymax_snd = np.log10(ymaxsnd(10**(log_x_snd_max) * 1e9))
log_ymin_snd = np.log10(yminsnd(10**(log_x_snd_min) * 1e9))

# SBL
log_x_sbl_min = np.log10(np.logspace(-1, np.log10(minsbl), 500) / 1e9)  # Convert x_sbl_min to log10(GeV)
log_x_sbl_max = np.log10(np.logspace(-1, np.log10(maxsbl), 500) / 1e9)  # Convert x_sbl_max to log10(GeV)

log_ymax_sbl = np.log10(ymaxsbl(10**(log_x_sbl_max) * 1e9))
log_ymin_sbl = np.log10(yminsbl(10**(log_x_sbl_min) * 1e9))

# LBL
log_x_lbl_min = np.log10(np.logspace(-1, np.log10(minlbl), 500) / 1e9)  # Convert x_lbl_min to log10(GeV)
log_x_lbl_max = np.log10(np.logspace(-1, np.log10(maxlbl), 500) / 1e9)  # Convert x_lbl_max to log10(GeV)

log_ymax_lbl = np.log10(ymaxlbl(10**(log_x_lbl_max) * 1e9))
log_ymin_lbl = np.log10(yminlbl(10**(log_x_lbl_min) * 1e9))

# Solar
log_x_solar_min = np.log10(np.logspace(-1, np.log10(minsolar), 500) / 1e9)  # Convert x_solar_min to log10(GeV)
log_x_solar_max = np.log10(np.logspace(-1, np.log10(maxsolar), 500) / 1e9)  # Convert x_solar_max to log10(GeV)

log_ymax_solar = np.log10(ymaxsolar(10**(log_x_solar_max) * 1e9))
log_ymin_solar = np.log10(yminsolar(10**(log_x_solar_min) * 1e9))

# Astro
log_x_astro_min = np.log10(np.logspace(-1, np.log10(minastro), 500) / 1e9)  # Convert x_astro_min to log10(GeV)
log_x_astro_max = np.log10(np.logspace(-1, np.log10(maxastro), 500) / 1e9)  # Convert x_astro_max to log10(GeV)

log_ymax_astro = np.log10(ymaxastro(10**(log_x_astro_max) * 1e9))
log_ymin_astro = np.log10(yminastro(10**(log_x_astro_min) * 1e9))

# Plot DANSS
ax.plot(log_x_danss_max, log_ymax_danss, label="DANSS E = 10 MeV", linestyle='-', color='blue', linewidth=2)
ax.plot(log_x_danss_min, log_ymin_danss, label="DANSS E = 1 MeV", linestyle='--', color='blue', linewidth=2)

# Plot SND
ax.plot(log_x_snd_max, log_ymax_snd, label="SND E = 1 TeV", linestyle='-', color='orange', linewidth=2)
ax.plot(log_x_snd_min, log_ymin_snd, label="SND E = 100 GeV", linestyle='--', color='orange', linewidth=2)

# Plot SBL
ax.plot(log_x_sbl_max, log_ymax_sbl, label="SBL E = 10 GeV", linestyle='-', color='green', linewidth=2)
ax.plot(log_x_sbl_min, log_ymin_sbl, label="SBL E = 100 MeV", linestyle='--', color='green', linewidth=2)

# Plot LBL
ax.plot(log_x_lbl_max, log_ymax_lbl, label="LBL E = 10 GeV", linestyle='-', color='red', linewidth=2)
ax.plot(log_x_lbl_min, log_ymin_lbl, label="LBL E = 1 MeV", linestyle='--', color='red', linewidth=2)

# Plot Solar
ax.plot(log_x_solar_max, log_ymax_solar, label="Solar E = 18 MeV", linestyle='-', color='purple', linewidth=2)
ax.plot(log_x_solar_min, log_ymin_solar, label="Solar E = 100 KeV", linestyle='--', color='purple', linewidth=2)

# Plot Astro
ax.plot(log_x_astro_max, log_ymax_astro, label="Astro E = 1 PeV", linestyle='-', color='brown', linewidth=2)
ax.plot(log_x_astro_min, log_ymin_astro, label="Astro E = 1 TeV", linestyle='--', color='brown', linewidth=2)


### Shading ###

plt.fill_between(x_current_LNC,0.2,y_current_LNC, facecolor='k', alpha=0.075)
plt.fill_between(x_current_LNV,0.2,y_current_LNV, facecolor='grey', alpha=0.075)
plt.fill_between(x_CMB,y_CMB,z_CMB, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_CMB_BAO_H_2,0.1,y_CMB_BAO_H_2, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_BBN_2,0.1,y_BBN_2, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_CMB_Linear,y_CMB_Linear,z_CMB_Linear,facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_ShiSigl_SN,-3.6,y_ShiSigl_SN,facecolor='darkslateblue', alpha=0.01,lw=0)

## Filling for PBP
x_danss = [log_x_danss_max[0], log_x_danss_min[0], log_x_danss_min[-1], log_x_danss_max[-1]]
y_danss = [log_ymax_danss[0], log_ymin_danss[0], log_ymin_danss[-1], log_ymax_danss[-1]]
plt.fill(x_danss, y_danss, color='lightblue', alpha=0.5, label='DANSS Region')

## Filling for SND
x_snd = [log_x_snd_max[0], log_x_snd_min[0], log_x_snd_min[-1], log_x_snd_max[-1]]
y_snd = [log_ymax_snd[0], log_ymin_snd[0], log_ymin_snd[-1], log_ymax_snd[-1]]
plt.fill(x_snd, y_snd, color='lightcoral', alpha=0.5, label='SND Region')

## Filling for SBL
x_sbl = [log_x_sbl_max[0], log_x_sbl_min[0], log_x_sbl_min[-1], log_x_sbl_max[-1]]
y_sbl = [log_ymax_sbl[0], log_ymin_sbl[0], log_ymin_sbl[-1], log_ymax_sbl[-1]]
plt.fill(x_sbl, y_sbl, color='palegreen', alpha=0.5, label='SBL Region')

## Filling for LBL
x_lbl = [log_x_lbl_max[0], log_x_lbl_min[0], log_x_lbl_min[-1], log_x_lbl_max[-1]]
y_lbl = [log_ymax_lbl[0], log_ymin_lbl[0], log_ymin_lbl[-1], log_ymax_lbl[-1]]
plt.fill(x_lbl, y_lbl, color='plum', alpha=0.5, label='LBL Region')

## Filling for Solar
x_solar = [log_x_solar_max[0], log_x_solar_min[0], log_x_solar_min[-1], log_x_solar_max[-1]]
y_solar = [log_ymax_solar[0], log_ymin_solar[0], log_ymin_solar[-1], log_ymax_solar[-1]]
plt.fill(x_solar, y_solar, color='khaki', alpha=0.5, label='Solar Region')

## Filling for Astro
x_astro = [log_x_astro_max[0], log_x_astro_min[0], log_x_astro_min[-1], log_x_astro_max[-1]]
y_astro = [log_ymax_astro[0], log_ymin_astro[0], log_ymin_astro[-1], log_ymax_astro[-1]]
plt.fill(x_astro, y_astro, color='lightsteelblue', alpha=0.5, label='Astro Region')

### Labels ###

plt.text(10.7-9, -11.2, r'$\mathrm{FCC-ee} $',fontsize=16,rotation=0,color='limegreen')
# plt.text(11.6-9, -6.6, r'$\mathrm{FCC-he} $',fontsize=16,rotation=0,color='lime')
plt.text(9.1-9, -10.65, r'$\mathrm{SHiP} $',fontsize=16,rotation=0,color='purple')
plt.text(8.25-9, -10.8, r'$\mathrm{DUNE} $',fontsize=16,rotation=0,color='navy')
plt.text(10.65-9, -8.5, r'$\mathrm{ATLAS} $',fontsize=15,rotation=0,color='blue')
plt.text(10.18-9, -10.0, r'$\mathrm{CMS} $',fontsize=14,rotation=0,color='red')
plt.text(9.8-9, -7.3, r'$\mathrm{LHCb} $',fontsize=15,rotation=300,color='darkgreen')
plt.text(9.35-9, -8.9, r'$\mathrm{MATHUSLA} $',fontsize=15,rotation=300,color='gold')
plt.text(8.25-9, -4.9, r'$\mathrm{FASER2} $',fontsize=15,rotation=0,color='c')
plt.text(7.4-9, -6.0, r'$\mathrm{AL3X} $',fontsize=15,rotation=0,color='sienna')
plt.text(6.7-9, -3.2, r'$\mathrm{NA62} $',fontsize=15,rotation=0,color='teal')
plt.text(4.25-9, -3.8, r'$\mathrm{KATRIN} $',fontsize=15,rotation=0,color='tomato')
plt.text(4.25-9, -7.4, r'$\mathrm{TRISTAN} $',fontsize=15,rotation=0,color='mediumslateblue')
plt.text(11.2-9, -4.7, r'$\mathrm{ILC} $',fontsize=16,rotation=0,color='fuchsia')
plt.text(5.5-9, -11.5, r'$\mathrm{CMB}+\mathrm{BAO}+H_0$',fontsize=16,rotation=0,color='grey')
plt.text(9.4-9, -11.7, r'$\mathrm{BBN}$',fontsize=16,rotation=0,color='grey')
# plt.text(-0.9-9, -1.7, r'$\mathrm{JUNO}$',fontsize=16,rotation=0,color='seagreen')
plt.text(2.15-9, -10.3, r'$\mathrm{ATHENA}$',fontsize=16,rotation=0,color='orange')
plt.text(2.25-9, -11.4, r'$\mathrm{eROSITA}$',fontsize=16,rotation=0,color='olivedrab')
plt.text(10.7-9, -11.8, r'$\mathrm{Seesaw}$',fontsize=16,rotation=0,color='black')
plt.text(0.2-9, -5.4, r'$\mathrm{CMB}$',fontsize=16,rotation=0,color='dimgrey')
plt.text(12.15-9, -2.35, r'$\mathrm{FCC-hh}$',fontsize=14,rotation=0,color='y')
plt.text(12.1-9, -1.1, r'$\mathrm{HL-LHC}$',fontsize=14,rotation=0,color='cadetblue')
plt.text(7.88-9, -4.45, r'$\mathrm{Future\,LNV}$',fontsize=12,rotation=90,color='mediumseagreen')
plt.text(12.15-9, -3.7, r'$\mathrm{LHeC} $',fontsize=16,rotation=0,color='indianred')
plt.text(12.3-9, -5.1, r'$\mathrm{CLIC} $',fontsize=16,rotation=0,color='darkslategrey')
plt.text(7-9, -10.65, r'$\mathrm{PIONEER}$',fontsize=16,rotation=0,color='olive')
plt.text(6.9-9, -0.80, r'$\mathrm{DUNE}\,\mathrm{Indirect}$',fontsize=12,rotation=0,color='navy')
plt.text(4.5-9, -10.0, r'$\mathrm{HUNTER}$',fontsize=16,rotation=0,color='mediumvioletred')
plt.text(6-9, -6.2, r'$\mathrm{BeEST} $',fontsize=15,rotation=0,color='rosybrown')

axes.set_xticks([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
axes.xaxis.set_ticklabels([r'',r'$10^{-9}$',r'',r'',r'$10^{-6}$',r'',r'',r'$10^{-3}$',r'',r'',r'$1$',r'',r'',r'$10^{3}$',r'',r'',r'$10^{6}$'],fontsize =16)
ax.set_yticks([i for i in range(-30, 7)])
ax.yaxis.set_ticklabels([
    r'$10^{-30}$', r'', r'', r'$10^{-27}$', r'', r'', r'$10^{-24}$', r'', r'', r'$10^{-21}$',
    r'', r'', r'$10^{-18}$', r'', r'', r'$10^{-15}$', r'', r'', r'$10^{-12}$', r'', r'', r'$10^{-9}$',
    r'', r'', r'$10^{-6}$', r'', r'', r'$10^{-3}$', r'', r'', r'$1$', r'', r'', r'$10^{3}$',
    r'', r'', r'$10^{6}$'
], fontsize=12)
axes.tick_params(axis='x', which='major', pad=7.5)

axes.set_ylabel(r'$|U_{eN}|^2$',fontsize=20,rotation=90)
axes.set_xlabel(r'$m_N \, [\mathrm{GeV}]$',fontsize=20,rotation=0)

axes.xaxis.set_label_coords(0.52,-0.08)
axes.yaxis.set_label_coords(-0.09,0.5)
axes.set_xlim(-9.1,4.1)
axes.set_ylim(-30.1,0.1)

### Set aspect ratio (golden ratio) ###

x0,x1 = axes.get_xlim()
y0,y1 = axes.get_ylim()
axes.set_aspect(2*(x1-x0)/(1+Sqrt(5))/(y1-y0))


plt.legend(
    loc='lower right',
    fontsize=12,
    title=r"$\mathbf{PBP\ Bounds}$",  # Bold and math-styled
    title_fontsize=14,
    frameon=True,  # Add a frame around the legend
    edgecolor='black',  # Frame color
    facecolor='whitesmoke',  # Background color
    shadow=True  # Add a shadow for effect
)

# Define the path and file name
save_path = "/home/as/Documents/Wolfram_Mathematica/Codes/Geometric Phase/data/Ue4_future_sterile.png"

# Save the plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()
# plt.savefig("../../../plots/UeNsq_future.pdf",bbox_inches='tight')
# plt.savefig("../../../plots/UeNsq_future.png",bbox_inches='tight')