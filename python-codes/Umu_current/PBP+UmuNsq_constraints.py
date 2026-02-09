"""
UmuNsq_constraints.py - 31/03/2023

Summary: 
Code for plotting constraints on the mixing squared between
the muon neutrino and sterile neutrino |U_{\mu N}|^2 as 
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

base_path = "/home/as/Downloads/neutrino data/data_Umu_current/data"


base_path = "/home/as/Downloads/neutrino data/data_Umu_current/data"

df_IceCube = pd.read_csv(f"{base_path}/IceCube_data.csv", header=None, sep=",", names=["X", "Y"])
df_MiniBooNE = pd.read_csv(f"{base_path}/MiniBooNE_data.csv", header=None, sep=",", names=["X", "Y"])
df_SBN = pd.read_csv(f"{base_path}/SBN_data.csv", header=None, sep=",", names=["X", "Y"])
df_CDHS = pd.read_csv(f"{base_path}/CDHS_data.csv", header=None, sep=",", names=["X", "Y"])
df_CCFR = pd.read_csv(f"{base_path}/CCFR_data.csv", header=None, sep=",", names=["X", "Y"])
df_MINOS = pd.read_csv(f"{base_path}/MINOS_data.csv", header=None, sep=",", names=["X", "Y"])
df_NOvA = pd.read_csv(f"{base_path}/NOvA_data.csv", header=None, sep=",", names=["X", "Y"])
df_IC_DC = pd.read_csv(f"{base_path}/IC_DC_data.csv", header=None, sep=",", names=["X", "Y"])
df_SuperK_mu = pd.read_csv(f"{base_path}/SuperK_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_MuSpectrum = pd.read_csv(f"{base_path}/MuSpectrum_data.csv", header=None, sep=",", names=["X", "Y"])
df_PIENU_mu = pd.read_csv(f"{base_path}/PIENU_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_Belle_mu = pd.read_csv(f"{base_path}/Belle_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_LHCb = pd.read_csv(f"{base_path}/LHCb_data.csv", header=None, sep=",", names=["X", "Y"])
df_CHARM_mu = pd.read_csv(f"{base_path}/CHARM_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_BEBC = pd.read_csv(f"{base_path}/BEBC_data.csv", header=None, sep=",", names=["X", "Y"])
df_NuTeV = pd.read_csv(f"{base_path}/NuTeV_data.csv", header=None, sep=",", names=["X", "Y"])
df_FMMF = pd.read_csv(f"{base_path}/FMMF_data.csv", header=None, sep=",", names=["X", "Y"])
df_NA3_mu = pd.read_csv(f"{base_path}/NA3_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_PS191_mu = pd.read_csv(f"{base_path}/PS191_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_E949 = pd.read_csv(f"{base_path}/E949_data.csv", header=None, sep=",", names=["X", "Y"])
df_NA62_mu = pd.read_csv(f"{base_path}/NA62_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_KEK = pd.read_csv(f"{base_path}/KEK_data.csv", header=None, sep=",", names=["X", "Y"])
df_T2K_mu = pd.read_csv(f"{base_path}/T2K_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_SuperK_mu_decay = pd.read_csv(f"{base_path}/SuperK_mu_decay_data.csv", header=None, sep=",", names=["X", "Y"])
df_PSI = pd.read_csv(f"{base_path}/PSI_data.csv", header=None, sep=",", names=["X", "Y"])
df_MicroBooNE = pd.read_csv(f"{base_path}/MicroBooNE_data.csv", header=None, sep=",", names=["X", "Y"])
df_MicroBooNE_2 = pd.read_csv(f"{base_path}/MicroBooNE_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_MesonDecays_LNV_mu = pd.read_csv(f"{base_path}/MesonDecays_LNV_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_DELPHI = pd.read_csv(f"{base_path}/DELPHI_data.csv", header=None, sep=",", names=["X", "Y"])
df_L3_mu = pd.read_csv(f"{base_path}/L3_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_1_mu = pd.read_csv(f"{base_path}/ATLAS_1_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_2_mu = pd.read_csv(f"{base_path}/ATLAS_2_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_3_mu = pd.read_csv(f"{base_path}/ATLAS_3_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMS_mu = pd.read_csv(f"{base_path}/CMS_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_Higgs_mu = pd.read_csv(f"{base_path}/Higgs_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_EWPD_mu = pd.read_csv(f"{base_path}/EWPD_mu_data.csv", header=None, sep=",", names=["X", "Y"])
df_Planck = pd.read_csv(f"{base_path}/Planck_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB = pd.read_csv(f"{base_path}/CMB_data.csv", header=None, sep=",", names=["X", "Y"])
df_Xray = pd.read_csv(f"{base_path}/Xray_data.csv", header=None, sep=",", names=["X", "Y"])
df_Xray_2 = pd.read_csv(f"{base_path}/Xray_2_data.csv", header=None, sep=",", names=["X", "Y"])
# df_CMB_Linear = pd.read_csv(f"{base_path}/CMB_linear_data.csv", header=None, sep=",", names=["X", "Y"])
df_Kopp_SN = pd.read_csv(f"{base_path}/Kopp_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_Kopp_SN_2 = pd.read_csv(f"{base_path}/Kopp_SN_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_BAO_H = pd.read_csv(f"{base_path}/CMB_BAO_H_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_H_only = pd.read_csv(f"{base_path}/CMB_H_only_data.csv", header=None, sep=",", names=["X", "Y"])
df_BBN = pd.read_csv(f"{base_path}/BBN_data.csv", header=None, sep=",", names=["X", "Y"])


### Read to (x,y) = (m_N,|V_{eN}|^2)  ###

x_IceCube, y_IceCube = [], []
for i in range(len(df_IceCube.index)):
    x_IceCube.append(df_IceCube.iloc[i]['X'])
    y_IceCube.append(df_IceCube.iloc[i]['Y'])

x_MiniBooNE, y_MiniBooNE = [], []
for i in range(len(df_MiniBooNE.index)):
    x_MiniBooNE.append(df_MiniBooNE.iloc[i]['X'])
    y_MiniBooNE.append(df_MiniBooNE.iloc[i]['Y'])

x_SBN, y_SBN = [], []
for i in range(len(df_SBN.index)):
    x_SBN.append(df_SBN.iloc[i]['X'])
    y_SBN.append(df_SBN.iloc[i]['Y'])

x_CDHS, y_CDHS = [], []
for i in range(len(df_CDHS.index)):
    x_CDHS.append(df_CDHS.iloc[i]['X'])
    y_CDHS.append(df_CDHS.iloc[i]['Y'])

x_CCFR, y_CCFR = [], []
for i in range(len(df_CCFR.index)):
    x_CCFR.append(df_CCFR.iloc[i]['X'])
    y_CCFR.append(df_CCFR.iloc[i]['Y'])

x_MINOS, y_MINOS = [], []
for i in range(len(df_MINOS.index)):
    x_MINOS.append(df_MINOS.iloc[i]['X'])
    y_MINOS.append(df_MINOS.iloc[i]['Y'])

x_NOvA, y_NOvA = [], []
for i in range(len(df_NOvA.index)):
    x_NOvA.append(df_NOvA.iloc[i]['X'])
    y_NOvA.append(df_NOvA.iloc[i]['Y'])

x_IC_DC, y_IC_DC = [], []
for i in range(len(df_IC_DC.index)):
    x_IC_DC.append(df_IC_DC.iloc[i]['X'])
    y_IC_DC.append(df_IC_DC.iloc[i]['Y'])

x_SuperK_mu, y_SuperK_mu = [], []
for i in range(len(df_SuperK_mu.index)):
    x_SuperK_mu.append(df_SuperK_mu.iloc[i]['X'])
    y_SuperK_mu.append(df_SuperK_mu.iloc[i]['Y'])

x_MuSpectrum, y_MuSpectrum = [], []
for i in range(len(df_MuSpectrum.index)):
    x_MuSpectrum.append(df_MuSpectrum.iloc[i]['X'])
    y_MuSpectrum.append(df_MuSpectrum.iloc[i]['Y'])

x_PIENU_mu, y_PIENU_mu = [], []
for i in range(len(df_PIENU_mu.index)):
    x_PIENU_mu.append(df_PIENU_mu.iloc[i]['X'])
    y_PIENU_mu.append(df_PIENU_mu.iloc[i]['Y'])

x_Belle_mu, y_Belle_mu = [], []
for i in range(len(df_Belle_mu.index)):
    x_Belle_mu.append(df_Belle_mu.iloc[i]['X'])
    y_Belle_mu.append(df_Belle_mu.iloc[i]['Y'])

x_LHCb, y_LHCb = [], []
for i in range(len(df_LHCb.index)):
    x_LHCb.append(df_LHCb.iloc[i]['X'])
    y_LHCb.append(df_LHCb.iloc[i]['Y'])

x_CHARM_mu, y_CHARM_mu = [], []
for i in range(len(df_CHARM_mu.index)):
    x_CHARM_mu.append(df_CHARM_mu.iloc[i]['X'])
    y_CHARM_mu.append(df_CHARM_mu.iloc[i]['Y'])

x_BEBC, y_BEBC = [], []
for i in range(len(df_BEBC.index)):
    x_BEBC.append(df_BEBC.iloc[i]['X'])
    y_BEBC.append(df_BEBC.iloc[i]['Y'])

x_NuTeV, y_NuTeV = [], []
for i in range(len(df_NuTeV.index)):
    x_NuTeV.append(df_NuTeV.iloc[i]['X'])
    y_NuTeV.append(df_NuTeV.iloc[i]['Y'])

x_FMMF, y_FMMF = [], []
for i in range(len(df_FMMF.index)):
    x_FMMF.append(df_FMMF.iloc[i]['X'])
    y_FMMF.append(df_FMMF.iloc[i]['Y'])

x_NA3_mu, y_NA3_mu = [], []
for i in range(len(df_NA3_mu.index)):
    x_NA3_mu.append(df_NA3_mu.iloc[i]['X'])
    y_NA3_mu.append(df_NA3_mu.iloc[i]['Y'])

x_PS191_mu, y_PS191_mu = [], []
for i in range(len(df_PS191_mu.index)):
    x_PS191_mu.append(df_PS191_mu.iloc[i]['X'])
    y_PS191_mu.append(df_PS191_mu.iloc[i]['Y'])

x_E949, y_E949 = [], []
for i in range(len(df_E949.index)):
    x_E949.append(df_E949.iloc[i]['X'])
    y_E949.append(df_E949.iloc[i]['Y'])

x_NA62_mu, y_NA62_mu = [], []
for i in range(len(df_NA62_mu.index)):
    x_NA62_mu.append(df_NA62_mu.iloc[i]['X'])
    y_NA62_mu.append(df_NA62_mu.iloc[i]['Y'])

x_KEK, y_KEK = [], []
for i in range(len(df_KEK.index)):
    x_KEK.append(df_KEK.iloc[i]['X'])
    y_KEK.append(df_KEK.iloc[i]['Y'])

x_T2K_mu, y_T2K_mu = [], []
for i in range(len(df_T2K_mu.index)):
    x_T2K_mu.append(df_T2K_mu.iloc[i]['X'])
    y_T2K_mu.append(df_T2K_mu.iloc[i]['Y'])

x_SuperK_mu_decay, y_SuperK_mu_decay = [], []
for i in range(len(df_SuperK_mu_decay.index)):
    x_SuperK_mu_decay.append(df_SuperK_mu_decay.iloc[i]['X'])
    y_SuperK_mu_decay.append(df_SuperK_mu_decay.iloc[i]['Y'])

x_PSI, y_PSI = [], []
for i in range(len(df_PSI.index)):
    x_PSI.append(df_PSI.iloc[i]['X'])
    y_PSI.append(df_PSI.iloc[i]['Y'])

x_MicroBooNE, y_MicroBooNE = [], []
for i in range(len(df_MicroBooNE.index)):
    x_MicroBooNE.append(df_MicroBooNE.iloc[i]['X'])
    y_MicroBooNE.append(df_MicroBooNE.iloc[i]['Y'])

x_MicroBooNE_2, y_MicroBooNE_2 = [], []
for i in range(len(df_MicroBooNE_2.index)):
    x_MicroBooNE_2.append(df_MicroBooNE_2.iloc[i]['X'])
    y_MicroBooNE_2.append(df_MicroBooNE_2.iloc[i]['Y'])

x_MesonDecays_LNV_mu, y_MesonDecays_LNV_mu = [], []
for i in range(len(df_MesonDecays_LNV_mu.index)):
    x_MesonDecays_LNV_mu.append(df_MesonDecays_LNV_mu.iloc[i]['X'])
    y_MesonDecays_LNV_mu.append(df_MesonDecays_LNV_mu.iloc[i]['Y'])

x_DELPHI, y_DELPHI = [], []
for i in range(len(df_DELPHI.index)):
    x_DELPHI.append(df_DELPHI.iloc[i]['X'])
    y_DELPHI.append(df_DELPHI.iloc[i]['Y'])

x_L3_mu, y_L3_mu = [], []
for i in range(len(df_L3_mu.index)):
    x_L3_mu.append(df_L3_mu.iloc[i]['X'])
    y_L3_mu.append(df_L3_mu.iloc[i]['Y'])

x_ATLAS_1_mu, y_ATLAS_1_mu = [], []
for i in range(len(df_ATLAS_1_mu.index)):
    x_ATLAS_1_mu.append(df_ATLAS_1_mu.iloc[i]['X'])
    y_ATLAS_1_mu.append(df_ATLAS_1_mu.iloc[i]['Y'])

x_ATLAS_2_mu, y_ATLAS_2_mu = [], []
for i in range(len(df_ATLAS_2_mu.index)):
    x_ATLAS_2_mu.append(df_ATLAS_2_mu.iloc[i]['X'])
    y_ATLAS_2_mu.append(df_ATLAS_2_mu.iloc[i]['Y'])

x_ATLAS_3_mu, y_ATLAS_3_mu = [], []
for i in range(len(df_ATLAS_3_mu.index)):
    x_ATLAS_3_mu.append(df_ATLAS_3_mu.iloc[i]['X'])
    y_ATLAS_3_mu.append(df_ATLAS_3_mu.iloc[i]['Y'])

x_CMS_mu, y_CMS_mu = [], []
for i in range(len(df_CMS_mu.index)):
    x_CMS_mu.append(df_CMS_mu.iloc[i]['X'])
    y_CMS_mu.append(df_CMS_mu.iloc[i]['Y'])

x_Higgs_mu, y_Higgs_mu = [], []
for i in range(len(df_Higgs_mu.index)):
    x_Higgs_mu.append(df_Higgs_mu.iloc[i]['X'])
    y_Higgs_mu.append(df_Higgs_mu.iloc[i]['Y'])

x_EWPD_mu, y_EWPD_mu = [], []
for i in range(len(df_EWPD_mu.index)):
    x_EWPD_mu.append(df_EWPD_mu.iloc[i]['X'])
    y_EWPD_mu.append(df_EWPD_mu.iloc[i]['Y'])

x_Planck, y_Planck = [], []
for i in range(len(df_Planck.index)):
    x_Planck.append(df_Planck.iloc[i]['X'])
    y_Planck.append(df_Planck.iloc[i]['Y'])

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

x_Xray, y_Xray, z_Xray = [], [], []
for i in range(len(df_Xray.index)):
    x_Xray.append(df_Xray.iloc[i]['X'])
    y_Xray.append(df_Xray.iloc[i]['Y'])
    z_Xray.append(df_Xray.iloc[i]['Y']+0.9)

x_Xray_2, y_Xray_2 = [], []
for i in range(len(df_Xray_2.index)):
    x_Xray_2.append(df_Xray_2.iloc[i]['X'])
    y_Xray_2.append(df_Xray_2.iloc[i]['Y'])

x_Xray_2_shift, y_Xray_2_shift = [], []
for i in range(1,len(df_Xray_2.index)+1):
    x_Xray_2_shift.append(df_Xray_2.iloc[-i]['X'])
    y_Xray_2_shift.append(df_Xray_2.iloc[-i]['Y']+0.7)
for i in range(len(df_Xray_2.index)):
    x_Xray_2_shift.append(df_Xray_2.iloc[i]['X'])
    y_Xray_2_shift.append(df_Xray_2.iloc[i]['Y'])

x_Kopp_SN, y_Kopp_SN = [], []
for i in range(len(df_Kopp_SN.index)):
    x_Kopp_SN.append(df_Kopp_SN.iloc[i]['X'])
    y_Kopp_SN.append(df_Kopp_SN.iloc[i]['Y'])

x_Kopp_SN_2, y_Kopp_SN_2 = [], []
for i in range(len(df_Kopp_SN_2.index)):
    x_Kopp_SN_2.append(df_Kopp_SN_2.iloc[i]['X'])
    y_Kopp_SN_2.append(df_Kopp_SN_2.iloc[i]['Y'])

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


# Set up the combined figure
fig, ax = plt.subplots(figsize=(20, 16))

axes =ax


spacing=0.2
m = np.arange(-12,6+spacing, spacing)
seesaw_bound = np.log10(0.05*10**(-9)/10**(m))

### Plot Current Constraints ###

axes.plot(x_DELPHI,y_DELPHI,linewidth=1.5,linestyle='--',color='teal') # DEPLHI
axes.plot(x_MesonDecays_LNV_mu,y_MesonDecays_LNV_mu,linewidth=1.5,linestyle='-',color='mediumaquamarine') # LNV meson decays
axes.plot(x_L3_mu,y_L3_mu,linewidth=1.5,linestyle='--',color='salmon') # L3
axes.plot(x_CMS_mu,y_CMS_mu,linewidth=1.5,linestyle='--',color='red') # CMS trilepton
axes.plot(x_ATLAS_1_mu,y_ATLAS_1_mu,linewidth=1.5,linestyle='--',color='mediumblue') # ATLAS dilepton + jets
axes.plot(x_ATLAS_2_mu,y_ATLAS_2_mu,linewidth=1.5,linestyle='--',color='mediumblue') # ATLAS dilepton + jets
axes.plot(x_ATLAS_3_mu,y_ATLAS_3_mu,linewidth=1.5,linestyle='--',color='mediumblue') # ATLAS dilepton + jets
axes.plot(x_EWPD_mu,y_EWPD_mu,linewidth=1.5,linestyle='--',color='darkgoldenrod') # Electroweak precision data
axes.plot(x_Belle_mu,y_Belle_mu,linewidth=1.5,linestyle='-',color='darkgreen') # BELLE
axes.plot(x_LHCb,y_LHCb,linewidth=1.5,linestyle='-',color='y') # LHCb
axes.plot(x_CHARM_mu,y_CHARM_mu,linewidth=1.5,linestyle='-',color='darkslategray') # CHARM
axes.plot(x_BEBC,y_BEBC,linewidth=1.5,linestyle='-.',color='orange') # BEBC
axes.plot(x_SuperK_mu_decay,y_SuperK_mu_decay,linewidth=1.5,linestyle='-',color='mediumseagreen') # Super-Kamiokande
axes.plot(x_NuTeV,y_NuTeV,linewidth=1.5,linestyle='-.',color='blueviolet') # NuTeV
axes.plot(x_FMMF,y_FMMF,linewidth=1.5,linestyle='-',color='chocolate') # FMMF
axes.plot(x_NA3_mu,y_NA3_mu,linewidth=1.5,linestyle='-',color='cornsilk') # NA3
axes.plot(x_PS191_mu,y_PS191_mu,linewidth=1.5,linestyle='--',color='magenta') # PS191
axes.plot(x_E949,y_E949,linewidth=1.5,linestyle='--',color='green') # E949
axes.plot(x_NA62_mu,y_NA62_mu,linewidth=1.5,linestyle='-',color='teal') # NA62
axes.plot(x_KEK,y_KEK,linewidth=1.5,linestyle='-',color='tomato') # Kaon decay KEK
axes.plot(x_T2K_mu,y_T2K_mu,linewidth=1.5,linestyle='-',color='purple') # T2K ND
axes.plot(x_PSI,y_PSI,linewidth=1.5,linestyle='-',color='dodgerblue') # PSI
axes.plot(x_PIENU_mu,y_PIENU_mu,linewidth=1.5,linestyle='-',color='gold') # PIENU
axes.plot(x_MuSpectrum,y_MuSpectrum,linewidth=1.5,linestyle='-',color='darkturquoise') # Mu spectrum
axes.plot(x_MINOS,y_MINOS,linewidth=1.5,linestyle='-',color='forestgreen') # MINOS
# axes.plot(x_SBN,y_SBN,linewidth=1.5,linestyle='-',color='crimson') # SBN
axes.plot(x_IceCube,y_IceCube,linewidth=1.5,linestyle='--',color='darkmagenta') # IceCube
axes.plot(x_MicroBooNE,y_MicroBooNE,linewidth=1.5,linestyle='-',color='plum') # MicroBooNE
axes.plot(x_SuperK_mu,y_SuperK_mu,linewidth=1.5,linestyle='--',color='darkorange') # SuperK upper limit
axes.plot(x_CDHS,y_CDHS,linewidth=1.5,linestyle='-.',color='orchid') # CDHS
axes.plot(x_CCFR,y_CCFR,linewidth=1.5,linestyle='-.',color='blue') # CCFR
axes.plot(x_MiniBooNE,y_MiniBooNE,linewidth=1.5,linestyle='-',color='indigo') # MiniBooNE
axes.plot(x_NOvA,y_NOvA,linewidth=1.5,linestyle='-',color='black') # Nova
axes.plot(x_IC_DC,y_IC_DC,linewidth=1.5,linestyle='--',color='darkcyan') # IceCube + DeepCore
axes.plot(x_Higgs_mu,y_Higgs_mu,linewidth=1.5,linestyle='--',color='sandybrown') # Higgs Constraints

axes.plot(x_CMB,y_CMB,linewidth=1,linestyle='-.',color='dimgrey') # Evans data
# axes.plot(x_CMB_Linear,y_CMB_Linear,linewidth=1,linestyle='-.',color='dimgrey') # Linear CMB
axes.plot(x_CMB_BAO_H,y_CMB_BAO_H,linewidth=0.5,linestyle='-',color='grey') # # Decay after BBN constraints
# axes.plot(x_CMB_H_only,y_CMB_H_only,linewidth=1.5,linestyle='--',color='red') # Decay after BBN, Hubble only
axes.plot(x_BBN,y_BBN,linewidth=0.5,linestyle='-',color='grey') # Decay before BBN constraints
# axes.plot(x_Xray,y_Xray,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray observations OLD
axes.plot(x_Xray_2,y_Xray_2,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray data
# axes.plot([2.986328125,2.986328125],[1,-13],linewidth=0.5,linestyle='--',color='black') # Tremaine-Gunn / Lyman-alpha
# axes.plot(x_Kopp_SN,y_Kopp_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Kopp Supernova constraints
axes.plot(x_Kopp_SN_2,y_Kopp_SN_2,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.4) # Kopp Supernova constraints 2

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
ax.plot(log_x_solar_max, log_ymax_solar, label="Solar E = 10 GeV", linestyle='-', color='purple', linewidth=2)
ax.plot(log_x_solar_min, log_ymin_solar, label="Solar E = 1 MeV", linestyle='--', color='purple', linewidth=2)

# Plot Astro
ax.plot(log_x_astro_max, log_ymax_astro, label="Astro E = 10 GeV", linestyle='-', color='brown', linewidth=2)
ax.plot(log_x_astro_min, log_ymin_astro, label="Astro E = 1 MeV", linestyle='--', color='brown', linewidth=2)


### Shading ###

plt.fill_between(x_DELPHI,0.1,y_DELPHI, facecolor='teal', alpha=0.075)
plt.fill_between(x_Belle_mu,0.1,y_Belle_mu, facecolor='darkgreen', alpha=0.075)
plt.fill_between(x_ATLAS_3_mu,-2.84,y_ATLAS_3_mu, facecolor='mediumblue', linewidth=0, alpha=0.075)
plt.fill_between(x_EWPD_mu,0.1,y_EWPD_mu, facecolor='darkgoldenrod', alpha=0.075)
# plt.fill_between(x_LHCb,y_LHCb, facecolor='yellow', alpha=0.075)
plt.fill_between(x_CHARM_mu,0.1,y_CHARM_mu, facecolor='darkslategray', alpha=0.075)
plt.fill_between(x_BEBC,0.1,y_BEBC, facecolor='orange', alpha=0.075)
plt.fill_between(x_NuTeV,0.1,y_NuTeV, facecolor='blueviolet', alpha=0.075)
plt.fill_between(x_FMMF,0.1,y_FMMF, facecolor='chocolate', alpha=0.075)
plt.fill_between(x_NA62_mu,0.1,y_NA62_mu, facecolor='teal', alpha=0.075)
plt.fill_between(x_KEK,0.1,y_KEK, facecolor='tomato', alpha=0.075)
plt.fill_between(x_T2K_mu,0.1,y_T2K_mu, facecolor='purple', alpha=0.075)
plt.fill_between(x_PIENU_mu,0.1,y_PIENU_mu, facecolor='gold', alpha=0.075)
plt.fill_between(x_PSI,0.1,y_PSI, facecolor='dodgerblue', alpha=0.075)
# plt.fill_between(x_SBN,0.1,y_SBN, facecolor='crimson', alpha=0.075,lw=0)
# plt.fill_between(x_SuperK_mu,0.1,y_SuperK_mu, facecolor='darkorange', alpha=0.075,lw=0)
plt.fill_between(x_MINOS,0.1,y_MINOS, facecolor='forestgreen', alpha=0.075,lw=0)
plt.fill_between(x_IceCube,0.1,y_IceCube, facecolor='darkmagenta', alpha=0.075,lw=0)
plt.fill_between(x_CCFR,0.1,y_CCFR,color='blue', alpha=0.075,lw=0)

plt.fill_between(x_CMB,y_CMB,z_CMB, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_CMB_BAO_H_2,0.1,y_CMB_BAO_H_2, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_BBN_2,0.1,y_BBN_2, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_CMB_Linear,y_CMB_Linear,z_CMB_Linear,facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_Xray_2_shift,-20,y_Xray_2_shift,color='orangered', alpha=0.075,lw=0)
# plt.fill_between(x_Kopp_SN,y_Kopp_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
plt.fill_between(x_Kopp_SN_2,-6.6,y_Kopp_SN_2,facecolor='darkslateblue', alpha=0.01,lw=0)

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

plt.text(9.91-9, -4.25, r'$\mathrm{DELPHI}$',fontsize=12,rotation=0,color='teal')
plt.text(11.8-9, -1.5, r'$\mathrm{CMS}$',fontsize=12,rotation=0,color='red')
plt.text(10.1-9, -5.5, r'$\mathrm{ATLAS}$',fontsize=12,rotation=0,color='mediumblue')
plt.text(10-9, -3.68, r'$\mathrm{L3}$',fontsize=12,rotation=0,color='salmon')
plt.text(12.1-9, -3.7, r'$\mathrm{EWPD}$',fontsize=12,rotation=0,color='darkgoldenrod')
plt.text(9.75-9, -2.9, r'$\mathrm{Belle}$',fontsize=12,rotation=0,color='darkgreen')
plt.text(9.75-9, -1.7, r'$\mathrm{LHCb}$',fontsize=12,rotation=0,color='y')
plt.text(9.15-9, -1.2, r'$\mathrm{CHARM}$',fontsize=12,rotation=0,color='darkslategray')
plt.text(9.35-9, -6.7, r'$\mathrm{BEBC}$',fontsize=12,rotation=0,color='orange')
plt.text(9.2-9, -7.35, r'$\mathrm{NuTeV}$',fontsize=12,rotation=0,color='blueviolet')
plt.text(8.65-9, -7.8, r'$\mathrm{MicroBooNE}$',fontsize=12,rotation=0,color='plum')
plt.text(8.95-9, -5.9, r'$\mathrm{FMMF}$',fontsize=12,rotation=0,color='chocolate')
plt.text(9.05-9, -0.7, r'$\mathrm{NA3}$',fontsize=12,rotation=0,color='cornsilk')
plt.text(7.42-9, -6.1, r'$\mathrm{PS191}$',fontsize=12,rotation=0,color='magenta')
plt.text(7.075-9, -6.65, r'$\mathrm{Super-K}$',fontsize=12,rotation=0,color='mediumseagreen')
plt.text(5.9-9, -4, r'$\mathrm{PSI}$',fontsize=12,rotation=0,color='dodgerblue')
plt.text(5.7-9, -1, r'$\mathrm{\mu \,spectrum}$',fontsize=12,rotation=0,color='darkturquoise')
plt.text(10.4-9, -11.5, r'$\mathrm{Seesaw}$',fontsize=12,rotation=0,color='black')
plt.text(6.4-9, -5.9, r'$\mathrm{PIENU}$',fontsize=12,rotation=0,color='gold')
plt.text(8.65-9, -9.3, r'$\mathrm{T2K} $',fontsize=12,rotation=0,color='purple')
plt.text(8.65-9, -8.4, r'$\mathrm{NA62}$',fontsize=12,rotation=0,color='teal')
plt.text(7.7-9, -8.5, r'$\mathrm{E949}$',fontsize=12,rotation=0,color='green')
plt.text(7.6-9, -3.6, r'$\mathrm{KEK}$',fontsize=12,rotation=0,color='tomato')
# plt.text(0.3-9, -3.2, r'$\mathrm{SBN}$',fontsize=16,rotation=0,color='crimson')
plt.text(1.6-9, -1.75, r'$\mathrm{MINOS(+)}$',fontsize=12,rotation=0,color='forestgreen')
plt.text(-0.07-9, -0.5, r'$\mathrm{IceCube}$',fontsize=12,rotation=0,color='darkmagenta')
plt.text(9.4-9, -11.7, r'$\mathrm{BBN}$',fontsize=12,rotation=0,color='grey')
plt.text(1.85-9, -9.2, r'$\mathrm{Supernovae}$',fontsize=12,alpha=0.7,rotation=0,color='darkslateblue')
plt.text(2.5-9, -10, r'$\mathrm{X-ray}$',fontsize=12,rotation=0,color='orangered')
plt.text(5.5-9, -11.5, r'$\mathrm{CMB}+\mathrm{BAO}+H_0$',fontsize=12,rotation=0,color='grey')
plt.text(0.2-9, -5.4, r'$\mathrm{CMB}$',fontsize=12,rotation=0,color='dimgrey')
plt.text(8.15-9, -3.5, r'$\mathrm{LNV\,Decays } $',fontsize=12,rotation=90,color='mediumaquamarine')
plt.text(1.05-9, -1.525, r'$\mathrm{SK}$',fontsize=12,rotation=0,color='darkorange')
plt.text(1.099-9, -0.75, r'$\mathrm{CDHS}$',fontsize=10,rotation=0,color='orchid')
plt.text(1.2-9, -3.0, r'$\mathrm{CCFR}$',fontsize=12,rotation=0,color='blue')
plt.text(1-9, -0.3, r'$\mathrm{MiniBooNE}$',fontsize=11,rotation=0,color='indigo')
plt.text(1.07-9, -1.08, r'$\mathrm{IC+DC,\,NO\nu A}$',fontsize=10,rotation=0,color='darkcyan')
plt.text(11.1-9, -3.7, r'$\mathrm{Higgs}$',fontsize=12,rotation=0,color='sandybrown')

axes.plot(m,seesaw_bound,linewidth=1,linestyle='dotted',color='black')

axes.set_xticks([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
axes.xaxis.set_ticklabels([r'',r'$10^{-9}$',r'',r'',r'$10^{-6}$',r'',r'',r'$10^{-3}$',r'',r'',r'$1$',r'',r'',r'$10^{3}$',r'',r'',r'$10^{6}$'],fontsize =15)
ax.set_yticks([i for i in range(-30, 7)])
ax.yaxis.set_ticklabels([
    r'$10^{-30}$', r'', r'', r'$10^{-27}$', r'', r'', r'$10^{-24}$', r'', r'', r'$10^{-21}$',
    r'', r'', r'$10^{-18}$', r'', r'', r'$10^{-15}$', r'', r'', r'$10^{-12}$', r'', r'', r'$10^{-9}$',
    r'', r'', r'$10^{-6}$', r'', r'', r'$10^{-3}$', r'', r'', r'$1$', r'', r'', r'$10^{3}$',
    r'', r'', r'$10^{6}$'
], fontsize=12)
axes.tick_params(axis='x', which='major', pad=7.5)

axes.set_ylabel(r'$|U_{\mu N}|^2$',fontsize=20,rotation=90)
axes.set_xlabel(r'$m_N \, [\mathrm{GeV}]$',fontsize=20,rotation=0)

axes.xaxis.set_label_coords(0.52,-0.08)
axes.yaxis.set_label_coords(-0.09,0.5)
axes.set_xlim(-9.1,4.1)
axes.set_ylim(-25.1,0.1)

### Set aspect ratio (golden ratio) ###

x0,x1 = axes.get_xlim()
y0,y1 = axes.get_ylim()
axes.set_aspect(2*(x1-x0)/(1+Sqrt(5))/(y1-y0))

fig.set_size_inches(16,16)

legend = plt.legend(
    loc='lower right',
    fontsize=10,
    title=r"$\mathbf{PBP\ Bounds}$",  # Bold and math-styled
    title_fontsize=14,
    frameon=True,  # Add a frame around the legend
    edgecolor='black',  # Frame color
    facecolor='whitesmoke',  # Background color
    shadow=False  # Add a shadow for effect
)

# Set the alpha for transparency
legend.get_frame().set_alpha(0.6)

# Define the path and file name
save_path = "/home/as/Documents/Wolfram_Mathematica/Codes/Geometric Phase/data/Umu4_current_sterile.png"

# Save the plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')

plt.show()
# plt.savefig("../../../plots/UmuNsq_constraints.pdf",bbox_inches='tight')
# plt.savefig("../../../plots/UmuNsq_constraints.png",bbox_inches='tight')