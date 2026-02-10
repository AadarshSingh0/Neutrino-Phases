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
base_path = "/home/as/Downloads/neutrino data/python-codes/data/data"

df_Daya_Bay = pd.read_csv(f"{base_path}/Daya_Bay_data.csv", header=None, sep=",", names=["X", "Y"])
df_DANSS = pd.read_csv(f"{base_path}/DANSS_data.csv", header=None, sep=",", names=["X", "Y"])
df_KamLAND = pd.read_csv(f"{base_path}/KamLAND_data.csv", header=None, sep=",", names=["X", "Y"])
#df_Solar_KamLAND = pd.read_csv(f"{base_path}/solar_KamLAND_data.csv", header=None, sep=",", names=["X", "Y"])
df_NEOS = pd.read_csv(f"{base_path}/NEOS_data.csv", header=None, sep=",", names=["X", "Y"])
df_PROSPECT = pd.read_csv(f"{base_path}/PROSPECT_data.csv", header=None, sep=",", names=["X", "Y"])
df_SKICDC = pd.read_csv(f"{base_path}/SKICDC_data.csv", header=None, sep=",", names=["X", "Y"])
df_Troitsk_2013 = pd.read_csv(f"{base_path}/Troitsk_2013_data.csv", header=None, sep=",", names=["X", "Y"])
df_Troitsk_2017 = pd.read_csv(f"{base_path}/Troitsk_2017_data.csv", header=None, sep=",", names=["X", "Y"])
df_Hiddemann = pd.read_csv(f"{base_path}/Hiddemann_data.csv", header=None, sep=",", names=["X", "Y"])
df_Mainz = pd.read_csv(f"{base_path}/Mainz_data.csv", header=None, sep=",", names=["X", "Y"])
df_Re187 = pd.read_csv(f"{base_path}/Re187_data.csv", header=None, sep=",", names=["X", "Y"])
df_Ni63 = pd.read_csv(f"{base_path}/Ni63_data.csv", header=None, sep=",", names=["X", "Y"])
df_S35 = pd.read_csv(f"{base_path}/S35_data.csv", header=None, sep=",", names=["X", "Y"])
df_Cu64 = pd.read_csv(f"{base_path}/Cu64_data.csv", header=None, sep=",", names=["X", "Y"])
df_Ca45 = pd.read_csv(f"{base_path}/Ca45_data.csv", header=None, sep=",", names=["X", "Y"])
df_CePr144 = pd.read_csv(f"{base_path}/CePr144_data.csv", header=None, sep=",", names=["X", "Y"])
df_F20 = pd.read_csv(f"{base_path}/F20_data.csv", header=None, sep=",", names=["X", "Y"])
df_BeEST = pd.read_csv(f"{base_path}/BeEST_data.csv", header=None, sep=",", names=["X", "Y"])
df_Rovno = pd.read_csv(f"{base_path}/Rovno_data.csv", header=None, sep=",", names=["X", "Y"])
df_Bugey = pd.read_csv(f"{base_path}/Bugey_data.csv", header=None, sep=",", names=["X", "Y"])
df_Borexino = pd.read_csv(f"{base_path}/Borexino_data.csv", header=None, sep=",", names=["X", "Y"])
df_PIENU = pd.read_csv(f"{base_path}/PIENU_data.csv", header=None, sep=",", names=["X", "Y"])
df_PionDecay = pd.read_csv(f"{base_path}/PionDecay_data.csv", header=None, sep=",", names=["X", "Y"])
df_PS191 = pd.read_csv(f"{base_path}/PS191_data.csv", header=None, sep=",", names=["X", "Y"])
df_PS191_Re = pd.read_csv(f"{base_path}/PS191_Re_data.csv", header=None, sep=",", names=["X", "Y"])
df_JINR = pd.read_csv(f"{base_path}/JINR_data.csv", header=None, sep=",", names=["X", "Y"])
df_NA62 = pd.read_csv(f"{base_path}/NA62_data.csv", header=None, sep=",", names=["X", "Y"])
df_T2K = pd.read_csv(f"{base_path}/T2K_data.csv", header=None, sep=",", names=["X", "Y"])
df_SuperK = pd.read_csv(f"{base_path}/SuperK_data.csv", header=None, sep=",", names=["X", "Y"])
df_MesonDecays_LNV = pd.read_csv(f"{base_path}/MesonDecays_LNV_data.csv", header=None, sep=",", names=["X", "Y"])
df_CHARM = pd.read_csv(f"{base_path}/CHARM_data.csv", header=None, sep=",", names=["X", "Y"])
df_BEBC = pd.read_csv(f"{base_path}/BEBC_data.csv", header=None, sep=",", names=["X", "Y"])
df_BESIII_1 = pd.read_csv(f"{base_path}/BESIII_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_BESIII_2 = pd.read_csv(f"{base_path}/BESIII_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_NA3 = pd.read_csv(f"{base_path}/NA3_data.csv", header=None, sep=",", names=["X", "Y"])
# df_Belle = pd.read_csv(f"{base_path}/BELLE_data.csv", header=None, sep=",", names=["X", "Y"])
df_DELPHI = pd.read_csv(f"{base_path}/DELPHI_data.csv", header=None, sep=",", names=["X", "Y"])
df_L3_1 = pd.read_csv(f"{base_path}/L3_1_data.csv", header=None, sep=",", names=["X", "Y"])
df_L3_2 = pd.read_csv(f"{base_path}/L3_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS = pd.read_csv(f"{base_path}/ATLAS_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_LNV = pd.read_csv(f"{base_path}/ATLAS_LNV_data.csv", header=None, sep=",", names=["X", "Y"])
df_ATLAS_LNC = pd.read_csv(f"{base_path}/ATLAS_LNC_data.csv", header=None, sep=",", names=["X", "Y"])
df_Higgs = pd.read_csv(f"{base_path}/Higgs_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMS_SameSign = pd.read_csv(f"{base_path}/CMS_SameSign_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMS_TriLepton = pd.read_csv(f"{base_path}/CMS_TriLepton_data.csv", header=None, sep=",", names=["X", "Y"])
df_EWPD = pd.read_csv(f"{base_path}/EWPD_data.csv", header=None, sep=",", names=["X", "Y"])
df_Planck = pd.read_csv(f"{base_path}/Planck_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB = pd.read_csv(f"{base_path}/CMB_data.csv", header=None, sep=",", names=["X", "Y"])
# df_Xray = pd.read_csv(f"{base_path}/xray_data.csv", header=None, sep=",", names=["X", "Y"])
df_Xray_2 = pd.read_csv(f"{base_path}/Xray_2_data.csv", header=None, sep=",", names=["X", "Y"])
# df_CMB_Linear = pd.read_csv(f"{base_path}/CMB_linear_data.csv", header=None, sep=",", names=["X", "Y"])
df_Zhou_SN = pd.read_csv(f"{base_path}/Zhou_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_Zhou_SN_2 = pd.read_csv(f"{base_path}/Zhou_SN_2_data.csv", header=None, sep=",", names=["X", "Y"])
df_Raffelt_SN = pd.read_csv(f"{base_path}/Raffelt_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_ShiSigl_SN = pd.read_csv(f"{base_path}/ShiSigl_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_Valle_SN = pd.read_csv(f"{base_path}/Valle_SN_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_BAO_H = pd.read_csv(f"{base_path}/CMB_BAO_H_data.csv", header=None, sep=",", names=["X", "Y"])
df_CMB_H_only = pd.read_csv(f"{base_path}/CMB_H_only_data.csv", header=None, sep=",", names=["X", "Y"])
df_BBN = pd.read_csv(f"{base_path}/BBN_data.csv", header=None, sep=",", names=["X", "Y"])


# Read to (x, y) = (m_N, |V_{eN}|^2) 
x_Daya_Bay, y_Daya_Bay = [], []
for i in range(len(df_Daya_Bay.index)):
    x_Daya_Bay.append(df_Daya_Bay.iloc[i]['X'])
    y_Daya_Bay.append(df_Daya_Bay.iloc[i]['Y'])

x_DANSS, y_DANSS = [], []
for i in range(len(df_DANSS.index)):
    x_DANSS.append(df_DANSS.iloc[i]['X'])
    y_DANSS.append(df_DANSS.iloc[i]['Y'])

x_KamLAND, y_KamLAND = [], []
for i in range(len(df_KamLAND.index)):
    x_KamLAND.append(df_KamLAND.iloc[i]['X'])
    y_KamLAND.append(df_KamLAND.iloc[i]['Y'])

# x_Solar_KamLAND, y_Solar_KamLAND = [], []
# for i in range(len(df_Solar_KamLAND.index)):
#     x_Solar_KamLAND.append(df_Solar_KamLAND.iloc[i]['X'])
#     y_Solar_KamLAND.append(df_Solar_KamLAND.iloc[i]['Y'])

x_NEOS, y_NEOS = [], []
for i in range(len(df_NEOS.index)):
    x_NEOS.append(df_NEOS.iloc[i]['X'])
    y_NEOS.append(df_NEOS.iloc[i]['Y'])

x_PROSPECT, y_PROSPECT = [], []
for i in range(len(df_PROSPECT.index)):
    x_PROSPECT.append(df_PROSPECT.iloc[i]['X'])
    y_PROSPECT.append(df_PROSPECT.iloc[i]['Y'])

x_SKICDC, y_SKICDC = [], []
for i in range(len(df_SKICDC.index)):
    x_SKICDC.append(df_SKICDC.iloc[i]['X'])
    y_SKICDC.append(df_SKICDC.iloc[i]['Y'])

x_Troitsk_2013, y_Troitsk_2013 = [], []
for i in range(len(df_Troitsk_2013.index)):
    x_Troitsk_2013.append(df_Troitsk_2013.iloc[i]['X'])
    y_Troitsk_2013.append(df_Troitsk_2013.iloc[i]['Y'])

x_Troitsk_2017, y_Troitsk_2017 = [], []
for i in range(len(df_Troitsk_2017.index)):
    x_Troitsk_2017.append(df_Troitsk_2017.iloc[i]['X'])
    y_Troitsk_2017.append(df_Troitsk_2017.iloc[i]['Y'])

x_Tritium, y_Tritium = [], []
for i in range(len(df_Troitsk_2013.index)):
    x_Tritium.append(df_Troitsk_2013.iloc[i]['X'])
    y_Tritium.append(df_Troitsk_2013.iloc[i]['Y'])
for i in range(len(df_Troitsk_2017.index)):
    x_Tritium.append(df_Troitsk_2017.iloc[i]['X'])
    y_Tritium.append(df_Troitsk_2017.iloc[i]['Y'])

x_Hiddemann, y_Hiddemann = [], []
for i in range(len(df_Hiddemann.index)):
    x_Hiddemann.append(df_Hiddemann.iloc[i]['X'])
    y_Hiddemann.append(df_Hiddemann.iloc[i]['Y'])

x_Mainz, y_Mainz = [], []
for i in range(len(df_Mainz.index)):
    x_Mainz.append(df_Mainz.iloc[i]['X'])
    y_Mainz.append(df_Mainz.iloc[i]['Y'])

x_Re187, y_Re187 = [], []
for i in range(len(df_Re187.index)):
    x_Re187.append(df_Re187.iloc[i]['X'])
    y_Re187.append(df_Re187.iloc[i]['Y'])

x_Ni63, y_Ni63 = [], []
for i in range(len(df_Ni63.index)):
    x_Ni63.append(df_Ni63.iloc[i]['X'])
    y_Ni63.append(df_Ni63.iloc[i]['Y'])

x_S35, y_S35 = [], []
for i in range(len(df_S35.index)):
    x_S35.append(df_S35.iloc[i]['X'])
    y_S35.append(df_S35.iloc[i]['Y'])

x_Cu64, y_Cu64 = [], []
for i in range(len(df_Cu64.index)):
    x_Cu64.append(df_Cu64.iloc[i]['X'])
    y_Cu64.append(df_Cu64.iloc[i]['Y'])

x_Ca45, y_Ca45 = [], []
for i in range(len(df_Ca45.index)):
    x_Ca45.append(df_Ca45.iloc[i]['X'])
    y_Ca45.append(df_Ca45.iloc[i]['Y'])

x_CePr144, y_CePr144 = [], []
for i in range(len(df_CePr144.index)):
    x_CePr144.append(df_CePr144.iloc[i]['X'])
    y_CePr144.append(df_CePr144.iloc[i]['Y'])

x_F20, y_F20 = [], []
for i in range(len(df_F20.index)):
    x_F20.append(df_F20.iloc[i]['X'])
    y_F20.append(df_F20.iloc[i]['Y'])

x_BeEST, y_BeEST    = [], []
for i in range(len(df_BeEST.index)):
    x_BeEST.append(df_BeEST.iloc[i]['X'])
    y_BeEST.append(df_BeEST.iloc[i]['Y'])

x_Rovno, y_Rovno = [], []
for i in range(len(df_Rovno.index)):
    x_Rovno.append(df_Rovno.iloc[i]['X'])
    y_Rovno.append(df_Rovno.iloc[i]['Y'])

x_Bugey, y_Bugey = [], []
for i in range(len(df_Bugey.index)):
    x_Bugey.append(df_Bugey.iloc[i]['X'])
    y_Bugey.append(df_Bugey.iloc[i]['Y'])

x_Borexino, y_Borexino = [], []
for i in range(len(df_Borexino.index)):
    x_Borexino.append(df_Borexino.iloc[i]['X'])
    y_Borexino.append(df_Borexino.iloc[i]['Y'])

x_PionDecay, y_PionDecay = [], []
for i in range(len(df_PionDecay.index)):
    x_PionDecay.append(df_PionDecay.iloc[i]['X'])
    y_PionDecay.append(df_PionDecay.iloc[i]['Y'])

x_PIENU, y_PIENU = [], []
for i in range(len(df_PIENU.index)):
    x_PIENU.append(df_PIENU.iloc[i]['X'])
    y_PIENU.append(df_PIENU.iloc[i]['Y'])

x_PionDecay_tot, y_PionDecay_tot = [], []
for i in range(len(df_PionDecay.index)):
    x_PionDecay_tot.append(df_PionDecay.iloc[i]['X'])
    y_PionDecay_tot.append(df_PionDecay.iloc[i]['Y'])
for i in range(len(df_PIENU.index)):
    x_PionDecay_tot.append(df_PIENU.iloc[i]['X'])
    y_PionDecay_tot.append(df_PIENU.iloc[i]['Y'])

x_PS191, y_PS191 = [], []
for i in range(len(df_PS191.index)):
    x_PS191.append(df_PS191.iloc[i]['X'])
    y_PS191.append(df_PS191.iloc[i]['Y'])

x_PS191_Re, y_PS191_Re = [], []
for i in range(len(df_PS191_Re.index)):
    x_PS191_Re.append(df_PS191_Re.iloc[i]['X'])
    y_PS191_Re.append(df_PS191_Re.iloc[i]['Y'])

x_JINR, y_JINR = [], []
for i in range(len(df_JINR.index)):
    x_JINR.append(df_JINR.iloc[i]['X'])
    y_JINR.append(df_JINR.iloc[i]['Y'])

x_NA62, y_NA62 = [], []
for i in range(len(df_NA62.index)):
    x_NA62.append(df_NA62.iloc[i]['X'])
    y_NA62.append(df_NA62.iloc[i]['Y'])

x_T2K, y_T2K = [], []
for i in range(len(df_T2K.index)):
    x_T2K.append(df_T2K.iloc[i]['X'])
    y_T2K.append(df_T2K.iloc[i]['Y'])

x_SuperK, y_SuperK = [], []
for i in range(len(df_SuperK.index)):
    x_SuperK.append(df_SuperK.iloc[i]['X'])
    y_SuperK.append(df_SuperK.iloc[i]['Y'])

x_MesonDecays_LNV, y_MesonDecays_LNV = [], []
for i in range(len(df_MesonDecays_LNV.index)):
    x_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['X'])
    y_MesonDecays_LNV.append(df_MesonDecays_LNV.iloc[i]['Y'])

x_CHARM, y_CHARM = [], []
for i in range(len(df_CHARM.index)):
    x_CHARM.append(df_CHARM.iloc[i]['X'])
    y_CHARM.append(df_CHARM.iloc[i]['Y'])

x_BEBC, y_BEBC = [], []
for i in range(len(df_BEBC.index)):
    x_BEBC.append(df_BEBC.iloc[i]['X'])
    y_BEBC.append(df_BEBC.iloc[i]['Y'])

x_BESIII_1, y_BESIII_1 = [], []
for i in range(len(df_BESIII_1.index)):
    x_BESIII_1.append(df_BESIII_1.iloc[i]['X'])
    y_BESIII_1.append(df_BESIII_1.iloc[i]['Y'])

x_BESIII_2, y_BESIII_2 = [], []
for i in range(len(df_BESIII_2.index)):
    x_BESIII_2.append(df_BESIII_2.iloc[i]['X'])
    y_BESIII_2.append(df_BESIII_2.iloc[i]['Y'])

x_NA3, y_NA3 = [], []
for i in range(len(df_NA3.index)):
    x_NA3.append(df_NA3.iloc[i]['X'])
    y_NA3.append(df_NA3.iloc[i]['Y'])

# x_Belle, y_Belle = [], []
# for i in range(len(df_Belle.index)):
#     x_Belle.append(df_Belle.iloc[i]['X'])
#     y_Belle.append(df_Belle.iloc[i]['Y'])

x_DELPHI, y_DELPHI = [], []
for i in range(len(df_DELPHI.index)):
    x_DELPHI.append(df_DELPHI.iloc[i]['X'])
    y_DELPHI.append(df_DELPHI.iloc[i]['Y'])

x_L3_1, y_L3_1 = [], []
for i in range(len(df_L3_1.index)):
    x_L3_1.append(df_L3_1.iloc[i]['X'])
    y_L3_1.append(df_L3_1.iloc[i]['Y'])

x_L3_2, y_L3_2 = [], []
for i in range(len(df_L3_2.index)):
    x_L3_2.append(df_L3_2.iloc[i]['X'])
    y_L3_2.append(df_L3_2.iloc[i]['Y'])

x_ATLAS, y_ATLAS = [], []
for i in range(len(df_ATLAS.index)):
    x_ATLAS.append(df_ATLAS.iloc[i]['X'])
    y_ATLAS.append(df_ATLAS.iloc[i]['Y'])

x_ATLAS_LNV, y_ATLAS_LNV = [], []
for i in range(len(df_ATLAS_LNV.index)):
    x_ATLAS_LNV.append(df_ATLAS_LNV.iloc[i]['X'])
    y_ATLAS_LNV.append(df_ATLAS_LNV.iloc[i]['Y'])

x_ATLAS_LNC, y_ATLAS_LNC = [], []
for i in range(len(df_ATLAS_LNC.index)):
    x_ATLAS_LNC.append(df_ATLAS_LNC.iloc[i]['X'])
    y_ATLAS_LNC.append(df_ATLAS_LNC.iloc[i]['Y'])

x_Higgs, y_Higgs = [], []
for i in range(len(df_Higgs.index)):
    x_Higgs.append(df_Higgs.iloc[i]['X'])
    y_Higgs.append(df_Higgs.iloc[i]['Y'])

x_CMS_SameSign, y_CMS_SameSign = [], []
for i in range(len(df_CMS_SameSign.index)):
    x_CMS_SameSign.append(df_CMS_SameSign.iloc[i]['X'])
    y_CMS_SameSign.append(df_CMS_SameSign.iloc[i]['Y'])

x_CMS_TriLepton, y_CMS_TriLepton = [], []
for i in range(len(df_CMS_TriLepton.index)):
    x_CMS_TriLepton.append(df_CMS_TriLepton.iloc[i]['X'])
    y_CMS_TriLepton.append(df_CMS_TriLepton.iloc[i]['Y'])

x_EWPD, y_EWPD = [], []
for i in range(len(df_EWPD.index)):
    x_EWPD.append(df_EWPD.iloc[i]['X'])
    y_EWPD.append(df_EWPD.iloc[i]['Y'])

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
    x_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['X']+0.5)
    y_CMB_BAO_H_2.append(df_CMB_BAO_H.iloc[-i]['Y']+0.5)

x_CMB_H_only, y_CMB_H_only  = [], []
for i in range(len(df_CMB_H_only.index)):
    x_CMB_H_only.append(df_CMB_H_only.iloc[i]['X'])
    y_CMB_H_only.append(df_CMB_H_only.iloc[i]['Y'])

x_BBN, y_BBN  = [], []
for i in range(len(df_BBN.index)):
    x_BBN.append(df_BBN.iloc[i]['X'])
    y_BBN.append(df_BBN.iloc[i]['Y'])
x_BBN.append(-2.16)
y_BBN.append(0.1)

x_BBN_2, y_BBN_2  = [], []
x_BBN_2.append(-2.16 - 0.5)
y_BBN_2.append(0.1)
for i in range(1,len(df_BBN.index)+1):
    x_BBN_2.append(df_BBN.iloc[-i]['X'] - 0.5)
    y_BBN_2.append(df_BBN.iloc[-i]['Y'] - 0.5)
for i in range(len(df_BBN.index)):
    x_BBN_2.append(df_BBN.iloc[i]['X'])
    y_BBN_2.append(df_BBN.iloc[i]['Y'])
x_BBN_2.append(-2.16)
y_BBN_2.append(0.1)

# x_Xray, y_Xray, z_Xray = [], [], []
# for i in range(len(df_Xray.index)):
#     x_Xray.append(df_Xray.iloc[i]['X'])
#     y_Xray.append(df_Xray.iloc[i]['Y'])
#     z_Xray.append(df_Xray.iloc[i]['Y']+0.9)

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

x_Zhou_SN, y_Zhou_SN = [], []
for i in range(len(df_Zhou_SN.index)):
    x_Zhou_SN.append(df_Zhou_SN.iloc[i]['X'])
    y_Zhou_SN.append(df_Zhou_SN.iloc[i]['Y'])

x_Zhou_SN_shift, y_Zhou_SN_shift, z_Zhou_SN_shift = [], [], []
for i in range(43+1):
    x_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['Y']-9)
    y_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['X'])
    z_Zhou_SN_shift.append(df_Zhou_SN.iloc[i]['Y']+0.25)

x_Zhou_SN_shift2, y_Zhou_SN_shift2, z_Zhou_SN_shift2 = [], [], []
for i in range(len(df_Zhou_SN.index)-30,len(df_Zhou_SN.index)):
    x_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['X'])
    y_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['Y'])
    z_Zhou_SN_shift2.append(df_Zhou_SN.iloc[i]['Y']+0.5)

x_Zhou_SN_2, y_Zhou_SN_2, z_Zhou_SN_2 = [], [], []
for i in range(len(df_Zhou_SN_2.index)-3):
    x_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['X'])
    y_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['Y'])
    z_Zhou_SN_2.append(df_Zhou_SN_2.iloc[i]['Y']-0.3)

x_Raffelt_SN, y_Raffelt_SN = [], []
for i in range(len(df_Raffelt_SN.index)):
    x_Raffelt_SN.append(df_Raffelt_SN.iloc[i]['X'])
    y_Raffelt_SN.append(df_Raffelt_SN.iloc[i]['Y'])

x_ShiSigl_SN, y_ShiSigl_SN = [], []
for i in range(len(df_ShiSigl_SN.index)):
    x_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['X'])
    y_ShiSigl_SN.append(df_ShiSigl_SN.iloc[i]['Y'])
x_ShiSigl_SN.append(x_ShiSigl_SN[0])
y_ShiSigl_SN.append(y_ShiSigl_SN[0])

x_Valle_SN, y_Valle_SN = [], []
for i in range(len(df_Valle_SN.index)):
    x_Valle_SN.append(df_Valle_SN.iloc[i]['X'])
    y_Valle_SN.append(df_Valle_SN.iloc[i]['Y'])

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
fig, ax = plt.subplots(figsize=(18, 10))
spacing=0.2
m = np.arange(-12,6+spacing, spacing)
age_bound = np.log10(1.1 * 10**(-7) * 10**(-45) * ((50 * 10**(3))/10**(m))**5)
bbn_bound = np.log10(1.00*10**(34) * 10**(-45) * (1/10**(m))**5)
seesaw_bound = np.log10(0.05*10**(-9)/10**(m))

axes = ax

# Plot the current constraints
axes.plot(x_Daya_Bay,y_Daya_Bay,linewidth=1.5,linestyle='-',color='mediumorchid') # Daya bay
# axes.plot(x_DANSS,y_DANSS,linewidth=1.5,linestyle='-',color='darkred') # DANSS
# axes.plot(x_KamLAND,y_KamLAND,linewidth=1.5,linestyle='-',color='orangered') # Solar and KamLAND
# axes.plot(x_Solar_KamLAND,y_Solar_KamLAND,linewidth=1.5,linestyle='-',color='green') # Solar and KamLAND
axes.plot(x_NEOS,y_NEOS,linewidth=1.5,linestyle='-',color='forestgreen') # NEOS
axes.plot(x_PROSPECT,y_PROSPECT,linewidth=1.5,linestyle='-',color='darkturquoise') # PROSPECT
axes.plot(x_SKICDC,y_SKICDC,linewidth=1.5,linestyle='--',color='darkorange') # SuperK + IceCube + DeepCore

axes.plot(x_Tritium,y_Tritium,linewidth=1.5,linestyle='-',color='lime') # Troitsk 2013 Tritium (and Nu--Mass)
# axes.plot(x_Mainz,y_Mainz,linewidth=1.5,linestyle='-',color='lime') # Mainz Tritium
# axes.plot(x_Hiddemann,y_Hiddemann,linewidth=1.5,linestyle='-',color='b') # Hiddemannn Tritium
axes.plot(x_Re187,y_Re187,linewidth=1.5,linestyle='-',color='violet') # Re--187
axes.plot(x_Ni63,y_Ni63,linewidth=1.5,linestyle='-',color='darkmagenta') # Ni--63
axes.plot(x_S35,y_S35,linewidth=1.5,linestyle='-',color='y') # S--35
axes.plot(x_Ca45,y_Ca45,linewidth=1.5,linestyle='-',color='plum') # Ca--45
axes.plot(x_Cu64,y_Cu64,linewidth=1.5,linestyle='-',color='darkcyan') # Cu--64
axes.plot(x_CePr144,y_CePr144,linewidth=1.5,linestyle='--',color='crimson') # Ce Pr--144
axes.plot(x_F20,y_F20,linewidth=1.5,linestyle='--',color='green') # F--20
axes.plot(x_BeEST,y_BeEST,linewidth=1.5,linestyle='-',color='rosybrown') # Be--7

axes.plot(x_Rovno,y_Rovno,linewidth=1.5,linestyle='--',color='tomato') # Rovno reactor decay
axes.plot(x_Bugey,y_Bugey,linewidth=1.5,linestyle='--',color='turquoise') # Buguy reactor decay
axes.plot(x_Borexino,y_Borexino,linewidth=1.5,linestyle='-.',color='blue') # BOREXINO
axes.plot(x_PionDecay_tot,y_PionDecay_tot,linewidth=1.5,linestyle='-',color='gold') # Pion decay low and PIENU
axes.plot(x_PS191,y_PS191,linewidth=1.5,linestyle='--',color='magenta') # PS191
# axes.plot(x_PS191_Re,y_PS191_Re,linewidth=1.5,linestyle='-',color='purple') # PS191 Reanalysis
axes.plot(x_JINR,y_JINR,linewidth=1.5,linestyle='--',color='sienna') # JINR
axes.plot(x_NA62,y_NA62,linewidth=1.5,linestyle='-',color='teal') # NA62 Constraints New
axes.plot(x_T2K,y_T2K,linewidth=1.5,linestyle='-',color='purple') # T2K ND
axes.plot(x_SuperK,y_SuperK,linewidth=1.5,linestyle='-',color='mediumseagreen') # Super-Kamiokande
axes.plot(x_MesonDecays_LNV,y_MesonDecays_LNV,linewidth=1.5,linestyle='-',color='mediumaquamarine') # LNV meson decays
axes.plot(x_CHARM,y_CHARM,linewidth=1.5,linestyle='-.',color='darkslategray') # CHARM
axes.plot(x_BEBC,y_BEBC,linewidth=1.5,linestyle='-',color='yellowgreen') # BEBC
# axes.plot(x_BESIII_1,x_BESIII_1,linewidth=1.5,linestyle='-',color='r') # BESIII D0 decays
axes.plot(x_BESIII_2,y_BESIII_2,linewidth=1.5,linestyle='-',color='c') # BESIII D+ decays

axes.plot(x_NA3,y_NA3,linewidth=1.5,linestyle='-',color='cornsilk') # NA3
# axes.plot(x_Belle,y_Belle,linewidth=1.5,linestyle='-',color='darkgreen') # BELLE
axes.plot(x_DELPHI,y_DELPHI,linewidth=1.5,linestyle='--',color='teal') # DEPLHI
axes.plot(x_L3_1,y_L3_1,linewidth=1.5,linestyle='--',color='salmon') # L3
axes.plot(x_L3_2,y_L3_2,linewidth=1.5,linestyle='--',color='salmon') # L3
axes.plot(x_ATLAS,y_ATLAS,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS dilepton + jets
# axes.plot(x_ATLAS_LNC,y_ATLAS_LNC,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS prompt LNC
axes.plot(x_ATLAS_LNV,y_ATLAS_LNV,linewidth=1.5,linestyle='--',color='mediumblue') # ALTLAS trilepton prompt LNV
axes.plot(x_Higgs,y_Higgs,linewidth=1.5,linestyle='--',color='sandybrown') # Higgs Constraints
# axes.plot(x_CMS_SameSign,y_CMS_SameSign,linewidth=1.5,linestyle='--',color='r') # CMS Same Sign LNV
axes.plot(x_CMS_TriLepton,y_CMS_TriLepton,linewidth=1.5,linestyle='--',color='red') # CMS trilepton
axes.plot(x_EWPD,y_EWPD,linewidth=1.5,linestyle='--',color='darkgoldenrod') # Electroweak precision data

axes.plot(x_CMB,y_CMB,linewidth=1,linestyle='-.',color='dimgrey') # Evans data
# axes.plot(x_CMB_Linear,y_CMB_Linear,linewidth=1,linestyle='-.',color='dimgrey') # Linear CMB
axes.plot(x_CMB_BAO_H,y_CMB_BAO_H,linewidth=0.5,linestyle='-',color='grey') # # Decay after BBN constraints
# axes.plot(x_CMB_H_only,y_CMB_H_only,linewidth=1.5,linestyle='--',color='red') # Decay after BBN, Hubble only
axes.plot(x_BBN,y_BBN,linewidth=0.5,linestyle='-',color='grey') # Decay before BBN constraints
# axes.plot(x_Xray,y_Xray,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray observations OLD
axes.plot(x_Xray_2,y_Xray_2,linewidth=1.5,linestyle='-',color='orangered') # Combined X-ray data
# axes.plot([2.986328125-9,2.986328125-9],[1,-13],linewidth=0.5,linestyle='--',color='black') # Tremaine-Gunn / Lyman-alpha
# axes.plot(x_Kopp_SN,y_Kopp_SN,linewidth=1.5,linestyle=':',color='darkslategrey',alpha=0.15) # Kopp Supernova constraints
# axes.plot(x_Kopp_SN_2,y_Kopp_SN_2,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Kopp Supernova constraints 2
# axes.plot(x_Zhou_SN,y_Zhou_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Zhou Supernova constraints 1
# axes.plot(x_Zhou_SN_2,y_Zhou_SN_2,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Zhou Supernova constraints 2
# axes.plot(x_Raffelt_SN,y_Raffelt_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Raffelt and Zhou Supernova constraints
axes.plot(x_ShiSigl_SN,y_ShiSigl_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Shi and Sigl Supernova constraints
# axes.plot(x_Valle_SN,y_Valle_SN,linewidth=1.5,linestyle=':',color='darkslateblue',alpha=0.15) # Valle Supernova r nucleosynthesis

# axes.plot(x_Planck,y_Planck,linewidth=1.5,linestyle='--',color='m') # Planck
# axes.plot([np.log10(0.5*10**-6),np.log10(0.5*10**-6)],[-13,1],linewidth=1.5,linestyle='-.',color='black') # Tremaine-Gunn bound (assumes HNL is DM)

# axes.plot(m,age_bound,linewidth=1.5,linestyle='--',color='r') # Age of the universe decay
# axes.plot(m,bbn_bound,linewidth=1.5,linestyle='--',color='r') # BBN decay
axes.plot(m,seesaw_bound,linewidth=1,linestyle='dotted',color='black') # Seesaw line


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
ax.plot(log_x_snd_max, log_ymax_snd, label="FASER E = 1 TeV", linestyle='-', color='orange', linewidth=2)
ax.plot(log_x_snd_min, log_ymin_snd, label="FASER E = 100 GeV", linestyle='--', color='orange', linewidth=2)

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

# Shading the regions

plt.fill_between(x_Daya_Bay,0.1,y_Daya_Bay, facecolor='mediumorchid', alpha=0.075,lw=0)
plt.fill_between(x_PROSPECT,0.1,y_PROSPECT, facecolor='darkturquoise', alpha=0.075,lw=0)
plt.fill_between(x_NEOS,0.1,y_NEOS, facecolor='forestgreen', alpha=0.075,lw=0)
# plt.fill_between(x_SKICDC,y_SKICDC, facecolor='crimson', alpha=0.075,lw=0)
plt.fill_between(x_Tritium,0.1,y_Tritium, facecolor='lime', alpha=0.075)
# plt.fill_between(x_Mainz,y_Mainz, facecolor='navy', alpha=0.075)
plt.fill_between(x_Re187,0.1,y_Re187, facecolor='violet', alpha=0.075)
plt.fill_between(x_Ni63,0.1,y_Ni63, facecolor='darkmagenta', alpha=0.075)
plt.fill_between(x_S35,0.1,y_S35, facecolor='y', alpha=0.075)
# plt.fill_between(x_Ca45,0.1,y_Ca45, facecolor='plum', alpha=0.075)
# plt.fill_between(x_Cu64,0.1,y_Cu64, facecolor='mediumvioletred', alpha=0.075)
plt.fill_between(x_CePr144,0.1,y_CePr144, facecolor='crimson', alpha=0.075)
plt.fill_between(x_F20,0.1,y_F20, facecolor='green', alpha=0.075)
plt.fill_between(x_BeEST,0.1,y_BeEST, facecolor='rosybrown', alpha=0.075)
# plt.fill_between(x_Rovno,0.1,y_Rovno, facecolor='tomato', alpha=0.075)
# plt.fill_between(x_Bugey,0.1,y_Bugey, facecolor='black', alpha=0.075)
plt.fill_between(x_Borexino,0.1,y_Borexino, facecolor='blue', alpha=0.075)
plt.fill_between(x_PionDecay_tot,0.1,y_PionDecay_tot, facecolor='gold', alpha=0.075)
# plt.fill_between(x_PS191,0.1,y_PS191, facecolor='magenta', alpha=0.075)
# plt.fill_between(x23,0.1,y23, facecolor='gold', alpha=0.075)
plt.fill_between(x_T2K,0.1,y_T2K, facecolor='purple', alpha=0.075)
# plt.fill_between(x_NA62,0.1,y_NA62, facecolor='teal', alpha=0.075)
# plt.fill_between(x_JINR,0.1,y_JINR, facecolor='sienna', alpha=0.075)
# plt.fill_between(x_CHARM,0.1,y_CHARM, facecolor='darkslategray', alpha=0.075)
plt.fill_between(x_BEBC,0.1,y_BEBC, facecolor='yellowgreen', alpha=0.075)
plt.fill_between(x_DELPHI,0.1,y_DELPHI, facecolor='teal', alpha=0.075)
# plt.fill_between(x_Belle,0.1,y_Belle, facecolor='darkgreen', alpha=0.075)
# plt.fill_between(x_ATLAS_LNC,0.1,y_ATLAS_LNC, facecolor='mediumblue', alpha=0.075)
# plt.fill_between(x_CMS_TriLepton,0.1,y_CMS_TriLepton, facecolor='red', alpha=0.075)
plt.fill_between(x_EWPD,0.1,y_EWPD, facecolor='darkgoldenrod', alpha=0.075)

plt.fill_between(x_CMB,y_CMB,z_CMB, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_CMB_BAO_H_2,0.1,y_CMB_BAO_H_2, facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_BBN_2,0.1,y_BBN_2, facecolor='black', alpha=0.02,lw=0)
# plt.fill_between(x_CMB_Linear,y_CMB_Linear,z_CMB_Linear,facecolor='black', alpha=0.02,lw=0)
plt.fill_between(x_Xray_2_shift,-20,y_Xray_2_shift,color='orangered', alpha=0.075,lw=0)
# plt.fill_between(x_Kopp_SN,y_Kopp_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Kopp_SN_2,y_Kopp_SN_2,facecolor='darkslategrey', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN,y_Zhou_SN_2,y_Zhou_SN,facecolor='darkslategrey', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_shift,y_Zhou_SN_shift,z_Zhou_SN_2,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_shift2,y_Zhou_SN_shift2,z_Zhou_SN_3,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Zhou_SN_2,y_Zhou_SN_2,z_Zhou_SN_2_1,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Raffelt_SN,y_Raffelt_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
plt.fill_between(x_ShiSigl_SN,-3.6,y_ShiSigl_SN,facecolor='darkslateblue', alpha=0.005,lw=0)
# plt.fill_between(x_Valle_SN,0.1,y_Valle_SN,facecolor='darkslateblue', alpha=0.005,lw=0)

## Filling for PBP
x_danss = [log_x_danss_max[0], log_x_danss_min[0], log_x_danss_min[-1], log_x_danss_max[-1]]
y_danss = [log_ymax_danss[0], log_ymin_danss[0], log_ymin_danss[-1], log_ymax_danss[-1]]
plt.fill(x_danss, y_danss, color='lightblue', alpha=0.5, label='DANSS Region')

## Filling for SND
x_snd = [log_x_snd_max[0], log_x_snd_min[0], log_x_snd_min[-1], log_x_snd_max[-1]]
y_snd = [log_ymax_snd[0], log_ymin_snd[0], log_ymin_snd[-1], log_ymax_snd[-1]]
plt.fill(x_snd, y_snd, color='lightcoral', alpha=0.5, label='FASER Region')

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

# plt.text(-1.3-9, -3.4, r'$\mathrm{Daya\,Bay}$',fontsize=15,rotation=0,color='mediumorchid')
plt.text(-0.06-9, -0.5, r'$\mathrm{PROSPECT}$',fontsize=13,rotation=0,color='darkturquoise')
plt.text(0.08-9, -2.9, r'$\mathrm{NEOS}$',fontsize=15,rotation=0,color='forestgreen')
plt.text(0.71-9, -0.80, r'$\mathrm{SK+IC+DC}$',fontsize=12,rotation=0,color='darkorange')
plt.text(1.5-9, -2.55, r'$^{3}\mathrm{H}$',fontsize=16,rotation=0,color='lime')
plt.text(2.25-9, -1.65, r'$^{187}\mathrm{Re}$',fontsize=16,rotation=0,color='violet')
plt.text(3.75-9, -2.10, r'$^{45}\mathrm{Ca}$',fontsize=16,rotation=0,color='plum')
plt.text(4.8-9, -4.0, r'$^{35}\mathrm{S}$',fontsize=16,rotation=0,color='y')
plt.text(4.15-9, -4.35, r'$^{63}\mathrm{Ni}$',fontsize=16,rotation=0,color='darkmagenta')
plt.text(4.55-9, -1.35, r'$^{144}\mathrm{Ce}-^{144}\mathrm{Pr}$',fontsize=16,rotation=0,color='crimson')
plt.text(4-9, -1.55, r'$^{64}\mathrm{Cu}$',fontsize=16,rotation=0,color='darkcyan')
plt.text(5.5-9, -3.3, r'$^{20}\mathrm{F}$',fontsize=16,rotation=0,color='green')
plt.text(5.5-9, -4.4, r'$^{7}\mathrm{Be}$',fontsize=16,rotation=0,color='rosybrown')
plt.text(6.2-9, -2, r'$\mathrm{Rovno}$',fontsize=14,rotation=0,color='tomato')
plt.text(6.6-9, -4.5, r'$\mathrm{Bugey}$',fontsize=12,rotation=50,color='turquoise')
plt.text(5.55-9, -5, r'$\mathrm{Borexino}$',fontsize=16,rotation=0,color='blue')
plt.text(6.85-9, -7, r'$\mathrm{PIENU}$',fontsize=16,rotation=0,color='gold')
plt.text(8.75-9, -8.2, r'$\mathrm{NA62}$',fontsize=16,rotation=0,color='teal')
plt.text(7.5-9, -5.1, r'$\mathrm{Super-K}$',fontsize=16,rotation=290,color='mediumseagreen')
plt.text(7.16-9, -3.0, r'$\mathrm{IHEP-JINR}$',fontsize=12,rotation=287.5,color='sienna')
plt.text(9.725-9, -2.55, r'$\mathrm{Belle}$',fontsize=16,rotation=0,color='darkgreen')
plt.text(9.16-9, -1, r'$\mathrm{NA3}$',fontsize=14,rotation=0,color='cornsilk')
plt.text(9.4-9, -6.5, r'$\mathrm{CHARM}$',fontsize=16,rotation=0,color='darkslategray')
plt.text(9.4-9, -7.5, r'$\mathrm{BEBC}$',fontsize=16,rotation=0,color='yellowgreen')
plt.text(7.2-9, -6, r'$\mathrm{PS191}$',fontsize=12,rotation=290,color='magenta')
plt.text(8.8-9, -9, r'$\mathrm{T2K} $',fontsize=16,rotation=0,color='purple')
plt.text(8.52-9, -5, r'$\mathrm{BESIII}$',fontsize=16,rotation=0,color='c')
plt.text(7.9-9, -3.4, r'$\mathrm{LNV\,Decays } $',fontsize=12,rotation=90,color='mediumaquamarine')
plt.text(10-9, -3.6, r'$\mathrm{L3}$',fontsize=16,rotation=0,color='salmon')
plt.text(9.91-9, -4.2, r'$\mathrm{DELPHI}$',fontsize=14,rotation=0,color='teal')
plt.text(11.7-9, -1.5, r'$\mathrm{CMS}$',fontsize=16,rotation=0,color='red')
plt.text(10.7-9, -4.9, r'$\mathrm{ATLAS}$',fontsize=16,rotation=0,color='mediumblue')
plt.text(11.1-9, -3.5, r'$\mathrm{Higgs}$',fontsize=16,rotation=0,color='sandybrown')
plt.text(12.1-9, -4.1, r'$\mathrm{EWPD}$',fontsize=16,rotation=0,color='darkgoldenrod')
# plt.text(-1.2-9, -4.8, r'$m^{\mathrm{sterile}}_{\mathrm{eff}},\,\Delta N_{\mathrm{eff}}$',fontsize=16,rotation=0,color='grey')
plt.text(0.2-9, -5.4, r'$\mathrm{CMB}$',fontsize=16,rotation=0,color='dimgrey')
plt.text(4.25-9, -11.5, r'$\mathrm{X-ray}$',fontsize=16,rotation=0,color='orangered')
# plt.text(-0.6-9, -5.8, r'$\mathrm{DM \, abundance}$',fontsize=16,rotation=0,color='grey')
plt.text(5.5-9, -11.5, r'$\mathrm{CMB}+\mathrm{BAO}+H_0$',fontsize=16,rotation=0,color='grey')
plt.text(8.75-9, -9.8, r'$\mathrm{BBN}$',fontsize=16,rotation=0,color='grey')
plt.text(10.4-9, -11.5, r'$\mathrm{Seesaw}$',fontsize=16,rotation=0,color='black')
plt.text(1.4-9, -9.2, r'$\mathrm{Supernovae}$',fontsize=16,alpha=0.7,rotation=0,color='darkslateblue')



# Customizing the plot
ax.set_xticks([-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6])
ax.xaxis.set_ticklabels([r'', r'$10^{-9}$', r'', r'', r'$10^{-6}$', r'', r'', r'$10^{-3}$', r'', r'', r'$1$', r'', r'', r'$10^{3}$', r'', r'', r'$10^{6}$'], fontsize=12)

ax.set_yticks([i for i in range(-30, 7)])
ax.yaxis.set_ticklabels([
    r'$10^{-30}$', r'', r'', r'$10^{-27}$', r'', r'', r'$10^{-24}$', r'', r'', r'$10^{-21}$',
    r'', r'', r'$10^{-18}$', r'', r'', r'$10^{-15}$', r'', r'', r'$10^{-12}$', r'', r'', r'$10^{-9}$',
    r'', r'', r'$10^{-6}$', r'', r'', r'$10^{-3}$', r'', r'', r'$1$', r'', r'', r'$10^{3}$',
    r'', r'', r'$10^{6}$'
], fontsize=12)

ax.set_xlabel(r'$m_N \, [\mathrm{GeV}]$', fontsize=16)
ax.set_ylabel(r'$|U_{eN}|^2$', fontsize=16)
ax.set_xlim(-9.1, 4.1)
ax.set_ylim(-30.1, 0.1)

# Set aspect ratio (golden ratio)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_aspect(2 * (x1 - x0) / (1 + Sqrt(5)) / (y1 - y0))

# Add the legend
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


# Show the plot
plt.tight_layout()

# Define the path and file name
save_path = "/home/as/Documents/Wolfram_Mathematica/Codes/Geometric Phase/data/Ue4_current_sterile.png"

# Save the plot
plt.savefig(save_path, dpi=300, bbox_inches='tight')

# Show the plot (optional)
plt.show()

