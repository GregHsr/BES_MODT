#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 15:12:11 2024

@author: nturner
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# CONSTANTES

Re=23000
U_inlet=13
D=0.026


# Load data - DONNEES PAPIERS COOPER 1993
data_rd1_cw_uv = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_1/ij2lr-10-cw-uv.dat")
data_rd1_cw_vv = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_1/ij2lr-10-cw-vv.dat")
data_rd1_sw_mu = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_1/ij2lr-10-sw-mu.dat")
data_rd1_sw_uu = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_1/ij2lr-10-sw-uu.dat")

data_rd3_cw_uv = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_3/ij2lr-30-cw-uv.dat")
data_rd3_cw_vv = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_3/ij2lr-30-cw-vv.dat")
data_rd3_sw_mu = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_3/ij2lr-30-sw-mu.dat")
data_rd3_sw_uu = np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_rD_3/ij2lr-30-sw-uu.dat")

data_nu=np.loadtxt("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/DATA_EXP/exp_data_nu/ij2lr-nuss.dat")



# Read data - EXPERIMENTALES STAR-CCM - QUESTION 2
yparoi_starccm_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/heat_transfert.csv", usecols=[0], header=None, dtype=float,skiprows=1)
nu_starccm_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/heat_transfert.csv", usecols=[1], header=None, dtype=float,skiprows=1)

pos_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/mean_vel_q2.csv", usecols=[0], header=None, dtype=float,skiprows=1)
mean_velocity_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/mean_vel_q2.csv", usecols=[1], header=None, dtype=float,skiprows=1)
k_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/k_q2.csv", usecols=[1], header=None, dtype=float,skiprows=1)

pos_d3_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/mean_vel_d3_q2.csv", usecols=[0], header=None, dtype=float,skiprows=1)
mean_vel_d3_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/mean_vel_d3_q2.csv", usecols=[1], header=None, dtype=float,skiprows=1)
k_d3_q2=pd.read_csv("/home/nturner/Env_Informatique/3A/modt/BE_MTSS_impiging_jet/BE/BES_MODT/k_d3_q2.csv", usecols=[1], header=None, dtype=float,skiprows=1)




# Read data - COOPER 1993
yd_cw=data_rd1_cw_uv[:,0]
yd_sw=data_rd1_sw_mu[:,0]
rd=data_nu[:,0]
#r/D=1
uv_Ub2_1=data_rd1_cw_uv[:,1]
vv_Ub2_1=data_rd1_cw_vv[:,1]
U_Ub_1=data_rd1_sw_mu[:,1]
uu_Ub2_1=data_rd1_sw_uu[:,1]
#r/D=3
uv_Ub2_3=data_rd3_cw_uv[:,1]
vv_Ub2_3=data_rd3_cw_vv[:,1]
U_Ub_3=data_rd3_sw_mu[:,1]
uu_Ub2_3=data_rd3_sw_uu[:,1]
#nu/Re**0.7
nu_Re=data_nu[:,1]


##########################################
# Calculer énergie cinétique turbulente:
# k = 1/2 (<u'u'> + <v'v'> + <w'w'>)
# Hypo => <w'w'> du même ordre que <v'v'> (pas déconnant pour un jet)
# => k = 0.5 <u'u'> + <v'v'>

# #r/D=1
# k_1 = 0.5 * uu_Ub2_1 + vv_Ub2_1

# #r/D=3
# k_3 = 0.5 * uu_Ub2_3 + vv_Ub2_3


# FAIRE INTERPOLATION !!!!

# Interpolate uu and vv data to a common y/D grid
common_yD = np.linspace(min(yd_cw.min(), yd_sw.min()), max(yd_cw.max(), yd_sw.max()), 40)

interp_uu_1 = interp1d(yd_sw, uu_Ub2_1, kind='linear', bounds_error = False)
interp_vv_1 = interp1d(yd_cw, vv_Ub2_1, kind='linear', bounds_error = False)

uu_interpolated_1 = interp_uu_1(common_yD)
vv_interpolated_1 = interp_vv_1(common_yD)

# Compute k_1
k_1 = 0.5 * uu_interpolated_1 + vv_interpolated_1


interp_uu_3 = interp1d(yd_sw, uu_Ub2_3, kind='linear', bounds_error = False)
interp_vv_3 = interp1d(yd_cw, vv_Ub2_3, kind='linear', bounds_error = False)

uu_interpolated_3 = interp_uu_3(common_yD)
vv_interpolated_3 = interp_vv_3(common_yD)

# Compute k_3
k_3 = 0.5 * uu_interpolated_3 + vv_interpolated_3
##########################################




### Plot

## r/D = 1
#plt.figure(figsize=(10, 6))
#plt.plot(yd_cw, uv_Ub2_1, label="-uv/Ub²", marker='o')
#plt.plot(yd_cw, vv_Ub2_1, label="vv/Ub²", marker='x')
#plt.title("r/D = 1", fontsize=16, fontweight='bold')
#plt.xlabel("y/D", fontsize=12)
#plt.ylabel("", fontsize=12)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle='--', alpha=0.7)
#
#plt.figure(figsize=(10, 6))
#plt.plot(yd_sw, U_Ub_1, label="u/Ub", marker='o')
#plt.plot(yd_sw, uu_Ub2_1, label="uu/Ub²", marker='x')
#plt.title("r/D = 1", fontsize=16, fontweight='bold')
#plt.xlabel("y/D", fontsize=12)
#plt.ylabel("", fontsize=12)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle='--', alpha=0.7)
#
#
## r/D = 3
#plt.figure(figsize=(10, 6))
#plt.plot(yd_cw, uv_Ub2_3, label="-uv/Ub²", marker='o')
#plt.plot(yd_cw, vv_Ub2_3, label="vv/Ub²", marker='x')
#plt.title("r/D = 3", fontsize=16, fontweight='bold')
#plt.xlabel("y/D", fontsize=12)
#plt.ylabel("", fontsize=12)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle='--', alpha=0.7)
#
#plt.figure(figsize=(10, 6))
#plt.plot(yd_sw, U_Ub_3, label="u/Ub", marker='o')
#plt.plot(yd_sw, uu_Ub2_3, label="uu/Ub²", marker='x')
#plt.title("r/D = 3", fontsize=16, fontweight='bold')
#plt.xlabel("y/D", fontsize=12)
#plt.ylabel("", fontsize=12)
#plt.legend(fontsize=12)
#plt.grid(True, linestyle='--', alpha=0.7)



# QUESTION 2 - STAR-CCM
# Nusselt 
plt.figure(figsize=(10, 6))
plt.plot(rd, nu_Re, label="Cooper 1993", marker='o')
plt.scatter(yparoi_starccm_q2/0.0265, nu_starccm_q2/(Re**(0.7)), label="Star-CCM", marker='x', color='orange')
plt.title("", fontsize=16, fontweight='bold')
plt.xlabel("r/D", fontsize=12)
plt.ylabel("nu/Re**0.7", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

## r/D = 1
# Mean Velocity
plt.figure(figsize=(10, 6))
plt.plot(yd_sw, U_Ub_1, label="u/Ub", marker='o')
plt.scatter((pos_q2/D-8)*(-1), mean_velocity_q2/U_inlet, label="u/Ub", marker='x', color='orange')
plt.title("r/D = 1", fontsize=16, fontweight='bold')
plt.xlabel("y/D", fontsize=12)
plt.ylabel("", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)


## Turbulent kinetic energy
plt.figure(figsize=(10, 6))
plt.plot(common_yD, k_1, label="Cooper 1993", marker='o', linestyle='-', color='b')
plt.scatter((pos_d3_q2/D-8)*(-1), k_q2/(U_inlet**2), label="Star-CCM", marker='x', color='orange')
plt.title("r/D = 1", fontsize=16, fontweight='bold')
plt.xlabel("y/D", fontsize=12)
plt.ylabel("k NORMALISÉ", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



## r/D = 3
# Mean Velocity
plt.figure(figsize=(10, 6))
plt.plot(yd_sw, U_Ub_3, label="u/Ub", marker='o')
plt.scatter((pos_d3_q2/D-8)*(-1), mean_vel_d3_q2/U_inlet, label="u/Ub", marker='x', color='orange')
plt.title("r/D = 3", fontsize=16, fontweight='bold')
plt.xlabel("y/D", fontsize=12)
plt.ylabel("", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Turbulent kinetic energy
plt.figure(figsize=(10, 6))
plt.plot(common_yD, k_3, label="Cooper 1993", marker='o', linestyle='-', color='b')
plt.scatter((pos_d3_q2/D-8)*(-1), k_d3_q2/(U_inlet**2), label="Star-CCM", marker='x', color='orange')
plt.title("r/D = 3", fontsize=16, fontweight='bold')
plt.xlabel("y/D", fontsize=12)
plt.ylabel("k NORMALISÉ", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



# Affichage
plt.tight_layout()
plt.show()