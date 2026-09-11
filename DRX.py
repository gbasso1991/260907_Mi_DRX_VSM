#%% DRX en polvo de muestras M1, M2, M3, M4, M5, M6, M7, M9, M10, M11, M12
# Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
import matplotlib as mpl
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import minimum_filter, gaussian_filter
# %% Data
data_M1 = np.loadtxt('data_DRX/M01.ASC', skiprows=0)
data_M2 = np.loadtxt('data_DRX/M02.ASC', skiprows=0)
data_M3 = np.loadtxt('data_DRX/M03.ASC', skiprows=0)
data_M4 = np.loadtxt('data_DRX/M04.ASC', skiprows=0)
data_M5 = np.loadtxt('data_DRX/M05.ASC', skiprows=0)
data_M6 = np.loadtxt('data_DRX/M06.ASC', skiprows=0)
data_M7 = np.loadtxt('data_DRX/M07.ASC', skiprows=0)
data_M9 = np.loadtxt('data_DRX/M09.ASC', skiprows=0)
data_M10 = np.loadtxt('data_DRX/M10.ASC', skiprows=0)
data_M11 = np.loadtxt('data_DRX/M11.ASC', skiprows=0)
data_M12 = np.loadtxt('data_DRX/M12.ASC', skiprows=0)

dostheta_M1 = data_M1[:, 0]
intensity_M1 = data_M1[:, 1]

dostheta_M2 = data_M2[:, 0]
intensity_M2 = data_M2[:, 1]

dostheta_M3 = data_M3[:, 0]
intensity_M3 = data_M3[:, 1]

dostheta_M4 = data_M4[:, 0]
intensity_M4 = data_M4[:, 1]

dostheta_M5 = data_M5[:, 0]
intensity_M5 = data_M5[:, 1]

dostheta_M6 = data_M6[:, 0]
intensity_M6 = data_M6[:, 1]

dostheta_M7 = data_M7[:, 0]
intensity_M7 = data_M7[:, 1]

dostheta_M9 = data_M9[:, 0]
intensity_M9 = data_M9[:, 1]

dostheta_M10 = data_M10[:, 0]
intensity_M10 = data_M10[:, 1]

dostheta_M11 = data_M11[:, 0]
intensity_M11 = data_M11[:, 1]

dostheta_M12 = data_M12[:, 0]
intensity_M12 = data_M12[:, 1]

labels=['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M9', 'M10', 'M11', 'M12']
colors=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10']
# %% ploteo todos 
fig,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(10,8), constrained_layout=True,sharex=True)
ax.plot(dostheta_M1, intensity_M1/max(intensity_M1)+0.0,lw=0.9, label=labels[0],zorder=1)
ax.plot(dostheta_M2, intensity_M2/max(intensity_M2)+0.2,lw=0.9, label=labels[1],zorder=2)
ax.plot(dostheta_M3, intensity_M3/max(intensity_M3)+0.4,lw=0.9, label=labels[2],zorder=3)    
ax.plot(dostheta_M4, intensity_M4/max(intensity_M4)+0.6,lw=0.9, label=labels[3],zorder=1)

ax2.plot(dostheta_M5, intensity_M5/max(intensity_M5)+0.0,lw=0.9, label=labels[4],zorder=2)
ax2.plot(dostheta_M6, intensity_M6/max(intensity_M6)+0.2,lw=0.9, label=labels[5],zorder=3)    
ax2.plot(dostheta_M7, intensity_M7/max(intensity_M7)+0.4,lw=0.9, label=labels[6],zorder=1)
ax2.plot(dostheta_M9, intensity_M9/max(intensity_M9)+0.6,lw=0.9, label=labels[7],zorder=2)

ax3.plot(dostheta_M10, intensity_M10/max(intensity_M10)+0.4,lw=0.9, label=labels[8],zorder=3)    
ax3.plot(dostheta_M11, intensity_M11/max(intensity_M11)+0.0,lw=0.9, label=labels[9],zorder=1)
ax3.plot(dostheta_M12, intensity_M12/max(intensity_M12)+0.2,lw=0.9, label=labels[10],zorder=2)


for a in (ax,ax2,ax3):
    a.legend(ncol=2,frameon=True,shadow=True)
    a.grid()            
    a.set_ylabel('Intensidad (u.a.)')
    a.set_ylim(0,)
ax3.set_xlim(20,80)
ax3.set_xlabel('2θ (°)')
plt.suptitle('DRX en polvo')
plt.savefig('DRX_M1_M2_M3_M4_M5_M6_M7_M9_M10_M11_M12.png')
plt.show()

#%% Analisis de picos M1 
peaks_M1, props_M1 = find_peaks(intensity_M1,prominence=25,width=2,distance=50)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M1, intensity_M1,'-', label='raw')
ax.plot(dostheta_M1[peaks_M1], intensity_M1[peaks_M1], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M1 = zip(peaks_M1, dostheta_M1[peaks_M1], intensity_M1[peaks_M1])
print('M1',f'{len(peaks_M1)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M1:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%% M2
peaks_M2, props_M2 = find_peaks(intensity_M2,prominence=25,width=2,distance=30)


fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M2, intensity_M2,'-', label='raw')
ax.plot(dostheta_M2[peaks_M2], intensity_M2[peaks_M2], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M2 = zip(peaks_M2, dostheta_M2[peaks_M2], intensity_M2[peaks_M2])
print('M2',f'{len(peaks_M2)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M2:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')

#%% M3
peaks_M3, props_M3 = find_peaks(intensity_M3,prominence=24,width=2,distance=30)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M3, intensity_M3,'-', label='raw')
ax.plot(dostheta_M3[peaks_M3], intensity_M3[peaks_M3], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M3 = zip(peaks_M3, dostheta_M3[peaks_M3], intensity_M3[peaks_M3])
print('M3',f'{len(peaks_M3)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M3:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')   

#%% M4 
peaks_M4, props_M4 = find_peaks(intensity_M4,prominence=22,width=8,distance=20)


fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M4, intensity_M4,'-', label='raw')
ax.plot(dostheta_M4[peaks_M4], intensity_M4[peaks_M4], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M4 = zip(peaks_M4, dostheta_M4[peaks_M4], intensity_M4[peaks_M4])
print('M4',f'{len(peaks_M4)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M4:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')       

#%% M5 
peaks_M5, props_M5 = find_peaks(intensity_M5,prominence=6,
                                width=7,
                                distance=20)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M5, intensity_M5,'-', label='raw')
ax.plot(dostheta_M5[peaks_M5], intensity_M5[peaks_M5], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend(title='M5',frameon=True,shadow=True)
plt.show()

picos_M5 = zip(peaks_M5, dostheta_M5[peaks_M5], intensity_M5[peaks_M5])
print('M5',f'{len(peaks_M5)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M5:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')       
#%% M6 
peaks_M6, props_M6 = find_peaks(intensity_M6,
                                prominence=15,
                                width=3,
                                distance=20)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M6, intensity_M6,'-', label='raw')
ax.plot(dostheta_M6[peaks_M6], intensity_M6[peaks_M6], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M6 = zip(peaks_M6, dostheta_M6[peaks_M6], intensity_M6[peaks_M6])
print('M6',f'{len(peaks_M6)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M6:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')   

#%% M7 
peaks_M7, props_M7 = find_peaks(intensity_M7,
                                prominence=20,
                                width=7,
                                distance=20)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M7, intensity_M7,'-', label='raw')
ax.plot(dostheta_M7[peaks_M7], intensity_M7[peaks_M7], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M7 = zip(peaks_M7, dostheta_M7[peaks_M7], intensity_M7[peaks_M7])
print('M7',f'{len(peaks_M7)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M7:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')   
#%% M9
peaks_M9, props_M9 = find_peaks(intensity_M9,
                                height=30,
                                prominence=10,
                                width=3,
                                distance=10)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M9, intensity_M9,'-', label='raw')
ax.plot(dostheta_M9[peaks_M9], intensity_M9[peaks_M9], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend(title='M9',frameon=True,shadow=True)
plt.show()

picos_M9 = zip(peaks_M9, dostheta_M9[peaks_M9], intensity_M9[peaks_M9])
print('M9',f'{len(peaks_M9)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M9:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')       

#%% M10 
peaks_M10, props_M10 = find_peaks(intensity_M10,
                                height=24,
                                prominence=17,
                                width=2,
                                distance=15)


fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M10, intensity_M10,'-', label='raw')
ax.plot(dostheta_M10[peaks_M10], intensity_M10[peaks_M10], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend(title='M10',frameon=True,shadow=True)
plt.show()

picos_M10 = zip(peaks_M10, dostheta_M10[peaks_M10], intensity_M10[peaks_M10])
print('M10',f'{len(peaks_M10)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M10:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')    

#%%  M11
peaks_M11, props_M11 = find_peaks(intensity_M11,
                                  height=28,
                                prominence=10,
                                width=4,
                                distance=20)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M11, intensity_M11,'-', label='raw')
ax.plot(dostheta_M11[peaks_M11], intensity_M11[peaks_M11], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M11 = zip(peaks_M11, dostheta_M11[peaks_M11], intensity_M11[peaks_M11])
print('M11',f'{len(peaks_M11)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M11:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')


#%% M12 
peaks_M12, props_M12 = find_peaks(intensity_M12,prominence=25,width=2,distance=50)

fig,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M12, intensity_M12,'-', label='raw')
ax.plot(dostheta_M12[peaks_M12], intensity_M12[peaks_M12], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.show()

picos_M12 = zip(peaks_M12, dostheta_M12[peaks_M12], intensity_M12[peaks_M12])
print('M12',f'{len(peaks_M12)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M12:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%%
