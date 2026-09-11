#%% VSM 
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import os
from sklearn.metrics import r2_score 
from mlognormfit import fit3
from mvshtools import mvshtools as mt
import re
from uncertainties import ufloat
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
#%% Funciones
def lineal(x,m,n):
    return m*x+n

def coercive_field(H, M):
    """
    Devuelve los valores de campo coercitivo (Hc) donde la magnetización M cruza por cero.
    
    Parámetros:
    - H: np.array, campo magnético (en A/m o kA/m)
    - M: np.array, magnetización (en emu/g)
    
    Retorna:
    - hc_values: list de valores Hc (puede haber más de uno si hay múltiples cruces por cero)
    """
    H = np.asarray(H)
    M = np.asarray(M)
    hc_values = []

    for i in range(len(M)-1):
        if M[i]*M[i+1] < 0:  # Cambio de signo indica cruce por cero
            # Interpolación lineal entre (H[i], M[i]) y (H[i+1], M[i+1])
            h1, h2 = H[i], H[i+1]
            m1, m2 = M[i], M[i+1]
            hc = h1 - m1 * (h2 - h1) / (m2 - m1)
            hc_values.append(hc)

    return hc_values
#%% levanto datos 
labels = ['M1','M2','M3','M4','M5','M6','M7','M9']

data_M1 = np.loadtxt('data_VSM/M01.txt', skiprows=12)
H_M1 = data_M1[:, 0]  # Gauss
m_M1 = data_M1[:, 1]  # emu
masa_M1 = 57.51/1000 #g
m_M1_norm = m_M1/masa_M1 #emu/g

data_M2 = np.loadtxt('data_VSM/M02.txt', skiprows=12)
H_M2 = data_M2[:, 0]  # Gauss
m_M2 = data_M2[:, 1]  # emu
masa_M2 = 37.14/1000#g
m_M2_norm = m_M2/masa_M2 #emu/g

data_M3 = np.loadtxt('data_VSM/M03.txt', skiprows=12)
H_M3 = data_M3[:, 0]  # Gauss
m_M3 = data_M3[:, 1]  # emu
masa_M3 = 36.06/1000#g
m_M3_norm = m_M3/masa_M3 #emu/g

data_M4 = np.loadtxt('data_VSM/M04.txt', skiprows=12)
H_M4 = data_M4[:, 0]  # Gauss
m_M4 = data_M4[:, 1]  # emu
masa_M4 = 26.56/1000#g
m_M4_norm = m_M4/masa_M4 #emu/g

data_M5 = np.loadtxt('data_VSM/M05.txt', skiprows=12)
H_M5 = data_M5[:, 0]  # Gauss
m_M5 = data_M5[:, 1]  # emu
masa_M5 = 41.21/1000#g
m_M5_norm = m_M5/masa_M5 #emu/g

data_M6 = np.loadtxt('data_VSM/M06.txt', skiprows=12)
H_M6 = data_M6[:, 0]  # Gauss
m_M6 = data_M6[:, 1]  # emu
masa_M6 = 38.39/1000#g
m_M6_norm = m_M6/masa_M6 #emu/g

data_M7 = np.loadtxt('data_VSM/M07.txt', skiprows=12)
H_M7 = data_M7[:, 0]  # Gauss
m_M7 = data_M7[:, 1]  # emu
masa_M7 = 7.09/1000#g
m_M7_norm = m_M7/masa_M7 #emu/g

data_M9 = np.loadtxt('data_VSM/M09.txt', skiprows=12)
H_M9 = data_M9[:, 0]  # Gauss
m_M9 = data_M9[:, 1]  # emu
masa_M9 = 37.10/1000#g
m_M9_norm = m_M9/masa_M9 #emu/g

data_M10 = np.loadtxt('data_VSM/M10.txt', skiprows=12)
H_M10 = data_M10[:, 0]  # Gauss
m_M10 = data_M10[:, 1]  # emu
masa_M9 = 27.07/1000#g
m_M9_norm = m_M9/masa_M9 #emu/g

data_M11 = np.loadtxt('data_VSM/M11.txt', skiprows=12)
H_M11 = data_M11[:, 0]  # Gauss
m_M11 = data_M11[:, 1]  # emu
masa_M11 = 33.74/1000#g
m_M11_norm = m_M11/masa_M11 #emu/g

data_M12 = np.loadtxt('data_VSM/M12.txt', skiprows=12)
H_M12 = data_M12[:, 0]  # Gauss
m_M12 = data_M12[:, 1]  # emu
masa_M12 = 31.97/1000#g
m_M12_norm = m_M12/masa_M12 #emu/g
#%% Ploteo Normalizado por masa
fig, (axup,axdwn) = plt.subplots(2,4, figsize=(18, 7), constrained_layout=True,sharex=True, sharey=True)

axup[0].plot(H_M1, m_M1_norm,'-', label=f'M1\n{masa_M1 *1000:.3f} mg')
axup[1].plot(H_M2, m_M2_norm,'-', label=f'M2\n{masa_M2 *1000:.3f} mg')
axup[2].plot(H_M3, m_M3_norm,'-', label=f'M3\n{masa_M3 *1000:.3f} mg')
axup[3].plot(H_M4, m_M4_norm,'-', label=f'M4\n{masa_M4 *1000:.3f} mg')

axdwn[0].plot(H_M5, m_M5_norm,'-', label=f'M5\n{masa_M5 *1000:.3f} mg')
axdwn[1].plot(H_M6, m_M6_norm,'-', label=f'M6\n{masa_M6 *1000:.3f} mg')
axdwn[2].plot(H_M7, m_M7_norm,'-', label=f'M7\n{masa_M7 *1000:.3f} mg')
axdwn[3].plot(H_M9, m_M9_norm,'-', label=f'M9\n{masa_M9 *1000:.3f} mg')


for ax in axup:
    ax.grid()
    ax.legend(ncol=1,frameon=True,shadow=True)# axins.set_ylim(-4, 4)   # rango eje Y

for a in axdwn:
    a.grid()
    a.legend(ncol=1,frameon=True,shadow=True)# axins.set_ylim(-4, 4)   # rango eje Y
    a.set_xlabel('H (G)')

for a in [axup[0],axdwn[0]]:
    a.set_ylabel('m (emu/g)')

# axins.grid()
# plt.savefig('NF@cit_13_260630_norm.png',dpi=300)
# plt.show()
#%%
resultados_fit_cit = {}
H_fit_arrays_cit = {}
m_fit_arrays_cit = {}
for nombre, H, m ,mass in [('NF', H_cit, m_cit, masa_cit)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=True,mass=mass)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Yfit - linear_contrib
    resultados_fit_cit[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Yfit, #### ojo con estoooooo
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    H_fit_arrays_cit[nombre] = fit.X
    m_fit_arrays_cit[nombre] = fit.Yfit
#%%#Ploteo los fits

fig2, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)
ax.plot(H_cit, m_cit_norm,'o-', c='C0',alpha=0.4,label=f'Datos originales\nC = {conc_cit} mg/mL\nm = {masa_cit*1000:.3f} mg')
ax.plot(resultados_fit_cit['NF']['H_fit'], resultados_fit_cit['NF']['m_fit'],
        '.-', c='C1',label=f'Ajuste\nm$_s$ = {resultados_fit_cit["NF"]["Ms"]:.2uS} emu/g\n$<\mu_\mu$>= {resultados_fit_cit["NF"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')
ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(title=name_cit,ncol=1,frameon=True,shadow=True,labelspacing=1.2)
ax.grid()
ax.set_title(name_cit)

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 
# Volver a graficar las curvas en el inset
axins.plot(H_cit, m_cit_norm,'o-', c='C0',alpha=0.4)
axins.plot(resultados_fit_cit['NF']['H_fit'], resultados_fit_cit['NF']['m_fit'],'.-',c='C1')
# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-30, 30)   # rango eje X
axins.set_ylim(-10, 10)   # rango eje Y
axins.grid()
plt.savefig(name_cit+'_fit.png',dpi=300)
plt.show()

#%% Ahora lo mismo pero con NF@PAA 
#%% NF@PAA-260630 
data_PAA = np.loadtxt('data/NF@PAA260630.txt', skiprows=12)
H_PAA = data_PAA[:, 0]  # Gauss
m_PAA = data_PAA[:, 1]  # emu
    
conc_PAA = 24.5 #mg/mL Magnetita

conc_PAA_mm = conc_PAA/1000 #mg np/mg de solvente
masa_PAA = (0.1339-0.0809)*conc_PAA_mm
m_PAA_norm = m_PAA/masa_PAA #emu/g
name_PAA='NF@PAA 260630'

fig, a = plt.subplots( figsize=(8, 6), constrained_layout=True)
a.plot(H_PAA, m_PAA, '.-', label=name_PAA)
a.set_ylabel('m (emu)')
a.legend()
a.grid()
a.set_title(name_PAA)
a.set_xlabel('H (G)')
a.set_ylabel('m (emu)')
plt.show()
#%% NF@PAA-260630 Normalizado por masa
fig2, b = plt.subplots( figsize=(8, 6), constrained_layout=True)
b.plot(H_PAA, m_PAA_norm,'.-', label=f'Normalizada\nC = {conc_PAA} mg/mL\nm = {masa_PAA *1000:.3f} mg')
b.set_ylabel('m (emu/g)')
b.grid()
b.set_title('NF@PAA 260630 - Normalizada por masa')
b.set_xlabel('H (G)')
b.set_ylabel('m (emu/g)')
axins = inset_axes(b,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=b.transAxes,
    borderpad=0) 
b.legend(title=name_PAA,ncol=1,frameon=True,shadow=True,labelspacing=1.2)
# Volver a graficar las curvas en el inset
axins.plot(H_PAA, m_PAA_norm,'.-')
# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20,20)   # rango eje X
axins.set_ylim(-4, 4)   # rango eje Y
axins.grid()
plt.savefig('NF@PAA_13_260630_norm.png',dpi=300)
plt.show()
#%%
resultados_fit_PAA = {}
H_fit_arrays_PAA = {}
m_fit_arrays_PAA = {}
for nombre, H, m ,mass in [('NF', H_PAA, m_PAA, masa_PAA)]:
    H_anhist, m_anhist = mt.anhysteretic(H, m)
    fit = fit3.session(H_anhist, m_anhist, fname=nombre, divbymass=True,mass=mass)
    fit.fix('sig0')
    fit.fix('mu0')
    fit.free('dc')
    fit.fit()
    fit.update()
    fit.free('sig0')
    fit.free('mu0')
    fit.set_yE_as('sep')
    fit.fit()
    fit.update()
    fit.save()
    fit.print_pars()
    # Obtengo la contribución lineal usando los parámetros del fit
    C = fit.params['C'].value
    dc = fit.params['dc'].value
    linear_contrib = lineal(fit.X, C, dc)
    m_fit_sin_lineal = fit.Yfit - linear_contrib
    resultados_fit_PAA[nombre]={'H_anhist': H_anhist,
                            'm_anhist': m_anhist,
                            'H_fit': fit.X,
                            'm_fit': fit.Yfit,
                            'm_fit_sin_lineal': m_fit_sin_lineal,
                            'linear_contrib': linear_contrib,
                            'Ms':fit.derived_parameters()['m_s'],
                            'fit': fit}
    H_fit_arrays_PAA[nombre] = fit.X
    m_fit_arrays_PAA[nombre] = fit.Yfit


#%% Ploteo los fits

fig, ax = plt.subplots( figsize=(8, 6), constrained_layout=True)

ax.plot(H_PAA, m_PAA_norm,'o-', c='C0',alpha=0.4,label=f'Datos originales\nC = {conc_PAA} mg/mL\nm = {masa_PAA*1000:.3f} mg')

ax.plot(resultados_fit_PAA['NF']['H_fit'], resultados_fit_PAA['NF']['m_fit'],
        '.-', c='C1',label=f'Ajuste\nm$_s$ = {resultados_fit_PAA["NF"]["Ms"]:.2uS} emu/g\n$<\mu_\mu$>= {resultados_fit_PAA["NF"]["fit"].derived_parameters()["<mu>_mu"]:.1uS} $\mu_B$')


ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(title=name_PAA,ncol=1,frameon=True,shadow=True,labelspacing=1.2)
ax.grid()
ax.set_title(name_PAA)

axins = inset_axes(    ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0) 

# Volver a graficar las curvas en el inset
axins.plot(H_PAA, m_PAA_norm,'o-', c='C0',alpha=0.4)
axins.plot(resultados_fit_PAA['NF']['H_fit'], resultados_fit_PAA['NF']['m_fit'],'.-',c='C1')

# Definir región de zoom (AJUSTAR ESTOS VALORES)
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()

plt.savefig(name_PAA+'_fit.png',dpi=300)
plt.show()
# %% Comparo los fits descontando la contribución lineal
fig ,ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

ax.plot(resultados_fit_cit['NF']['H_fit'], resultados_fit_cit['NF']['m_fit_sin_lineal'],
        '.-', c='C0',
        label=f'NF@cit fit\nm$_s$ = {resultados_fit_cit["NF"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit_cit["NF"]["fit"].derived_parameters()["<mu>_mu"]:.2uS} $\mu_B$')


ax.plot(resultados_fit_PAA['NF']['H_fit'], resultados_fit_PAA['NF']['m_fit_sin_lineal'],
        '.-', c='C1',
        label=f'NF@PAA fit\nm$_s$ = {resultados_fit_PAA["NF"]["Ms"]:.1uS} emu/g\n$<\mu_\mu$>= {resultados_fit_PAA["NF"]["fit"].derived_parameters()["<mu>_mu"]:.2uS} $\mu_B$')



ax.set_ylabel('m (emu/g)')
ax.set_xlabel('H (G)')
ax.legend(ncol=1,frameon=True,shadow=True,labelspacing=1.2)
ax.grid()
ax.set_title(name_PAA)

axins = inset_axes(ax,
    width="40%",
    height="40%",
    loc='lower right',
    bbox_to_anchor=(-0.01, 0.08, 0.98, 1), 
    bbox_transform=ax.transAxes,
    borderpad=0)

axins.plot(resultados_fit_cit['NF']['H_fit'], resultados_fit_cit['NF']['m_fit_sin_lineal'],'.-', c='C0')
axins.plot(resultados_fit_PAA['NF']['H_fit'], resultados_fit_PAA['NF']['m_fit_sin_lineal'],'.-', c='C1')
axins.set_xlim(-20, 20)   # rango eje X
axins.set_ylim(-8, 8)   # rango eje Y
axins.grid()


plt.savefig('comparativa_NF@cit_NF@PAA_fit_sin_lineal.png',dpi=300)
# %%
