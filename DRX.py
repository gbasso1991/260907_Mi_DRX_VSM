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

#%%pvoigt_fondo
def pvoigt_fondo(x, A, xc, fwhm, eta, b0, b1):
    """
    Calcula un perfil pseudo-Voigt sobre un fondo lineal.

    Parámetros
    ----------
    x : array_like
        Posiciones angulares 2theta, en grados.
    A : float
        Amplitud del pico respecto del fondo.
    xc : float
        Posición del centro del pico, en grados de 2theta.
    fwhm : float
        Anchura a media altura del pico (FWHM), en grados.
    eta : float
        Fracción lorentziana del perfil, entre 0 y 1.
        eta = 0 corresponde a una gaussiana y eta = 1
        a una lorentziana.
    b0 : float
        Intensidad del fondo en xc.
    b1 : float
        Pendiente del fondo lineal, en unidades de intensidad
        por grado.

    Retorna
    -------
    array_like
        Intensidad calculada para cada posición angular x.
    """
    z = (x-xc)/fwhm
    gauss = np.exp(-4*np.log(2)*z**2)
    lorentz = 1/(1+4*z**2)
    perfil = (1-eta)*gauss + eta*lorentz
    return b0 + b1*(x-xc) + A*perfil

#%% analizar_pico
def analizar_pico(x, y, centro_ref, ventana=0.9, picos_detectados=None):
    """
    Ajusta un pico de difracción mediante una pseudo-Voigt
    sobre un fondo lineal.

    Si se proporcionan picos_detectados, utiliza el máximo detectado
    más cercano a centro_ref como estimación inicial del centro y
    de la amplitud. Si no encuentra ninguno dentro de los límites
    permitidos para el centro, utiliza la inicialización original.

    Parámetros
    ----------
    x : array_like
        Posiciones angulares 2theta, en grados.
    y : array_like
        Intensidades medidas correspondientes a x.
    centro_ref : float
        Posición angular aproximada del pico, en grados de 2theta.
    ventana : float, opcional
        Semianchura de la región de ajuste, en grados.
        Por defecto, 0.9 grados.
    picos_detectados : iterable o None, opcional
        Iterable de tuplas (índice, posición angular, intensidad),
        como el obtenido con find_peaks. Si es None, se utiliza
        la inicialización original.

    Retorna
    -------
    xp : ndarray
        Posiciones angulares utilizadas en el ajuste.
    popt : ndarray
        Parámetros ajustados: [A, xc, fwhm, eta, b0, b1].
    pcov : ndarray
        Matriz de covarianza estimada de los parámetros ajustados.
    """
    mask = (x >= centro_ref-ventana) & (x <= centro_ref+ventana)
    xp, yp = x[mask], y[mask]
    if len(xp) < 10:
        raise ValueError('Pocos puntos para ajustar el pico')

    # Estimaciones iniciales del fondo y del ancho
    b0 = np.percentile(yp, 10)
    fwhm0 = max(3*np.median(np.diff(xp)), 0.15)

    # Inicialización del centro y la amplitud
    candidatos = []
    if picos_detectados is not None:
        candidatos = [
            (indice, theta, intensidad)
            for indice, theta, intensidad in picos_detectados
            if centro_ref-ventana/2 <= theta <= centro_ref+ventana/2
        ]

    if candidatos:
        indice, xc0, intensidad_max = min(
            candidatos, key=lambda pico: abs(pico[1]-centro_ref)
        )
        A0 = max(intensidad_max-b0, 1e-6)
    else:
        imax = np.argmax(yp)
        xc0 = xp[imax]
        A0 = max(yp[imax]-b0, 1e-6)

    popt, pcov = curve_fit(
        pvoigt_fondo, xp, yp,
        p0=[A0, xc0, fwhm0, 0.5, b0, 0],
        bounds=(
            [0, centro_ref-ventana/2, 0.005, 0, -np.inf, -np.inf],
            [np.inf, centro_ref+ventana/2, 2*ventana, 1, np.inf, np.inf]
        ),
        maxfev=30000
    )
    return xp, popt, pcov
#%% Parámetros para Análisis de Scherrer: pseudo-Voigt + fondo lineal
K = 0.9
lambda_X = 1.5406  # Å, Cu Kα
beta_inst_deg = None  # Sin patrón de referencia: no se corrige

# Posiciones aproximadas de las reflexiones de magnetita (2theta)
reflexiones = {
    '(220)': 30.1,
    '(311)': 35.5,
    '(400)': 43.1,
    '(422)': 53.4,
    '(511)': 57.0,
    '(440)': 62.6,
    '(620)': 71.0,
    '(533)': 74.0,
    '(444)': 79.0}

#%% Patrón de calibración LaB6
data_LaB6 = np.loadtxt('data_DRX/LAB6PHI2.ASC', skiprows=0)
dostheta_LaB6 = data_LaB6[:, 0]
intensity_LaB6 = data_LaB6[:, 1]

# Visualización del difractograma
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
ax.set_xlim(20,85)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('Patrón de calibración LaB$_6$')
ax.grid()
ax.legend()
plt.show()


#%% Detección de picos del patrón LaB6
prominencia_LaB6 = 0.01*np.ptp(intensity_LaB6)
peaks_LaB6, props_LaB6 = find_peaks(
    intensity_LaB6,
    prominence=prominencia_LaB6,
    width=2,
    distance=5
)

picos_LaB6 = list(zip(peaks_LaB6,dostheta_LaB6[peaks_LaB6],intensity_LaB6[peaks_LaB6]))

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
ax.plot(dostheta_LaB6[peaks_LaB6], intensity_LaB6[peaks_LaB6], 'r.', ms=6, label='Picos detectados')
ax.set_xlim(20, 85)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('Detección de máximos del patrón LaB$_6$')
ax.grid()
ax.legend()
plt.show()

print(f'LaB6: {len(peaks_LaB6)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i, theta, intens in picos_LaB6:
    print(f'{i:4} | {theta:7.2f} | {intens:10.0f}')


#%% Parámetro de red del LaB6 a partir de las posiciones de los picos
lambda_Ka1 = 1.5406  # Å
a_ref_LaB6 = 4.156826  # Å; valor certificado de referencia

# Reflexiones cúbicas de LaB6: (hkl) y h²+k²+l²
reflexiones_LaB6 = {
    '(100)': (1, 21.34),
    '(110)': (2, 30.38),
    '(111)': (3, 37.42),
    '(200)': (4, 43.50),
    '(210)': (5, 48.94),
    '(211)': (6, 53.96),
    '(220)': (8, 63.20),
    '(300)/(221)': (9, 67.54),
    '(310)': (10, 71.72),
    '(311)': (11, 75.82),
    '(222)': (12, 79.84),
    '(320)': (13, 83.82),
    '(321)': (14, 87.78),
    '(331)': (19, 95.66),
    '(420)': (20, 99.64)
}

resultados_LaB6_red = []

for hkl, (N, centro_ref) in reflexiones_LaB6.items():
    try:
        xp, popt, pcov = analizar_pico(
            dostheta_LaB6, intensity_LaB6,
            centro_ref, ventana=0.45
        )
        A, xc, fwhm_deg, eta, b0, b1 = popt

        theta_rad = np.deg2rad(xc/2)
        a_A = lambda_Ka1*np.sqrt(N)/(2*np.sin(theta_rad))

        xc_err = np.sqrt(max(pcov[1, 1], 0))
        theta_err_rad = np.deg2rad(xc_err/2)
        a_err_A = a_A/np.tan(theta_rad)*theta_err_rad

        resultados_LaB6_red.append({
            'Reflexión': hkl,
            'N': N,
            '2theta (°)': xc,
            'FWHM (°)': fwhm_deg,
            'Error FWHM (°)': np.sqrt(max(pcov[2, 2], 0)),
            'a (Å)': a_A,
            'Error a (Å)': a_err_A
        })
    except (ValueError, RuntimeError) as e:
        print(f'{hkl}: no se pudo ajustar ({e})')

tabla_LaB6_red = pd.DataFrame(resultados_LaB6_red)
print(tabla_LaB6_red.to_string(index=False, formatters={
    '2theta (°)': lambda v: f'{v:.4f}',
    'FWHM (°)': lambda v: f'{v:.4f}',
    'Error FWHM (°)': lambda v: f'{v:.4f}',
    'a (Å)': lambda v: f'{v:.6f}',
    'Error a (Å)': lambda v: f'{v:.6f}'
}))

if len(tabla_LaB6_red) > 0:
    print(f'\nParámetro de red de referencia: {a_ref_LaB6:.7f} Å')
    print(f'Promedio simple obtenido: {tabla_LaB6_red["a (Å)"].mean():.7f} Å')
#%%
#%% Doble pseudo-Voigt para el doblete Cu Kalpha
lambda_Ka1 = 1.5406  # Å
lambda_Ka2 = 1.5444  # Å
rel_int_Ka2 = 0.5    # Intensidad Kalpha2 / Kalpha1

def doblete_pvoigt_fondo(x, A, xc, fwhm, eta, b0, b1):
    """
    Ajusta un doblete Cu Kalpha1/Kalpha2 con dos pseudo-Voigt
    de igual anchura y fracción lorentziana, sobre un fondo lineal.

    A es la amplitud de Kalpha1.
    xc es la posición 2theta de Kalpha1.
    """
    theta1 = np.deg2rad(xc/2)
    theta2 = np.arcsin(np.clip(
        lambda_Ka2/lambda_Ka1*np.sin(theta1), -1, 1
    ))
    xc2 = np.rad2deg(2*theta2)

    z1 = (x-xc)/fwhm
    z2 = (x-xc2)/fwhm

    gauss1 = np.exp(-4*np.log(2)*z1**2)
    lorentz1 = 1/(1+4*z1**2)
    perfil1 = (1-eta)*gauss1 + eta*lorentz1

    gauss2 = np.exp(-4*np.log(2)*z2**2)
    lorentz2 = 1/(1+4*z2**2)
    perfil2 = (1-eta)*gauss2 + eta*lorentz2

    return b0 + b1*(x-xc) + A*(perfil1 + rel_int_Ka2*perfil2)

#%% Ajuste del doblete Cu Kalpha en LaB6
resultados_LaB6_doblete = []
fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.7', lw=0.8, label='LaB$_6$')

for hkl, (N, centro_ref) in reflexiones_LaB6.items():
    if centro_ref > 85:
        continue
    try:
        ventana = 0.45
        mask = (
            (dostheta_LaB6 >= centro_ref-ventana) &
            (dostheta_LaB6 <= centro_ref+ventana)
        )
        xp = dostheta_LaB6[mask]
        yp = intensity_LaB6[mask]
        if len(xp) < 10:
            raise ValueError('Pocos puntos para ajustar el pico')

        b0 = np.percentile(yp, 10)
        imax = np.argmax(yp)
        xc0 = xp[imax]
        A0 = max(yp[imax]-b0, 1e-6)
        fwhm0 = max(3*np.median(np.diff(xp)), 0.05)

        popt, pcov = curve_fit(
            doblete_pvoigt_fondo, xp, yp,
            p0=[A0, xc0, fwhm0, 0.5, b0, 0],
            bounds=(
                [0, centro_ref-0.25, 0.005, 0, -np.inf, -np.inf],
                [np.inf, centro_ref+0.25, 0.3, 1, np.inf, np.inf]
            ),
            maxfev=50000
        )

        A, xc1, fwhm_deg, eta, b0, b1 = popt
        err_fwhm = np.sqrt(max(pcov[2, 2], 0))
        err_xc = np.sqrt(max(pcov[1, 1], 0))

        theta1 = np.deg2rad(xc1/2)
        theta2 = np.arcsin(lambda_Ka2/lambda_Ka1*np.sin(theta1))
        xc2 = np.rad2deg(2*theta2)
        a_A = lambda_Ka1*np.sqrt(N)/(2*np.sin(theta1))

        resultados_LaB6_doblete.append({
            'Reflexión': hkl,
            'N': N,
            '2theta Ka1 (°)': xc1,
            '2theta Ka2 (°)': xc2,
            'FWHM común (°)': fwhm_deg,
            'Error FWHM (°)': err_fwhm,
            'eta': eta,
            'a (Å)': a_A,
            'Error centro (°)': err_xc
        })

        xfit = np.linspace(xp.min(), xp.max(), 400)
        ax.plot(xfit, doblete_pvoigt_fondo(xfit, *popt), lw=1.2,
                label=f'{hkl}: FWHM={fwhm_deg:.4f}°')

    except (ValueError, RuntimeError) as e:
        print(f'{hkl}: no se pudo ajustar ({e})')

ax.set_xlim(20, 85)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('LaB$_6$: ajuste de doblete Cu Kα')
ax.grid()
ax.legend(ncol=2, fontsize=8)
plt.show()

#%% Tabla de resultados del doblete
tabla_LaB6_doblete = pd.DataFrame(resultados_LaB6_doblete)
print(tabla_LaB6_doblete.to_string(index=False, formatters={
    '2theta Ka1 (°)': lambda v: f'{v:.4f}',
    '2theta Ka2 (°)': lambda v: f'{v:.4f}',
    'FWHM común (°)': lambda v: f'{v:.4f}',
    'Error FWHM (°)': lambda v: f'{v:.4f}',
    'eta': lambda v: f'{v:.3f}',
    'a (Å)': lambda v: f'{v:.6f}',
    'Error centro (°)': lambda v: f'{v:.4f}'
}))

#%% Ajuste pseudo-Voigt de los picos del patrón LaB6
# resultados_LaB6 = []
# fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
# ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.7', lw=0.8, label='LaB$_6$')

# for i, centro_ref, intensidad in picos_LaB6:
#     try:
#         xp, popt, pcov = analizar_pico(
#             dostheta_LaB6,
#             intensity_LaB6,
#             centro_ref,
#             ventana=0.9,
#             picos_detectados=picos_LaB6
#         )
#         A, xc, fwhm_deg, eta, b0, b1 = popt
#         fwhm_err_deg = np.sqrt(max(pcov[2, 2], 0))
#         xc_err_deg = np.sqrt(max(pcov[1, 1], 0))

#         resultados_LaB6.append({
#             'Índice': i,
#             '2theta detectado (°)': centro_ref,
#             '2theta ajustado (°)': xc,
#             'Error 2theta (°)': xc_err_deg,
#             'FWHM (°)': fwhm_deg,
#             'Error FWHM (°)': fwhm_err_deg,
#             'eta': eta
#         })

#         xfit = np.linspace(xp.min(), xp.max(), 300)
#         ax.plot(xfit, pvoigt_fondo(xfit, *popt), lw=1.2,
#                 label=f'{xc:.2f}°; FWHM={fwhm_deg:.4f}°')
#     except (ValueError, RuntimeError) as e:
#         print(f'Pico en {centro_ref:.2f}°: no se pudo ajustar ({e})')

# ax.set_xlim(20, 100)
# ax.set_ylim(0, None)
# ax.set_xlabel(r'$2\theta$ (°)')
# ax.set_ylabel('Intensidad (u.a.)')
# ax.set_title('LaB$_6$: ajuste pseudo-Voigt')
# ax.grid()
# ax.legend(ncol=2, fontsize=8, frameon=True)
# plt.show()

# #%% Tabla de anchuras del patrón LaB6
# tabla_LaB6 = pd.DataFrame(resultados_LaB6)
# formatos_LaB6 = {
#     'Índice': lambda v: f'{v:^7.0f}|',
#     '2theta detectado (°)': lambda v: f'{v:^20.4f}|',
#     '2theta ajustado (°)': lambda v: f'{v:^20.4f}|',
#     'Error 2theta (°)': lambda v: f'{v:^17.4f}|',
#     'FWHM (°)': lambda v: f'{v:^10.4f}|',
#     'Error FWHM (°)': lambda v: f'{v:^15.4f}|',
#     'eta': lambda v: f'{v:^7.3f}|'
# }
# print(tabla_LaB6.to_string(index=False, formatters=formatos_LaB6))


#%% Importo Data
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
# fig,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(10,8), constrained_layout=True,sharex=True)
# ax.plot(dostheta_M1, intensity_M1/max(intensity_M1)+0.0,lw=0.9, label=labels[0],zorder=1)
# ax.plot(dostheta_M2, intensity_M2/max(intensity_M2)+0.2,lw=0.9, label=labels[1],zorder=2)
# ax.plot(dostheta_M3, intensity_M3/max(intensity_M3)+0.4,lw=0.9, label=labels[2],zorder=3)    
# ax.plot(dostheta_M4, intensity_M4/max(intensity_M4)+0.6,lw=0.9, label=labels[3],zorder=1)

# ax2.plot(dostheta_M5, intensity_M5/max(intensity_M5)+0.0,lw=0.9, label=labels[4],zorder=2)
# ax2.plot(dostheta_M6, intensity_M6/max(intensity_M6)+0.2,lw=0.9, label=labels[5],zorder=3)    
# ax2.plot(dostheta_M7, intensity_M7/max(intensity_M7)+0.4,lw=0.9, label=labels[6],zorder=1)
# ax2.plot(dostheta_M9, intensity_M9/max(intensity_M9)+0.6,lw=0.9, label=labels[7],zorder=2)

# ax3.plot(dostheta_M10, intensity_M10/max(intensity_M10)+0.4,lw=0.9, label=labels[8],zorder=3)    
# ax3.plot(dostheta_M11, intensity_M11/max(intensity_M11)+0.0,lw=0.9, label=labels[9],zorder=1)
# ax3.plot(dostheta_M12, intensity_M12/max(intensity_M12)+0.2,lw=0.9, label=labels[10],zorder=2)


# for a in (ax,ax2,ax3):
#     a.legend(ncol=2,frameon=True,shadow=True)
#     a.grid()            
#     a.set_ylabel('Intensidad (u.a.)')
#     a.set_ylim(0,)
# ax3.set_xlim(20,80)
# ax3.set_xlabel('2θ (°)')
# plt.suptitle('DRX en polvo')
# plt.savefig('DRX_M1_M2_M3_M4_M5_M6_M7_M9_M10_M11_M12.png')
# plt.show()

#%% Analisis de picos M1 
peaks_M1, props_M1 = find_peaks(intensity_M1,prominence=25,width=2,distance=50)

fig11,ax=plt.subplots(figsize=(10,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M1, intensity_M1,'-', label='raw')
ax.plot(dostheta_M1[peaks_M1], intensity_M1[peaks_M1], 'r.', label='peaks')

ax.set_xlim(20,80)
ax.set_ylim(0,)
ax.grid()
ax.legend()
plt.savefig('DRX_M1_peaks.png')
plt.show()

picos_M1 = list(zip(peaks_M1, dostheta_M1[peaks_M1], intensity_M1[peaks_M1]))
print('M1',f'{len(peaks_M1)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M1:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')

#% Ajuste individual de cada reflexión
resultados_M1 = []

fig12, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_M1, intensity_M1,'-', color='0.4', lw=0.9, label='M1')

for hkl, centro_ref in reflexiones.items():
    try:
        xp, popt, pcov = analizar_pico(dostheta_M1, intensity_M1, centro_ref, picos_detectados=picos_M1)
        A, xc, fwhm_deg, eta, b0, b1 = popt

        # Scherrer: beta debe estar en radianes de 2theta
        theta_rad = np.deg2rad(xc/2)
        beta_obs = np.deg2rad(fwhm_deg)

        # Sin patrón de referencia, usamos la anchura observada
        beta = beta_obs
        D_nm = K*lambda_X/(beta*np.cos(theta_rad))/10

        # Incertidumbre estadística aproximada de la FWHM
        fwhm_err_deg = np.sqrt(max(pcov[2, 2], 0))
        beta_err = np.deg2rad(fwhm_err_deg)
        D_err_nm = D_nm*beta_err/beta

        resultados_M1.append({
            'Reflexión': hkl,
            '2theta teo (°)': centro_ref,
            '2theta (°)': xc,
            'FWHM (°)': fwhm_deg,
            'eta': eta,
            'D aparente (nm)': D_nm,
            'Error ajuste (nm)': D_err_nm
        })

        # Curva ajustada en la ventana del pico
        xfit = np.linspace(xp.min(), xp.max(), 300)
        ax.plot(xfit, pvoigt_fondo(xfit, *popt), lw=1.2,label=f'{hkl}: {xc:.2f}°')

    except (ValueError, RuntimeError) as e:
        print(f'{hkl}: no se pudo ajustar ({e})')

ax.set_xlim(20, 80)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M1: ajuste pseudo-Voigt en reflexiones de magnetita')
ax.grid()
ax.legend(ncol=1,frameon=True,shadow=True)
plt.show()

#% Tabla de resultados
tabla_M1 = pd.DataFrame(resultados_M1)
formatos = {
    'Reflexión': lambda v: f'{v:^9s}|',
    '2theta teo (°)': lambda v: f'{v:^14.2f}|',
    '2theta (°)': lambda v: f'{v:^10.4f}|',
    'FWHM (°)': lambda v: f'{v:^8.4f}|',
    'eta': lambda v: f'{v:^5.3f}|',
    'D aparente (nm)': lambda v: f'{v:^15.2f}|',
    'Error ajuste (nm)': lambda v: f'{v:^17.2f}|'
    }

print(tabla_M1.to_string(index=False, formatters=formatos))

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

picos_M2 = list(zip(peaks_M2, dostheta_M2[peaks_M2], intensity_M2[peaks_M2]))
print('M2',f'{len(peaks_M2)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M2:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#% Ajuste individual de cada reflexión M2
resultados_M2 = []
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_M2, intensity_M2, '-', color='0.4', lw=0.9, label='M2')
for hkl, centro_ref in reflexiones.items():
    try:
        xp, popt, pcov = analizar_pico(dostheta_M2, intensity_M2, centro_ref, picos_detectados=picos_M2)
        A, xc, fwhm_deg, eta, b0, b1 = popt
        theta_rad = np.deg2rad(xc/2)
        beta_obs = np.deg2rad(fwhm_deg)
        beta = beta_obs
        D_nm = K*lambda_X/(beta*np.cos(theta_rad))/10
        fwhm_err_deg = np.sqrt(max(pcov[2, 2], 0))
        beta_err = np.deg2rad(fwhm_err_deg)
        D_err_nm = D_nm*beta_err/beta
        resultados_M2.append({
            'Reflexión': hkl,
            '2theta teo (°)': centro_ref,
            '2theta (°)': xc,
            'FWHM (°)': fwhm_deg,
            'eta': eta,
            'D aparente (nm)': D_nm,
            'Error ajuste (nm)': D_err_nm
        })
        xfit = np.linspace(xp.min(), xp.max(), 300)
        ax.plot(xfit, pvoigt_fondo(xfit, *popt), lw=1.2, label=f'{hkl}: {xc:.2f}°')
    except (ValueError, RuntimeError) as e:
        print(f'{hkl}: no se pudo ajustar ({e})')
ax.set_xlim(20, 80)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M2: ajuste pseudo-Voigt en reflexiones de magnetita')
ax.grid()
ax.legend(ncol=1, frameon=True, shadow=True)
plt.show()
tabla_M2 = pd.DataFrame(resultados_M2)
formatos = {
    'Reflexión': lambda v: f'{v:^9s}|',
    '2theta teo (°)': lambda v: f'{v:^14.2f}|',
    '2theta (°)': lambda v: f'{v:^10.4f}|',
    'FWHM (°)': lambda v: f'{v:^8.4f}|',
    'eta': lambda v: f'{v:^5.3f}|',
    'D aparente (nm)': lambda v: f'{v:^15.2f}|',
    'Error ajuste (nm)': lambda v: f'{v:^17.2f}|'
}
print(tabla_M2.to_string(index=False, formatters=formatos))

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
