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
lambda_Ka1 = 1.5406
lambda_Ka2 = 1.5444
#%% Parámetros para Análisis de Scherrer:
K = 0.9
lambda_X = 1.5406  # Å, Cu Kα
beta_inst_deg = None  # Sin patrón de referencia: no se corrige
D_LaB6 = 800  # nm, estimación aproximada del tamaño de dominio de LaB6
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
data_LaB6 = np.loadtxt('data_DRX/resultados_DRX/LAB6PHI2.ASC', skiprows=0)
dostheta_LaB6 = data_LaB6[:, 0]
intensity_LaB6 = data_LaB6[:, 1]
#%%% pseudovoigt_fondo
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

#%% Funciones
def posicion_Ka2(xc_Ka1, lambda_Ka1=lambda_Ka1, lambda_Ka2=lambda_Ka2):
    """Calcula 2θ de Kα2 a partir de 2θ de Kα1 mediante la ley de Bragg."""
    theta1 = np.deg2rad(xc_Ka1/2)
    argumento = (lambda_Ka2/lambda_Ka1)*np.sin(theta1)
    if abs(argumento) > 1:
        return np.nan
    return np.rad2deg(2*np.arcsin(argumento))

def perfil_pvoigt(x, xc, fwhm, eta):
    """Perfil pseudo-Voigt normalizado a amplitud unitaria en el máximo."""
    z = (x-xc)/fwhm
    gauss = np.exp(-4*np.log(2)*z**2)
    lorentz = 1/(1+4*z**2)
    return (1-eta)*gauss + eta*lorentz

def doblete_pvoigt_fondo(x, A, xc, fwhm_Ka1, fwhm_Ka2, eta, rel_int_Ka2, b0, b1):
    """Doblete Cu Kα con centros ligados por Bragg, FWHM independientes y relación de intensidades libre."""
    xc2 = posicion_Ka2(xc)
    fondo = b0 + b1*(x-xc)
    perfil1 = perfil_pvoigt(x, xc, fwhm_Ka1, eta)
    perfil2 = perfil_pvoigt(x, xc2, fwhm_Ka2, eta)
    return fondo + A*(perfil1 + rel_int_Ka2*perfil2)

def ajustar_doblete(x, y, centro_ref, ventana=0.9):
    """Ajusta un doblete Cu Kα y devuelve datos, parámetros, covarianza y modelo."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mascara = np.isfinite(x) & np.isfinite(y) & (x >= centro_ref-ventana/2) & (x <= centro_ref+ventana/2)
    xp, yp = x[mascara], y[mascara]
    if len(xp) < 10:
        raise ValueError(f"No hay suficientes puntos alrededor de {centro_ref:.2f}°.")
    fondo0 = np.percentile(yp, 10)
    A0 = max(np.max(yp)-fondo0, np.finfo(float).eps)
    p0 = [A0, centro_ref, 0.08, 0.08, 0.5, 0.5, fondo0, 0]
    limites_inf = [0, centro_ref-ventana/3, 0.003, 0.003, 0, 0, -np.inf, -np.inf]
    limites_sup=[np.inf, centro_ref+ventana/3, ventana/2, ventana/2, 1, 1.0, np.inf, np.inf]
    popt, pcov = curve_fit(
        doblete_pvoigt_fondo, xp, yp, p0=p0,
        bounds=(limites_inf, limites_sup), maxfev=100000
    )
    return xp, yp, popt, pcov, doblete_pvoigt_fondo

def scherrer_D(fwhm_deg, xc_deg, K=K, lambda_X=lambda_X):
    """Calcula D en nm a partir del FWHM en grados de 2θ."""
    beta = np.deg2rad(fwhm_deg)
    theta = np.deg2rad(xc_deg/2)
    if not np.isfinite(beta) or beta <= 0:
        return np.nan
    return K*(lambda_X/10)/(beta*np.cos(theta))

# Visualización del difractograma
fig_LaB6a, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
ax.set_xlim(20,85)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('Patrón de calibración LaB$_6$')
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
plt.show()

#% Detección de picos del patrón LaB6
prominencia_LaB6 = 0.01*np.ptp(intensity_LaB6)
peaks_LaB6, props_LaB6 = find_peaks(
    intensity_LaB6,prominence=prominencia_LaB6,
    width=2,    distance=5)

picos_LaB6 = list(zip(peaks_LaB6,dostheta_LaB6[peaks_LaB6],intensity_LaB6[peaks_LaB6]))

fig_LaB6b, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
ax.plot(dostheta_LaB6, intensity_LaB6, '-', color='0.3', lw=0.9, label='LaB$_6$')
ax.plot(dostheta_LaB6[peaks_LaB6], intensity_LaB6[peaks_LaB6], 'r.', ms=6, label='Picos detectados')
ax.set_xlim(20, 85)
ax.set_ylim(0, None)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('Detección de máximos del patrón LaB$_6$')
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
plt.show()

print(f'LaB6: {len(peaks_LaB6)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i, theta, intens in picos_LaB6:
    print(f'{i:4} | {theta:7.2f} | {intens:10.0f}')

#%% Ajuste del doblete Cu Kα para LaB6
resultados_LaB6 = []
ajustes_LaB6 = {}

rel_int_Ka2 = 0.5
a_ref_LaB6 = 4.156826  # Å, NIST SRM 660c; usar solo si corresponde al patrón

# Reflexiones: índice, N=h²+k²+l² y máximo principal detectado (Kalpha1)
reflexiones_LaB6 = [
    ('(100)', 1, 21.34),
    ('(110)', 2, 30.38),
    ('(111)', 3, 37.42),
    ('(200)', 4, 43.50),
    ('(210)', 5, 48.94),
    ('(211)', 6, 53.96),
    ('(220)', 8, 63.20),
    ('(300)/(221)', 9, 67.54),
    ('(310)', 10, 71.72),
    ('(311)', 11, 75.82),
    ('(222)', 12, 79.84),
    ('(320)', 13, 83.82)
]

for hkl, N, centro_ref in reflexiones_LaB6:
    try:
        xp, yp, popt, pcov, modelo = ajustar_doblete(
            dostheta_LaB6, intensity_LaB6, centro_ref, ventana=1.1
        )
        errores = np.sqrt(np.maximum(np.diag(pcov), 0))
        A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
        err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
        theta1 = np.deg2rad(xc/2)
        a = lambda_Ka1*np.sqrt(N)/(2*np.sin(theta1))
        err_a = abs(a/np.tan(theta1))*np.deg2rad(err_xc/2)
        resultados_LaB6.append({
            'Reflexión': hkl,
            'N': N,
            '2theta Ka1 (°)': xc,
            '2theta Ka2 (°)': posicion_Ka2(xc),
            'FWHM Ka1 (°)': fwhm1,
            'Error FWHM Ka1 (°)': err_fwhm1,
            'FWHM Ka2 (°)': fwhm2,
            'Error FWHM Ka2 (°)': err_fwhm2,
            'eta': eta,
            'Relación intensidades Ka2/Ka1': ratio,
            'a (Å)': a,
            'Error centro (°)': err_xc,
            'Error a (Å)': err_a
        })
        ajustes_LaB6[hkl] = (xp, yp, popt, pcov)
    except (RuntimeError, ValueError) as e:
        print(f'{hkl}: falló el ajuste ({e})')

df_LaB6 = pd.DataFrame(resultados_LaB6).sort_values('2theta Ka1 (°)').reset_index(drop=True)
display(df_LaB6.round(6))
#%% Gráficos de los ajustes del doblete LaB6
fig_LaB6c, axs = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
axs = axs.ravel()

for ax, (hkl, N, centro_ref) in zip(axs, reflexiones_LaB6):
    if hkl not in ajustes_LaB6:
        ax.set_visible(False)
        continue
    xp, yp, popt, pcov = ajustes_LaB6[hkl]
    A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
    xfit = np.linspace(xp.min(), xp.max(), 1000)
    ax.plot(xp, yp, '.', ms=2, color='0.35', label='Datos')
    ax.plot(xfit, doblete_pvoigt_fondo(xfit, *popt), '-', lw=1.3, label='Ajuste total')
    ax.axvline(xc, color='tab:red', ls='--', lw=0.8, label=r'$K\alpha_1$')
    ax.axvline(posicion_Ka2(xc), color='tab:blue', ls=':', lw=0.8, label=r'$K\alpha_2$')
    ax.set_title(hkl)
    ax.set_xlabel(r'$2\theta$ (°)')
    ax.grid(alpha=0.3)

axs[0].set_ylabel('Intensidad (u.a.)')
axs[4].set_ylabel('Intensidad (u.a.)')
axs[8].set_ylabel('Intensidad (u.a.)')
axs[0].legend(fontsize=8)
for ax in axs[len(reflexiones_LaB6):]:
    ax.set_visible(False)
plt.show()

#%% Perfil instrumental estimado a partir de LaB6
fig_Lab6_perfil, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax.errorbar(
    df_LaB6['2theta Ka1 (°)'],
    df_LaB6['FWHM Ka1 (°)'],
    yerr=df_LaB6['Error FWHM Ka1 (°)'],
    fmt='o', capsize=3, label=r'LaB$_6$, $K\alpha_1$'
)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('FWHM (°)')
ax.set_title(r'Ancho de $K\alpha_1$ para LaB$_6$')
ax.grid()
ax.legend()
plt.show()

print(f'a promedio = {df_LaB6["a (Å)"].mean():.6f} Å')
print(f'Desviación estándar entre reflexiones = {df_LaB6["a (Å)"].std(ddof=1):.6f} Å')
print(f'Referencia NIST SRM 660c = {a_ref_LaB6:.6f} Å')
print(f'Diferencia relativa = {(df_LaB6["a (Å)"].mean()/a_ref_LaB6-1)*100:.4f} %')


#%% Corrección instrumental aproximada y tamaño de cristalita
def fwhm_voigt_aproximado(beta_G, beta_L):
    """Calcula el FWHM de un perfil Voigt mediante la aproximación de Olivero-Longbothum."""
    return 0.5346*beta_L + np.sqrt(0.2166*beta_L**2 + beta_G**2)

def interpolar_parametro_LaB6(xc, df_LaB6, columna):
    """Interpola un parámetro de LaB6 en la posición 2theta solicitada."""
    df = df_LaB6.sort_values('2theta Ka1 (°)')
    x = df['2theta Ka1 (°)'].to_numpy(dtype=float)
    y = df[columna].to_numpy(dtype=float)
    if len(x) < 2 or xc < x.min() or xc > x.max():
        raise ValueError(f'La posición {xc:.3f}° está fuera del intervalo angular de LaB6.')
    return np.interp(xc, x, y)

def corregir_fwhm_instrumental(fwhm_muestra, eta_muestra, xc, df_LaB6, D_LaB6=D_LaB6, K=K, lambda_X=lambda_X):
    """
    Corrige aproximadamente el FWHM de Kα1 usando LaB6 como referencia.
    Se supone que el ensanchamiento por tamaño de dominio de LaB6 es lorentziano.
    Devuelve el FWHM corregido en grados de 2theta.
    """
    fwhm_ref = interpolar_parametro_LaB6(xc, df_LaB6, 'FWHM Ka1 (°)')
    eta_ref = interpolar_parametro_LaB6(xc, df_LaB6, 'eta')
    theta = np.deg2rad(xc/2)

    # Componentes aproximadas del ancho medido de LaB6
    betaG_ref = fwhm_ref*np.sqrt(max(1-eta_ref, 0))
    betaL_ref = fwhm_ref*eta_ref

    # Ensanchamiento por tamaño de dominio de LaB6 (en grados de 2theta)
    beta_size_ref = np.rad2deg(K*(lambda_X/10)/(D_LaB6*np.cos(theta)))

    # Estimación de las contribuciones instrumentales
    betaG_inst = betaG_ref
    betaL_inst = max(betaL_ref-beta_size_ref, 0)

    # Componentes aproximadas del ancho medido de la muestra
    betaG_muestra = fwhm_muestra*np.sqrt(max(1-eta_muestra, 0))
    betaL_muestra = fwhm_muestra*eta_muestra

    # Deconvolución aproximada, componente por componente
    betaG_corr = np.sqrt(max(betaG_muestra**2-betaG_inst**2, 0))
    betaL_corr = max(betaL_muestra-betaL_inst, 0)

    if betaG_muestra <= betaG_inst and betaL_muestra <= betaL_inst:
        return np.nan

    fwhm_corr = fwhm_voigt_aproximado(betaG_corr, betaL_corr)
    return fwhm_corr if np.isfinite(fwhm_corr) and fwhm_corr > 0 else np.nan

#%% Análisis de la reflexión (311) de magnetita
def analizar_muestra_311(nombre, x, y, df_LaB6, centro_ref=35.5, ventana=1.1, graficar=True):
    """Ajusta el doblete Cu Kα de magnetita (311) y estima el tamaño de cristalita con Kα1."""
    xp, yp, popt, pcov, modelo = ajustar_doblete(x, y, centro_ref, ventana=ventana)
    errores = np.sqrt(np.maximum(np.diag(pcov), 0))
    A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
    err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
    xc2 = posicion_Ka2(xc)

    fwhm_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
    D_sin = scherrer_D(fwhm1, xc)
    D_con = scherrer_D(fwhm_corr, xc) if np.isfinite(fwhm_corr) else np.nan

    print(f'{nombre} - Reflexión (311)')
    print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
    print(f'2θ Kα₂ = {xc2:.5f}°')
    print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
    print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
    print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')
    print(f'D sin corrección = {D_sin:.2f} nm')
    print(f'FWHM Kα₁ corregido = {fwhm_corr:.6f}°')
    print(f'D corregido = {D_con:.2f} nm')

    tabla = pd.DataFrame([{
        'Muestra': nombre,
        '2theta Ka1 (°)': xc,
        'Error centro (°)': err_xc,
        '2theta Ka2 (°)': xc2,
        'FWHM Ka1 (°)': fwhm1,
        'Error FWHM Ka1 (°)': err_fwhm1,
        'FWHM Ka2 (°)': fwhm2,
        'Error FWHM Ka2 (°)': err_fwhm2,
        'eta': eta,
        'Relación intensidades Ka2/Ka1': ratio,
        'FWHM Ka1 corregido (°)': fwhm_corr,
        'D sin corrección (nm)': D_sin,
        'D corregido (nm)': D_con
    }])

    fig = None
    if graficar:
        xfit = np.linspace(xp.min(), xp.max(), 1500)
        xc2_fit = posicion_Ka2(xc)
        fondo = b0 + b1*(xfit-xc)
        perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
        perfil2 = A*ratio*perfil_pvoigt(xfit, xc2_fit, fwhm2, eta)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
        ax.plot(xfit, fondo+perfil1+perfil2, 'k-', lw=1.5, label='Ajuste total')
        ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
        ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
        ax.axvline(xc, color='tab:blue', ls='--', alpha=1.2)
        ax.axvline(xc2_fit, color='tab:orange', ls=':', alpha=1.2)
        ax.set_xlabel(r'$2\theta$ (°)')
        ax.set_ylabel('Intensidad (u.a.)')
        ax.set_title(f'{nombre}: magnetita (311)')
        ax.grid(alpha=0.25)
        ax.legend()
        plt.show()

    return tabla, fig, popt, pcov


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
#%% ploteo todos 
fig_all,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(11,8), constrained_layout=True,sharex=True)
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
    a.legend(ncol=1,bbox_to_anchor=(1,1),frameon=True,shadow=True)
    a.grid(axis='x')            
    a.set_ylabel('Intensidad (u.a.)')
    a.set_ylim(0,)
    for hkl, posicion in reflexiones.items():
        a.axvline(x=posicion, color='k', ls='--',alpha=0.8,lw=0.8)
        a.text(posicion,0.95,hkl,rotation=0,ha='right',va='top',
               transform=a.get_xaxis_transform(),fontsize=8)
    
    ax3.set_xlim(20,80)
ax3.set_xlabel('2θ (°)')
plt.suptitle('DRX en polvo')
plt.show()

resultados_cristalita = []


#%% M1
peaks_M1, props_M1 = find_peaks(intensity_M1, prominence=25, width=2, distance=50)
picos_M1 = list(zip(peaks_M1, dostheta_M1[peaks_M1], intensity_M1[peaks_M1]))

fig_M1a, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
ax.plot(dostheta_M1, intensity_M1, '-', label='raw')
ax.plot(dostheta_M1[peaks_M1], intensity_M1[peaks_M1], 'r.', label='peaks')

for i, theta, intens in picos_M1:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M1 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
plt.show()

print('M1',f'{len(peaks_M1)} picos detectados\n')
print('indx | 2θ (°) | Intensidad')
for i,theta,intens in picos_M1:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%% M1 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M1
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M1: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M1 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M1, intensity_M1, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM1 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M1 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M1b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=0.7)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=0.7)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M1 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M1_sin = scherrer_D(fwhm1, xc)
fwhm_M1_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M1_corr = scherrer_D(fwhm_M1_corr, xc) if np.isfinite(fwhm_M1_corr) else np.nan

print(f'\nM1 - Scherrer')
print(f'D sin corrección instrumental = {D_M1_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M1_corr:.6f}°')
print(f'D con corrección instrumental = {D_M1_corr:.2f} nm')

#%% M2
peaks_M2, props_M2 = find_peaks(intensity_M2,prominence=25,width=2,distance=30)
picos_M2 = list(zip(peaks_M2, dostheta_M2[peaks_M2], intensity_M2[peaks_M2]))
fig_M2a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M2, intensity_M2,'-', label='raw')
ax.plot(dostheta_M2[peaks_M2], intensity_M2[peaks_M2], 'r.', label='peaks')
for i, theta, intens in picos_M2:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M2 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

plt.show()

print('M2',f'{len(peaks_M2)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M2:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')

#%%#%% M2 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M2
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M2: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M2 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M2, intensity_M2, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM2 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M2 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M2b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M2 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M2_sin = scherrer_D(fwhm1, xc)
fwhm_M2_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M2_corr = scherrer_D(fwhm_M2_corr, xc) if np.isfinite(fwhm_M2_corr) else np.nan

print(f'\nM2 - Scherrer')
print(f'D sin corrección instrumental = {D_M2_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M2_corr:.6f}°')
print(f'D con corrección instrumental = {D_M2_corr:.2f} nm')


#%% M3 - Detección de picos
peaks_M3, props_M3 = find_peaks(intensity_M3, prominence=24, width=2, distance=30)
picos_M3 = list(zip(peaks_M3, dostheta_M3[peaks_M3], intensity_M3[peaks_M3]))

fig_M3a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M3, intensity_M3,'-', label='raw')
ax.plot(dostheta_M3[peaks_M3], intensity_M3[peaks_M3], 'r.', label='peaks')

for i, theta, intens in picos_M3:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M3 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

plt.show()

print('M3',f'{len(peaks_M3)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M3:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')

#%% M3 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M3
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M3: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M3 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M3, intensity_M3, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM3 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M3 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M3b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M3 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M3_sin = scherrer_D(fwhm1, xc)
fwhm_M3_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M3_corr = scherrer_D(fwhm_M3_corr, xc) if np.isfinite(fwhm_M3_corr) else np.nan

print(f'\nM3 - Scherrer')
print(f'D sin corrección instrumental = {D_M3_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M3_corr:.6f}°')
print(f'D con corrección instrumental = {D_M3_corr:.2f} nm')


#%% M4 
peaks_M4, props_M4 = find_peaks(intensity_M4, prominence=22, width=8, distance=20)
picos_M4 = list(zip(peaks_M4, dostheta_M4[peaks_M4], intensity_M4[peaks_M4]))

fig_M4a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M4, intensity_M4,'-', label='raw')
ax.plot(dostheta_M4[peaks_M4], intensity_M4[peaks_M4], 'r.', label='peaks')
for i, theta, intens in picos_M4:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)
ax.set_xlim(20,80)
ax.set_title('M4 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M4',f'{len(peaks_M4)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M4:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%% M4 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M4
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M4: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M4 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M4, intensity_M4, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM4 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M4 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M4b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M4 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M4_sin = scherrer_D(fwhm1, xc)
fwhm_M4_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M4_corr = scherrer_D(fwhm_M4_corr, xc) if np.isfinite(fwhm_M4_corr) else np.nan

print(f'\nM4 - Scherrer')
print(f'D sin corrección instrumental = {D_M4_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M4_corr:.6f}°')
print(f'D con corrección instrumental = {D_M4_corr:.2f} nm')
#%% M5 
peaks_M5, props_M5 = find_peaks(intensity_M5,prominence=6,
                                width=7,
                                distance=20)
picos_M5 = list(zip(peaks_M5, dostheta_M5[peaks_M5], intensity_M5[peaks_M5]))

fig_M5a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M5, intensity_M5,'-', label='raw')
ax.plot(dostheta_M5[peaks_M5], intensity_M5[peaks_M5], 'r.', label='peaks')
for i, theta, intens in picos_M5:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M5 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M5',f'{len(peaks_M5)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M5:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')       
#%% M5 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M5
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M5: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M5 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M5, intensity_M5, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM5 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M5 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M5b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M5 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M5_sin = scherrer_D(fwhm1, xc)
fwhm_M5_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M5_corr = scherrer_D(fwhm_M5_corr, xc) if np.isfinite(fwhm_M5_corr) else np.nan

print(f'\nM5 - Scherrer')
print(f'D sin corrección instrumental = {D_M5_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M5_corr:.6f}°')
print(f'D con corrección instrumental = {D_M5_corr:.2f} nm')
#%% M6 
peaks_M6, props_M6 = find_peaks(intensity_M6,
                                prominence=15,
                                width=3,
                                distance=20)
picos_M6 = list(zip(peaks_M6, dostheta_M6[peaks_M6], intensity_M6[peaks_M6]))

fig_M6a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M6, intensity_M6,'-', label='raw')
ax.plot(dostheta_M6[peaks_M6], intensity_M6[peaks_M6], 'r.', label='peaks')
for i, theta, intens in picos_M6:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M6 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M6',f'{len(peaks_M6)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M6:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')   

#%% M6 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M6
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M6: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M6 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M6, intensity_M6, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM6 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M6 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M6b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M6 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M6_sin = scherrer_D(fwhm1, xc)
fwhm_M6_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M6_corr = scherrer_D(fwhm_M6_corr, xc) if np.isfinite(fwhm_M6_corr) else np.nan

print(f'\nM6 - Scherrer')
print(f'D sin corrección instrumental = {D_M6_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M6_corr:.6f}°')
print(f'D con corrección instrumental = {D_M6_corr:.2f} nm')
#%% M7 
peaks_M7, props_M7 = find_peaks(intensity_M7,
                                prominence=20,
                                width=7,
                                distance=20)
picos_M7 = list(zip(peaks_M7, dostheta_M7[peaks_M7], intensity_M7[peaks_M7]))

fig_M7a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M7, intensity_M7,'-', label='raw')
ax.plot(dostheta_M7[peaks_M7], intensity_M7[peaks_M7], 'r.', label='peaks')
for i, theta, intens in picos_M7:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M7 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M7',f'{len(peaks_M7)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M7:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')  

#%% M7 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M7
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M7: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M7 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M7, intensity_M7, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM7 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M7 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M7b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M7 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M7_sin = scherrer_D(fwhm1, xc)
fwhm_M7_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M7_corr = scherrer_D(fwhm_M7_corr, xc) if np.isfinite(fwhm_M7_corr) else np.nan

print(f'\nM7 - Scherrer')
print(f'D sin corrección instrumental = {D_M7_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M7_corr:.6f}°')
print(f'D con corrección instrumental = {D_M7_corr:.2f} nm')
#%% M9
peaks_M9, props_M9 = find_peaks(intensity_M9,
                                height=30,
                                prominence=10,
                                width=3,
                                distance=10)
picos_M9 = list(zip(peaks_M9, dostheta_M9[peaks_M9], intensity_M9[peaks_M9]))

fig_M9a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M9, intensity_M9,'-', label='raw')
ax.plot(dostheta_M9[peaks_M9], intensity_M9[peaks_M9], 'r.', label='peaks')
for i, theta, intens in picos_M9:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M9 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M9',f'{len(peaks_M9)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M9:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')       
#%% M9 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M9
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M9: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M9 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M9, intensity_M9, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM9 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M9 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M9b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M9 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M9_sin = scherrer_D(fwhm1, xc)
fwhm_M9_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M9_corr = scherrer_D(fwhm_M9_corr, xc) if np.isfinite(fwhm_M9_corr) else np.nan

print(f'\nM9 - Scherrer')
print(f'D sin corrección instrumental = {D_M9_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M9_corr:.6f}°')
print(f'D con corrección instrumental = {D_M9_corr:.2f} nm')
#%% M10 
peaks_M10, props_M10 = find_peaks(intensity_M10,
                                height=24,
                                prominence=17,
                                width=2,
                                distance=15)
picos_M10 = list(zip(peaks_M10, dostheta_M10[peaks_M10], intensity_M10[peaks_M10]))

fig_M10a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M10, intensity_M10,'-', label='raw')
ax.plot(dostheta_M10[peaks_M10], intensity_M10[peaks_M10], 'r.', label='peaks')
for i, theta, intens in picos_M10:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M10 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')

print('M10',f'{len(peaks_M10)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M10:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')    
#%% M10 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M10
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M10: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M10 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M10, intensity_M10, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM10 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M10 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M10b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M10 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M10_sin = scherrer_D(fwhm1, xc)
fwhm_M10_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M10_corr = scherrer_D(fwhm_M10_corr, xc) if np.isfinite(fwhm_M10_corr) else np.nan

print(f'\nM10 - Scherrer')
print(f'D sin corrección instrumental = {D_M10_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M10_corr:.6f}°')
print(f'D con corrección instrumental = {D_M10_corr:.2f} nm')

#%%  M11
peaks_M11, props_M11 = find_peaks(intensity_M11,
                                  height=28,
                                prominence=10,
                                width=4,
                                distance=20)

picos_M11 = list(zip(peaks_M11, dostheta_M11[peaks_M11], intensity_M11[peaks_M11]))

fig_M11a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M11, intensity_M11,'-', label='raw')
ax.plot(dostheta_M11[peaks_M11], intensity_M11[peaks_M11], 'r.', label='peaks')
for i, theta, intens in picos_M11:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M11 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')


print('M11',f'{len(peaks_M11)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M11:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%% M11 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M11
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M11: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M11 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M11, intensity_M11, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM11 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M11 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M11b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M11 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M11_sin = scherrer_D(fwhm1, xc)
fwhm_M11_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M11_corr = scherrer_D(fwhm_M11_corr, xc) if np.isfinite(fwhm_M11_corr) else np.nan

print(f'\nM11 - Scherrer')
print(f'D sin corrección instrumental = {D_M11_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M11_corr:.6f}°')
print(f'D con corrección instrumental = {D_M11_corr:.2f} nm')

#%% M12 
peaks_M12, props_M12 = find_peaks(intensity_M12,prominence=25,width=2,distance=50)
picos_M12 = list(zip(peaks_M12, dostheta_M12[peaks_M12], intensity_M12[peaks_M12]))

fig_M12a,ax=plt.subplots(figsize=(9,5), constrained_layout=True,sharex=True)
ax.plot(dostheta_M12, intensity_M12,'-', label='raw')
ax.plot(dostheta_M12[peaks_M12], intensity_M12[peaks_M12], 'r.', label='peaks')
for i, theta, intens in picos_M12:
    ax.annotate(f'{theta:.2f}°', (theta, intens),
                xytext=(0,5), textcoords='offset points',
                ha='center', va='bottom', rotation=0, fontsize=8)

ax.set_xlim(20,80)
ax.set_title('M12 - Difractograma de RX')
ax.set_ylim(0,)
ax.grid()
ax.legend()
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
print('M12',f'{len(peaks_M12)} picos detectados\n')
print('indx |  2θ (°) | Intensidad')
for i,theta,intens in picos_M12:
    print(f'{i:4} | {theta:7.2f} | {intens:7.0f}')
#%% M12 - Ajuste del doblete Cu Kα para la reflexión (311)
centro_ref = reflexiones['(311)']
ventana = 1.1

# Buscar el máximo detectado más cercano a la reflexión (311)
candidatos_311 = [
    (i, theta, intens) for i, theta, intens in picos_M12
    if abs(theta-centro_ref) <= 0.5
]

if not candidatos_311:
    raise ValueError('M12: no se detectó ningún máximo cerca de la reflexión (311). Revisá picos_M12 o la ventana.')

indice_pico, xc0, intens0 = min(candidatos_311, key=lambda p: abs(p[1]-centro_ref))
print(f'Máximo seleccionado para (311): índice={indice_pico}, 2θ={xc0:.4f}°, intensidad={intens0:.0f}')

# Ajustar el doblete alrededor del máximo detectado
xp, yp, popt, pcov, modelo = ajustar_doblete(
    dostheta_M12, intensity_M12, centro_ref=xc0, ventana=ventana
)

errores = np.sqrt(np.maximum(np.diag(pcov), 0))
A, xc, fwhm1, fwhm2, eta, ratio, b0, b1 = popt
err_xc, err_fwhm1, err_fwhm2 = errores[1], errores[2], errores[3]
xc2 = posicion_Ka2(xc)

print(f'\nM12 - Reflexión (311)')
print(f'2θ Kα₁ = {xc:.5f} ± {err_xc:.5f}°')
print(f'2θ Kα₂ = {xc2:.5f}°')
print(f'FWHM Kα₁ = {fwhm1:.6f} ± {err_fwhm1:.6f}°')
print(f'FWHM Kα₂ = {fwhm2:.6f} ± {err_fwhm2:.6f}°')
print(f'η = {eta:.4f}')
print(f'Relación de intensidades Kα₂/Kα₁ = {ratio:.4f}')

# Diagnóstico del ajuste
denominador = np.sqrt(pcov[2,2]*pcov[3,3])
correlacion = pcov[2,3]/denominador if denominador > 0 else np.nan

print(f'\nDiagnóstico del ajuste de M12 (311)')
print(f'Diferencia FWHM = {fwhm1-fwhm2:.10e}°')
print(f'Correlación entre FWHM1 y FWHM2 = {correlacion:.6f}')

# Gráfico del ajuste y de las componentes
xfit = np.linspace(xp.min(), xp.max(), 1500)
fondo = b0 + b1*(xfit-xc)
perfil1 = A*perfil_pvoigt(xfit, xc, fwhm1, eta)
perfil2 = A*ratio*perfil_pvoigt(xfit, xc2, fwhm2, eta)

fig_M12b, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.plot(xp, yp, '.', ms=3, color='0.35', label='Datos experimentales')
ax.plot(xfit, modelo(xfit, *popt), 'k-', lw=1.8, label='Ajuste total')
ax.plot(xfit, fondo+perfil1, '--', lw=1.2, label=fr'$K\alpha_1$: FWHM = {fwhm1:.4f}°')
ax.plot(xfit, fondo+perfil2, ':', lw=1.2, label=fr'$K\alpha_2$: FWHM = {fwhm2:.4f}°')
ax.axvline(xc, color='tab:blue', ls='--', alpha=1)
ax.axvline(xc2, color='tab:orange', ls=':', alpha=1)
ax.set_xlabel('2θ (°)')
ax.set_ylabel('Intensidad (u.a.)')
ax.set_title('M12 - Ajuste del doblete de magnetita (311)')
ax.grid(alpha=0.25)
ax.legend()
plt.show()

# Tamaño de cristalita por Scherrer usando únicamente Kα1
D_M12_sin = scherrer_D(fwhm1, xc)
fwhm_M12_corr = corregir_fwhm_instrumental(fwhm1, eta, xc, df_LaB6)
D_M12_corr = scherrer_D(fwhm_M12_corr, xc) if np.isfinite(fwhm_M12_corr) else np.nan

print(f'\nM12 - Scherrer')
print(f'D sin corrección instrumental = {D_M12_sin:.2f} nm')
print(f'FWHM Kα₁ corregido = {fwhm_M12_corr:.6f}°')
print(f'D con corrección instrumental = {D_M12_corr:.2f} nm')

# %% salvo todas las figuras
#%% Guardar figuras DRX

fig_LaB6a.savefig('data_DRX/resultados_DRX/LaB6_DRX.png', dpi=300)
fig_LaB6b.savefig('data_DRX/resultados_DRX/LaB6_DRX_c_picos.png', dpi=300)
fig_LaB6c.savefig('data_DRX/resultados_DRX/LaB6_perfil_exp.png', dpi=300)
fig_Lab6_perfil.savefig('data_DRX/resultados_DRX/LaB6_perfil.png', dpi=300)
fig_all.savefig('data_DRX/resultados_DRX/DRX_M1_M2_M3_M4_M5_M6_M7_M9_M10_M11_M12.png', dpi=300)

fig_M1a.savefig('data_DRX/resultados_DRX/M1_DRX.png', dpi=300)
fig_M1b.savefig('data_DRX/resultados_DRX/M1_DRX_c_ajustes.png', dpi=300)

fig_M2a.savefig('data_DRX/resultados_DRX/M2_DRX.png', dpi=300)
fig_M2b.savefig('data_DRX/resultados_DRX/M2_DRX_c_ajustes.png', dpi=300)

fig_M3a.savefig('data_DRX/resultados_DRX/M3_DRX.png', dpi=300)
fig_M3b.savefig('data_DRX/resultados_DRX/M3_DRX_c_ajustes.png', dpi=300)

fig_M4a.savefig('data_DRX/resultados_DRX/M4_DRX.png', dpi=300)
fig_M4b.savefig('data_DRX/resultados_DRX/M4_DRX_c_ajustes.png', dpi=300)

fig_M5a.savefig('data_DRX/resultados_DRX/M5_DRX.png', dpi=300)
fig_M5b.savefig('data_DRX/resultados_DRX/M5_DRX_c_ajustes.png', dpi=300)

fig_M6a.savefig('data_DRX/resultados_DRX/M6_DRX.png', dpi=300)
fig_M6b.savefig('data_DRX/resultados_DRX/M6_DRX_c_ajustes.png', dpi=300)

fig_M7a.savefig('data_DRX/resultados_DRX/M7_DRX.png', dpi=300)
fig_M7b.savefig('data_DRX/resultados_DRX/M7_DRX_c_ajustes.png', dpi=300)

fig_M9a.savefig('data_DRX/resultados_DRX/M9_DRX.png', dpi=300)
fig_M9b.savefig('data_DRX/resultados_DRX/M9_DRX_c_ajustes.png', dpi=300)

fig_M10a.savefig('data_DRX/resultados_DRX/M10_DRX.png', dpi=300)
fig_M10b.savefig('data_DRX/resultados_DRX/M10_DRX_c_ajustes.png', dpi=300)

fig_M11a.savefig('data_DRX/resultados_DRX/M11_DRX.png', dpi=300)
fig_M11b.savefig('data_DRX/resultados_DRX/M11_DRX_c_ajustes.png', dpi=300)

fig_M12a.savefig('data_DRX/resultados_DRX/M12_DRX.png', dpi=300)
fig_M12b.savefig('data_DRX/resultados_DRX/M12_DRX_c_ajustes.png', dpi=300)
#%%
#%% Comparación del tamaño de cristalita por muestra
#%% Comparación del tamaño de cristalita por muestra

#%% Comparación del tamaño de cristalita por muestra

#%% Comparación del tamaño de cristalita por muestra
#%% Comparación del tamaño de cristalita por muestra

nombres_muestras = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M9', 'M10', 'M11', 'M12']

D_sin = np.array([D_M1_sin, D_M2_sin, D_M3_sin, D_M4_sin, D_M5_sin, D_M6_sin, D_M7_sin, D_M9_sin, D_M10_sin, D_M11_sin, D_M12_sin])
D_corr = np.array([D_M1_corr, D_M2_corr, D_M3_corr, D_M4_corr, D_M5_corr, D_M6_corr, D_M7_corr, D_M9_corr, D_M10_corr, D_M11_corr, D_M12_corr])

x = np.arange(len(nombres_muestras))
ancho = 0.35

fig_comparacion_cristalitas, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

ax.bar(x-ancho/2, D_sin, width=ancho, color='tab:blue', label='Sin corrección instrumental')
ax.bar(x+ancho/2, D_corr, width=ancho, color='tab:orange', label='Con corrección instrumental')

ax.set_xticks(x)
ax.set_xticklabels(nombres_muestras)
ax.set_xlabel('Muestra')
ax.set_ylabel('Tamaño de cristalita (nm)')
ax.set_title('Tamaño de cristalita por muestra')
ax.grid(axis='y')
ax.set_axisbelow(True)
ax.legend()
plt.show()
fig_comparacion_cristalitas.savefig('data_DRX/resultados_DRX/comparacion_tamano_cristalita.png', dpi=300)
#%% Exportar resumen de tamaños de cristalita

df_cristalitas = pd.DataFrame(resultados_cristalita)

df_cristalitas.to_csv(
    'Resultados_cristalitas_Scherrer.txt',
    sep='\t',
    index=False,
    float_format='%.4f'
)

print('\nResultados de tamaños de cristalita:')
print(df_cristalitas.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
print('\nArchivo guardado: Resultados_cristalitas_Scherrer.txt')

# %%
#%% Tamaño medio de cristalita por muestra
muestras = ['M1','M2','M3','M4','M5','M6','M7','M9','M10','M11','M12']

resumenes_sin = [resumen_M1_sin,resumen_M2_sin,resumen_M3_sin,resumen_M4_sin,
                 resumen_M5_sin,resumen_M6_sin,resumen_M7_sin,resumen_M9_sin,
                 resumen_M10_sin,resumen_M11_sin,resumen_M12_sin]
resumenes_con = [resumen_M1_con,resumen_M2_con,resumen_M3_con,resumen_M4_con,
                 resumen_M5_con,resumen_M6_con,resumen_M7_con,resumen_M9_con,
                 resumen_M10_con,resumen_M11_con,resumen_M12_con]

D_sin = np.array([r['D medio (nm)'] for r in resumenes_sin])
std_sin = np.array([r['Desviación estándar (nm)'] for r in resumenes_sin])
D_con = np.array([r['D medio (nm)'] for r in resumenes_con])
std_con = np.array([r['Desviación estándar (nm)'] for r in resumenes_con])


#%% Promedio conjunto del tamaño de cristalita por grupos de muestras
resumenes_sin = {
    'M1': resumen_M1_sin, 'M2': resumen_M2_sin, 'M3': resumen_M3_sin,
    'M4': resumen_M4_sin, 'M5': resumen_M5_sin, 'M6': resumen_M6_sin,
    'M7': resumen_M7_sin, 'M9': resumen_M9_sin, 'M10': resumen_M10_sin,
    'M11': resumen_M11_sin, 'M12': resumen_M12_sin
}
resumenes_con = {
    'M1': resumen_M1_con, 'M2': resumen_M2_con, 'M3': resumen_M3_con,
    'M4': resumen_M4_con, 'M5': resumen_M5_con, 'M6': resumen_M6_con,
    'M7': resumen_M7_con, 'M9': resumen_M9_con, 'M10': resumen_M10_con,
    'M11': resumen_M11_con, 'M12': resumen_M12_con
}

grupos = {
    'M1-M10': ['M1','M2','M3','M4','M5','M6','M7','M9','M10'],
    'M11-M12': ['M11','M12']
}

def promedio_grupo(nombres, resumenes):
    D = np.array([resumenes[m]['D medio (nm)'] for m in nombres], dtype=float)
    err = np.array([resumenes[m]['Incertidumbre (nm)'] for m in nombres], dtype=float)
    validos = np.isfinite(D) & np.isfinite(err) & (err > 0) & (D > 0)
    D, err = D[validos], err[validos]
    if len(D) < 2:
        raise ValueError('Se necesitan al menos dos muestras válidas')
    pesos = 1/err**2
    D_medio = np.sum(pesos*D)/np.sum(pesos)
    err_estadistico = np.sqrt(1/np.sum(pesos))
    chi2_red = np.sum(((D-D_medio)/err)**2)/(len(D)-1)
    err_final = err_estadistico*np.sqrt(max(chi2_red, 1))
    return {
        'D medio (nm)': D_medio,
        'Incertidumbre (nm)': err_final,
        'Error ponderado sin inflar (nm)': err_estadistico,
        'Desv. estándar entre muestras (nm)': np.std(D, ddof=1),
        'chi2 reducido': chi2_red,
        'N muestras': len(D)
    }

resultados_grupos = []
for nombre_grupo, nombres in grupos.items():
    for correccion, resumenes in [('Sin corrección', resumenes_sin),
                                  ('Con corrección LaB6', resumenes_con)]:
        r = promedio_grupo(nombres, resumenes)
        resultados_grupos.append({
            'Grupo': nombre_grupo,
            'Corrección': correccion,
            **r
        })

tabla_grupos = pd.DataFrame(resultados_grupos)
display(tabla_grupos.round(2))
# %%

fig_DRX_resumen, ax = plt.subplots(figsize=(11,5), constrained_layout=True)
x = np.arange(len(muestras))

media_sin, desv_sin = np.mean(D_sin[:-2]), np.std(D_sin[:-2],ddof=1)
media_con, desv_con = np.mean(D_con[:-2]), np.std(D_con[:-2],ddof=1)

media2_sin, desv2_sin = np.mean(D_sin[-2:]), np.std(D_sin[-2:],ddof=1)
media2_con, desv2_con = np.mean(D_con[-2:]), np.std(D_con[-2:],ddof=1)

ax.errorbar(x-0.08,D_sin,yerr=std_sin,fmt='o',capsize=4,
            lw=1.2,ms=6,label='Sin corrección instrumental')
ax.errorbar(x+0.08,D_con,yerr=std_con,fmt='s',capsize=4,
            lw=1.2,ms=6,label='Con corrección LaB$_6$')

ax.axhspan(media_sin-desv_sin,media_sin+desv_sin,color='C0',alpha=0.2)
ax.axhline(media_sin,color='C0',ls='--',lw=1.5, label=f'Sin corrección: {media_sin:.1f} ± {desv_sin:.1f} nm')

ax.axhspan(media_con-desv_con,media_con+desv_con,color='C1',alpha=0.2)
ax.axhline(media_con,color='C1',ls='--',lw=1.5,label=f'Con corrección LaB$_6$: {media_con:.1f} ± {desv_con:.1f} nm')

ax.axhspan(media2_sin-desv_sin,media2_sin+desv_sin,color='C0',alpha=0.2)
ax.axhline(media2_sin,color='C0',ls='-.',lw=1.5,
           label=f'Sin corrección: {media2_sin:.1f} ± {desv_sin:.1f} nm')

ax.axhspan(media2_con-desv_con,media2_con+desv_con,color='C1',alpha=0.2)
ax.axhline(media2_con,color='C1',ls='-.',lw=1.5,
           label=f'Con corrección LaB$_6$: {media2_con:.1f} ± {desv_con:.1f} nm')


ax.set_xticks(x)
ax.set_xticklabels(muestras)
ax.set_xlabel('Muestra')
ax.set_ylabel('Tamaño medio de cristalita, $D$ (nm)')
ax.set_title('Tamaño medio de cristalita por DRX')
ax.set_ylim(0,None)
ax.grid(axis='y',alpha=0.3)
ax.legend(ncol=3)
plt.savefig('data_DRX/resultados_DRX/tamaño_vs_muestra.png',dpi=300)
plt.show()
#%% Williamson-Hall para todas las muestras
tablas_DRX = {
    'M1': tabla_M1, 'M2': tabla_M2, 'M3': tabla_M3,
    'M4': tabla_M4, 'M5': tabla_M5, 'M6': tabla_M6,
    'M7': tabla_M7, 'M9': tabla_M9, 'M10': tabla_M10,
    'M11': tabla_M11, 'M12': tabla_M12}

resultados_WH_sin = {}
resultados_WH_con = {}

for nombre, tabla in tablas_DRX.items():
    resultados_WH_sin[nombre] = williamson_hall(
        tabla, nombre, 'FWHM sin (°)', 'Error FWHM sin (°)')
    resultados_WH_con[nombre] = williamson_hall(
        tabla, nombre, 'FWHM con (°)', 'Error FWHM con (°)')
#%%
