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
def analizar_pico(x, y, centro_ref, ventana=0.9, picos_detectados=None,
                  correcion_LaB6=False, df_LaB6=None,
                  K=0.9, lambda_X=1.5406, D_LaB6=800):
    """
    Ajusta un pico mediante una pseudo-Voigt sobre fondo lineal y,
    opcionalmente, estima el FWHM corregido por ensanchamiento instrumental.

    Parámetros adicionales
    ----------------------
    correcion_LaB6 : bool
        Si True, estima y aplica la corrección instrumental a partir de LaB6.
    df_LaB6 : DataFrame
        Tabla de calibración LaB6, con columnas:
        '2theta Ka1 (°)', 'FWHM Ka1 (°)' y 'eta'.
    K : float
        Factor de Scherrer.
    lambda_X : float
        Longitud de onda en Å.
    D_LaB6 : float
        Tamaño de dominio de referencia de LaB6, en nm.

    Retorna
    -------
    xp : ndarray
        Posiciones angulares usadas en el ajuste.
    popt : ndarray
        Parámetros ajustados [A, xc, fwhm, eta, b0, b1].
    pcov : ndarray
        Matriz de covarianza del ajuste.
    fwhm_corregido : float
        FWHM corregido en grados de 2theta. Si no se activa la
        corrección, coincide con el FWHM ajustado.
    """
    mask = (x >= centro_ref-ventana) & (x <= centro_ref+ventana)
    xp, yp = x[mask], y[mask]
    if len(xp) < 10:
        raise ValueError('Pocos puntos para ajustar el pico')

    b0 = np.percentile(yp, 10)
    fwhm0 = max(3*np.median(np.diff(xp)), 0.15)

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

    fwhm = popt[2]
    fwhm_corregido = fwhm

    if correcion_LaB6:
        if df_LaB6 is None:
            raise ValueError('Debes pasar df_LaB6 para corregir el ancho')

        # Anchos e intensidades de mezcla de la muestra
        eta_muestra = np.clip(popt[3], 0, 1)
        theta = np.deg2rad(popt[1]/2)

        # Interpolación de la calibración en la posición del pico
        df_cal = df_LaB6.sort_values('2theta Ka1 (°)')
        x_cal = df_cal['2theta Ka1 (°)'].to_numpy()
        beta_cal = df_cal['FWHM Ka1 (°)'].to_numpy()
        eta_cal = df_cal['eta'].to_numpy()

        if not x_cal.min() <= popt[1] <= x_cal.max():
            raise ValueError(
                'El pico está fuera del intervalo angular calibrado por LaB6'
            )

        beta_inst_obs = np.interp(popt[1], x_cal, beta_cal)
        eta_inst = np.clip(np.interp(popt[1], x_cal, eta_cal), 0, 1)

        # Ensanchamiento intrínseco aproximado de LaB6:
        # beta_size = K*lambda/(D*cos(theta)), en radianes de 2theta.
        beta_LaB6_intr = np.rad2deg(
            K*(lambda_X/(D_LaB6*10))*1/np.cos(theta)
        )

        # Descomposición heurística de los FWHM pseudo-Voigt.
        # No es una descomposición exacta de la función instrumental.
        beta_G_inst = beta_inst_obs*np.sqrt(1-eta_inst)
        beta_L_inst_obs = beta_inst_obs*eta_inst
        beta_L_inst = max(beta_L_inst_obs-beta_LaB6_intr, 0)

        beta_G_muestra = fwhm*np.sqrt(1-eta_muestra)
        beta_L_muestra = fwhm*eta_muestra

        # Deconvolución aproximada: gaussiana en cuadratura,
        # lorentziana por resta lineal.
        beta_G_corr = np.sqrt(max(beta_G_muestra**2-beta_G_inst**2, 0))
        beta_L_corr = max(beta_L_muestra-beta_L_inst, 0)

        # FWHM equivalente de Voigt (aproximación de Thompson-Cox-Hastings)
        fwhm_corregido = (
            0.5346*beta_L_corr
            + np.sqrt(0.2166*beta_L_corr**2 + beta_G_corr**2)
        )

    return xp, popt, pcov, fwhm_corregido
#%% promedio_cristalita
def promedio_cristalita(tabla, columna_D, columna_err):
    datos = tabla[[columna_D, columna_err]].dropna()
    datos = datos[
        np.isfinite(datos[columna_D]) &
        np.isfinite(datos[columna_err]) &
        (datos[columna_err] > 0) &
        (datos[columna_D] > 0)
    ]

    D = datos[columna_D].to_numpy()
    err = datos[columna_err].to_numpy()

    if len(D) < 2:
        raise ValueError('Se necesitan al menos dos reflexiones válidas')

    pesos = 1/err**2
    D_media = np.sum(pesos*D)/np.sum(pesos)
    err_media = np.sqrt(1/np.sum(pesos))

    chi2_red = np.sum(((D-D_media)/err)**2)/(len(D)-1)

    # Inflar la incertidumbre si la dispersión supera la esperada
    err_final = err_media*np.sqrt(max(chi2_red, 1))

    return {
        'D medio (nm)': D_media,
        'Incertidumbre (nm)': err_final,
        'Desviación estándar (nm)': np.std(D, ddof=1),
        'chi2 reducido': chi2_red,
        'N reflexiones': len(D)}

#%% Función general de análisis DRX por muestra
def analizar_muestra(nombre, dostheta, intensity, picos_detectados):
    picos = picos_detectados
    # A partir de acá, mantené el tratamiento de ajuste,
    # cálculo de Scherrer, tabla y gráficos de la función anterior.
    
    # Ajuste individual de las reflexiones
    resultados = []
    fig, ax = plt.subplots(figsize=(10,5), constrained_layout=True)
    ax.plot(dostheta, intensity, '-', color='0.4', lw=0.9, label=nombre)

    for hkl, centro_ref in reflexiones.items():
        if not any(abs(theta-centro_ref) <= 0.9/2
               for _, theta, intens in picos_detectados):
            continue
        try:
            # Ajuste sin corrección instrumental
            xp, popt, pcov, fwhm_sin = analizar_pico(
                dostheta, intensity, centro_ref,
                picos_detectados=picos,
                correcion_LaB6=False
            )
            A, xc, fwhm_deg, eta, b0, b1 = popt

            # Corrección instrumental LaB6
            _, popt_corr, pcov_corr, fwhm_con = analizar_pico(
                dostheta, intensity, centro_ref,
                picos_detectados=picos,
                correcion_LaB6=True,
                df_LaB6=df_LaB6
            )

            theta_rad = np.deg2rad(xc/2)
            beta_sin = np.deg2rad(fwhm_sin)
            beta_con = np.deg2rad(fwhm_con)

            D_sin = K*lambda_X/(beta_sin*np.cos(theta_rad))/10
            D_con = K*lambda_X/(beta_con*np.cos(theta_rad))/10 if beta_con > 0 else np.nan

            # Incertidumbres aproximadas del ajuste
            fwhm_err_deg = np.sqrt(max(pcov[2,2], 0))
            beta_err = np.deg2rad(fwhm_err_deg)
            D_err_sin = D_sin*beta_err/beta_sin

            fwhm_err_corr_deg = np.sqrt(max(pcov_corr[2,2], 0))
            beta_err_corr = np.deg2rad(fwhm_err_corr_deg)
            D_err_con = D_con*beta_err_corr/beta_con if beta_con > 0 else np.nan

            resultados.append({
                'Reflexión': hkl,
                '2theta teo (°)': centro_ref,
                '2theta (°)': xc,
                'FWHM sin (°)': fwhm_sin,
                'FWHM con (°)': fwhm_con,
                'eta': eta,
                'D sin (nm)': D_sin,
                'Error sin (nm)': D_err_sin,
                'D con (nm)': D_con,
                'Error con (nm)': D_err_con
            })

            xfit = np.linspace(xp.min(), xp.max(), 300)
            ax.plot(xfit, pvoigt_fondo(xfit, *popt), lw=1.2,
                    label=f'{hkl}: {xc:.2f}°')

        except (ValueError, RuntimeError) as e:
            print(f'{hkl}: no se pudo ajustar ({e})')

    ax.set_xlim(20,80)
    ax.set_title('M1 - Difractograma de RX')
    ax.set_ylim(0,None)
    ax.set_xlabel(r'$2\theta$ (°)')
    ax.set_ylabel('Intensidad (u.a.)')
    ax.set_title(f'{nombre}: ajuste pseudo-Voigt en reflexiones de magnetita')
    ax.grid()
    ax.legend(ncol=1, fontsize=8)
    plt.show()

    tabla = pd.DataFrame(resultados)
    if tabla.empty:
        raise ValueError(f'{nombre}: no se pudo ajustar ninguna reflexión')

    display(tabla.round(5))

    # Resumen estadístico
    resumen_sin = promedio_cristalita(tabla, 'D sin (nm)', 'Error sin (nm)')
    resumen_con = promedio_cristalita(tabla, 'D con (nm)', 'Error con (nm)')

    print(
        f"{nombre} sin corrección: D = {resumen_sin['D medio (nm)']:.1f} "
        f"± {resumen_sin['Incertidumbre (nm)']:.1f} nm; "
        f"desviación estándar = {resumen_sin['Desviación estándar (nm)']:.1f} nm; "
        f"χ² reducido = {resumen_sin['chi2 reducido']:.1f}; "
        f"N = {resumen_sin['N reflexiones']}"
    )
    print(
        f"{nombre} con corrección: D = {resumen_con['D medio (nm)']:.1f} "
        f"± {resumen_con['Incertidumbre (nm)']:.1f} nm; "
        f"desviación estándar = {resumen_con['Desviación estándar (nm)']:.1f} nm; "
        f"χ² reducido = {resumen_con['chi2 reducido']:.1f}; "
        f"N = {resumen_con['N reflexiones']}"
    )

    # Comparación de D frente a 2theta
    fig2, ax = plt.subplots(figsize=(9,5), constrained_layout=True)

    x = tabla['2theta (°)'].to_numpy()
    D_sin = tabla['D sin (nm)'].to_numpy()
    err_sin = tabla['Error sin (nm)'].to_numpy()
    D_con = tabla['D con (nm)'].to_numpy()
    err_con = tabla['Error con (nm)'].to_numpy()

    orden = np.argsort(x)
    x, D_sin, err_sin, D_con, err_con = (
        v[orden] for v in (x, D_sin, err_sin, D_con, err_con)
    )

    ax.errorbar(x, D_sin, yerr=err_sin, fmt='o-', capsize=3,
                lw=1.2, ms=5, label='Sin corrección instrumental')
    ax.errorbar(x, D_con, yerr=err_con, fmt='s--', capsize=3,
                lw=1.2, ms=5, label='Con corrección LaB$_6$')

    ax.axhspan(
        resumen_sin['D medio (nm)'] - resumen_sin['Desviación estándar (nm)'],
        resumen_sin['D medio (nm)'] + resumen_sin['Desviación estándar (nm)'],
        color='C0', alpha=0.10
    )
    ax.axhline(
        resumen_sin['D medio (nm)'], color='C0', ls=':', lw=1.5,
        label=f"Media sin corrección: {resumen_sin['D medio (nm)']:.1f} ± {resumen_sin['Desviación estándar (nm)']:.1f} nm"
    )

    ax.axhspan(
        resumen_con['D medio (nm)'] - resumen_con['Desviación estándar (nm)'],
        resumen_con['D medio (nm)'] + resumen_con['Desviación estándar (nm)'],
        color='C1', alpha=0.10
    )
    ax.axhline(
        resumen_con['D medio (nm)'], color='C1', ls=':', lw=1.5,
        label=f"Media con corrección: {resumen_con['D medio (nm)']:.1f} ± {resumen_con['Desviación estándar (nm)']:.1f} nm"
    )

    for _, fila in tabla.iterrows():
        ax.annotate(
            fila['Reflexión'],
            (fila['2theta (°)'], fila['D sin (nm)']),
            xytext=(0,-15), textcoords='offset points',
            ha='center', fontsize=8
        )

    ax.set_xlabel(r'$2\theta$ (°)')
    ax.set_ylabel('Tamaño de cristalita, $D$ (nm)')
    ax.set_title(f'{nombre}: comparación de tamaños de cristalita por Scherrer')
    ax.set_xlim(28,82)
    ax.set_ylim(0,)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.show()

    return tabla, resumen_sin, resumen_con, picos, fig,fig2
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


#%% Ajuste de dobletes LaB6: pseudo-Voigt + fondo lineal
lambda_Ka1 = 1.5406
lambda_Ka2 = 1.5444
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

# def doblete_pvoigt_fondo(x, A, xc, fwhm, eta, b0, b1):
#     # xc es la posición 2theta de Kalpha1
#     theta1 = np.deg2rad(xc/2)
#     argumento = (lambda_Ka2/lambda_Ka1)*np.sin(theta1)
#     theta2 = np.arcsin(np.clip(argumento, -1, 1))
#     xc2 = np.rad2deg(2*theta2)
#     z1 = (x-xc)/fwhm
#     z2 = (x-xc2)/fwhm
#     g1 = np.exp(-4*np.log(2)*z1**2)
#     l1 = 1/(1+4*z1**2)
#     g2 = np.exp(-4*np.log(2)*z2**2)
#     l2 = 1/(1+4*z2**2)
#     perfil1 = (1-eta)*g1 + eta*l1
#     perfil2 = (1-eta)*g2 + eta*l2
#     return b0 + b1*(x-xc) + A*(perfil1 + rel_int_Ka2*perfil2)

def doblete_pvoigt_fondo(x, A, xc, fwhm, eta, rel_int_Ka2, b0, b1):
    theta1 = np.deg2rad(xc/2)
    argumento = (lambda_Ka2/lambda_Ka1)*np.sin(theta1)
    theta2 = np.arcsin(np.clip(argumento, -1, 1))
    xc2 = np.rad2deg(2*theta2)

    z1 = (x-xc)/fwhm
    z2 = (x-xc2)/fwhm
    g1 = np.exp(-4*np.log(2)*z1**2)
    l1 = 1/(1+4*z1**2)
    g2 = np.exp(-4*np.log(2)*z2**2)
    l2 = 1/(1+4*z2**2)

    perfil1 = (1-eta)*g1 + eta*l1
    perfil2 = (1-eta)*g2 + eta*l2

    return b0 + b1*(x-xc) + A*(perfil1 + rel_int_Ka2*perfil2)

resultados_LaB6 = []
ajustes_LaB6 = {}

for hkl, N, centro_ref in reflexiones_LaB6:
    # Ventana local alrededor de la reflexión
    ventana = 0.55
    mask = (dostheta_LaB6 >= centro_ref-ventana) & (dostheta_LaB6 <= centro_ref+ventana)
    xp = dostheta_LaB6[mask]
    yp = intensity_LaB6[mask]
    if len(xp) < 10:
        print(f'{hkl}: no hay suficientes puntos para ajustar')
        continue

    # Inicialización usando el máximo detectado más cercano
    candidatos = [
        (i, theta, intens) for i, theta, intens in picos_LaB6
        if centro_ref-0.25 <= theta <= centro_ref+0.25
    ]
    if candidatos:
        _, xc0, intens0 = min(candidatos, key=lambda p: abs(p[1]-centro_ref))
    else:
        xc0 = xp[np.argmax(yp)]
        intens0 = np.max(yp)

    b0 = np.percentile(yp, 10)
    A0 = max(intens0-b0, 1e-6)
    fwhm0 = 0.07

    try:
        popt, pcov = curve_fit(
            doblete_pvoigt_fondo, xp, yp,
            p0=[A0, xc0, fwhm0, 0.5, 0.5, b0, 0],
            bounds=(
                [0, centro_ref-0.25, 0.005, 0, 0, -np.inf, -np.inf],
                [np.inf, centro_ref+0.25, 0.30, 1, 1, np.inf, np.inf]
            ),maxfev=50000)

        A, xc, fwhm, eta, rel_int_Ka2, b0_fit, b1 = popt        
        errores = np.sqrt(np.maximum(np.diag(pcov), 0))
        err_xc = errores[1]
        err_fwhm = errores[2]

        theta1 = np.deg2rad(xc/2)
        theta2 = np.arcsin(np.clip(lambda_Ka2/lambda_Ka1*np.sin(theta1), -1, 1))
        xc2 = np.rad2deg(2*theta2)

        # Parámetro de red calculado con Kalpha1
        a = lambda_Ka1*np.sqrt(N)/(2*np.sin(theta1))
        err_a = abs(a/np.tan(theta1))*np.deg2rad(err_xc/2)

        resultados_LaB6.append({
            'Reflexión': hkl,
            'N': N,
            '2theta Ka1 (°)': xc,
            '2theta Ka2 (°)': xc2,
            'FWHM Ka1 (°)': fwhm,
            'Error FWHM (°)': err_fwhm,
            'eta': eta,
            'a (Å)': a,
            'Error centro (°)': err_xc,
            'Error a (Å)': err_a})

        ajustes_LaB6[hkl] = (xp, yp, popt)

    except (RuntimeError, ValueError) as e:
        print(f'{hkl}: fallo el ajuste ({e})')

df_LaB6 = pd.DataFrame(resultados_LaB6)
display(df_LaB6.round(6))

#%% Gráficos de los ajustes del doblete
fig_LaB6c, axs = plt.subplots(3, 4, figsize=(17, 10), constrained_layout=True)
axs = axs.ravel()

for ax, (hkl, N, centro_ref) in zip(axs, reflexiones_LaB6):
    if hkl not in ajustes_LaB6:
        ax.set_visible(False)
        continue

    xp, yp, popt = ajustes_LaB6[hkl]
    xfit = np.linspace(xp.min(), xp.max(), 1000)
    yfit = doblete_pvoigt_fondo(xfit, *popt)

    ax.plot(xp, yp, '.', ms=2, color='0.35', label='Datos')
    ax.plot(xfit, yfit, '-', lw=1.3, label='Ajuste doblete')
    ax.axvline(popt[1], color='tab:red', ls='--', lw=0.8, label=r'$K\alpha_1$')
    theta2 = np.arcsin(np.clip(lambda_Ka2/lambda_Ka1*np.sin(np.deg2rad(popt[1]/2)), -1, 1))
    xc2 = np.rad2deg(2*theta2)
    ax.axvline(xc2, color='tab:blue', ls=':', lw=0.8, label=r'$K\alpha_2$')
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

#% Perfil instrumental: FWHM de Kalpha1 en función de 2theta
fig_Lab6_perfil, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
ax.errorbar(
    df_LaB6['2theta Ka1 (°)'],
    df_LaB6['FWHM Ka1 (°)'],
    yerr=df_LaB6['Error FWHM (°)'],
    fmt='o', capsize=3, label=r'$K\alpha_1$'
)
ax.set_xlabel(r'$2\theta$ (°)')
ax.set_ylabel('FWHM (°)')
ax.set_title(r'Ancho del perfil de $K\alpha_1$ para LaB$_6$')
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlabel('2θ (°)')
plt.show()

#% Resumen del parámetro de red
print(f'a promedio = {df_LaB6["a (Å)"].mean():.6f} Å')
print(f'Desviación estándar entre reflexiones = {df_LaB6["a (Å)"].std(ddof=1):.6f} Å')
print(f'Referencia NIST SRM 660c = {a_ref_LaB6:.6f} Å')
print(f'Diferencia relativa = {(df_LaB6["a (Å)"].mean()/a_ref_LaB6-1)*100:.4f} %')

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

#M1 - Análisis de Scherrer
tabla_M1, resumen_M1_sin, resumen_M1_con,picos_M1, fig_M1b,fig_M1c = analizar_muestra(
    'M1', dostheta_M1, intensity_M1, picos_M1)

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
#% M2 - Análisis de Scherrer
tabla_M2, resumen_M2_sin, resumen_M2_con, picos_M2, fig_M2b,fig_M2c = analizar_muestra(
    'M2', dostheta_M2, intensity_M2, picos_M2)

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

#% M3 - Análisis de Scherrer
tabla_M3, resumen_M3_sin, resumen_M3_con, picos_M3, fig_M3b,fig_M3c = analizar_muestra(
    'M3', dostheta_M3, intensity_M3, picos_M3)
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
#% M4 - Análisis de Scherrer
tabla_M4, resumen_M4_sin, resumen_M4_con, picos_M4, fig_M4b,fig_M4c= analizar_muestra(
    'M4', dostheta_M4, intensity_M4, picos_M4)
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
#M5 - Análisis de Scherrer
tabla_M5, resumen_M5_sin, resumen_M5_con, picos_M5, fig_M5b,fig_M5c = analizar_muestra(
    'M5', dostheta_M5, intensity_M5, picos_M5)    
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

#M6 - Análisis de Scherrer
tabla_M6, resumen_M6_sin, resumen_M6_con, picos_M6, fig_M6b,fig_M6c = analizar_muestra(
    'M6', dostheta_M6, intensity_M6, picos_M6)
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

#M7 - Análisis de Scherrer
tabla_M7, resumen_M7_sin, resumen_M7_con, picos_M7, fig_M7b,fig_M7c = analizar_muestra(
    'M7', dostheta_M7, intensity_M7, picos_M7)

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
#M9 - Análisis de Scherrer
tabla_M9, resumen_M9_sin, resumen_M9_con, picos_M9, fig_M9b,fig_M9c = analizar_muestra(
    'M9', dostheta_M9, intensity_M9, picos_M9)
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

#M10 - Análisis de Scherrer
tabla_M10, resumen_M10_sin, resumen_M10_con, picos_M10, fig_M10b,fig_M10c = analizar_muestra(
    'M10', dostheta_M10, intensity_M10, picos_M10)

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
#M11 - Análisis de Scherrer
tabla_M11, resumen_M11_sin, resumen_M11_con, picos_M11, fig_M11b,fig_M11c = analizar_muestra(
    'M11', dostheta_M11, intensity_M11, picos_M11[:-1])
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

#M12 - Análisis de Scherrer
tabla_M12, resumen_M12_sin, resumen_M12_con, picos_M12, fig_M12b,fig_M12c = analizar_muestra(
    'M12', dostheta_M12, intensity_M12, picos_M12)

# %% salvo todas las figuras
fig_LaB6a.savefig('LaB6_DRX',dpi=300)
fig_LaB6b.savefig('LaB6_DRX_c_picos',dpi=300)
fig_LaB6c.savefig('LaB6_perfil_exp',dpi=300)

fig_all.savefig('DRX_M1_M2_M3_M4_M5_M6_M7_M9_M10_M11_M12.png')
fig_M1a.savefig('M1_DRX',dpi=300)
fig_M1b.savefig('M1_DRX_c_ajustes',dpi=300)
fig_M1c.savefig('M1_tamaño_cristalita',dpi=300)

fig_M2a.savefig('M2_DRX',dpi=300)
fig_M2b.savefig('M2_DRX_c_ajustes',dpi=300)
fig_M2c.savefig('M2_tamaño_cristalita',dpi=300)

fig_M3a.savefig('M3_DRX',dpi=300)
fig_M3b.savefig('M3_DRX_c_ajustes',dpi=300)
fig_M3c.savefig('M3_tamaño_cristalita',dpi=300)

fig_M4a.savefig('M4_DRX',dpi=300)
fig_M4b.savefig('M4_DRX_c_ajustes',dpi=300)
fig_M4c.savefig('M4_tamaño_cristalita',dpi=300)

fig_M5a.savefig('M5_DRX',dpi=300)
fig_M5b.savefig('M5_DRX_c_ajustes',dpi=300)
fig_M5c.savefig('M5_tamaño_cristalita',dpi=300)

fig_M6a.savefig('M6_DRX',dpi=300)
fig_M6b.savefig('M6_DRX_c_ajustes',dpi=300)
fig_M6c.savefig('M6_tamaño_cristalita',dpi=300)

fig_M7a.savefig('M7_DRX',dpi=300)
fig_M7b.savefig('M7_DRX_c_ajustes',dpi=300)
fig_M7c.savefig('M7_tamaño_cristalita',dpi=300)

fig_M9a.savefig('M9_DRX',dpi=300)
fig_M9b.savefig('M9_DRX_c_ajustes',dpi=300)
fig_M9c.savefig('M9_tamaño_cristalita',dpi=300)

fig_M10a.savefig('M10_DRX',dpi=300)
fig_M10b.savefig('M10_DRX_c_ajustes',dpi=300)
fig_M10c.savefig('M10_tamaño_cristalita',dpi=300)

fig_M11a.savefig('M11_DRX',dpi=300)
fig_M11b.savefig('M11_DRX_c_ajustes',dpi=300)
fig_M11c.savefig('M11_tamaño_cristalita',dpi=300)

fig_M12a.savefig('M12_DRX',dpi=300)
fig_M12b.savefig('M12_DRX_c_ajustes',dpi=300)
fig_M12c.savefig('M12_tamaño_cristalita',dpi=300)
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
plt.savefig('tamaño_vs_muestra.png',dpi=300)
plt.show()
# %%
