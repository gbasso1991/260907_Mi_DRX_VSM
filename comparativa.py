#%% 
'''
Comparativa de muestras: M1, M2, M3, M4, M5, M6, M7, M9 M10, M11, M12
ESAR
SEM
DRX
VSM
'''
#%% Librerias y paquetes
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from scipy.interpolate import interp1d
from clase_resultados import ResultadosESAR
import pandas as pd

pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	ESAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))

                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor

                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    ESAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, ESAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentraciong/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float) #kA/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer ESAR, tau y Hc de resultados
def extraer_ESAR_tau(resultados):
    ESAR, tau, Hc = [], [], []
    time , temperatura = [], []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)
        ESAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m'])
    return ESAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% funcion Warming rate promedio por campo
def warming_rates_promedio(resultados):
    
    rates = []

    # warming rate de cada una de las 9 mediciones
    for r in resultados:
        dt = r.time[-1] - r.time[0]
        dT = r.temperatura[-1] - r.temperatura[0]
        rates.append(dT / dt)

    # promedio de cada grupo de 3
    rates_promedio = []

    for i in range(0, 9, 3):
        grupo = rates[i:i+3]
        
        promedio = np.mean(grupo)
        std = np.std(grupo, ddof=1)
        
        rates_promedio.append(ufloat(promedio, std))

    return rates_promedio

#%%Campo magnético calculado a partir del nombre de cada archivo
def campo_desde_archivo(archivo):
    nombre_archivo = os.path.basename(archivo)
    Idc = float(nombre_archivo.split('kHz_')[1].split('dA')[0])/10
    return (Idc*pendiente_HvsI + ordenada_HvsI)/1000
#%% Agrupar warming rates por campo
def warming_rates_por_campo(res, campos):
    grupos = {}
    for r, H0 in zip(res, campos):
        dt = r.time[-1] - r.time[0]
        dT = r.temperatura[-1] - r.temperatura[0]
        grupos.setdefault(round(H0, 3), []).append(dT/dt)

    WR = {}
    for campo, rates in grupos.items():
        promedio = np.mean(rates)
        std = np.std(rates, ddof=1) if len(rates) > 1 else 0.0
        WR[campo] = ufloat(promedio, std)

    return WR
#%% funcion leer resultados Feret
def leer_resultados_feret(archivo='Resultados_lognormal_Feret.txt'):
    """
    Lee la tabla de resultados del ajuste lognormal de los diámetros Feret.

    Parameters
    ----------
    archivo : str, optional
        Ruta al archivo de resultados, separado por tabulaciones.

    Returns
    -------
    pandas.DataFrame
        Tabla con los resultados de cada muestra:
        Muestra, N, Media, u(Media), Std, u(Std), Mediana, Moda,
        shape y scale.
    """
    df = pd.read_csv(archivo, sep='\t')
    return df

#%% funcion leer resultados cristalita
def leer_resultados_cristalita(archivo='Resultados_cristalitas_Scherrer.txt'):
    """
    Lee la tabla de resultados de tamaños de cristalita obtenidos mediante
    la ecuación de Scherrer, con y sin corrección instrumental.

    Parámetros
    ----------
    archivo : str
        Ruta al archivo de resultados, separado por tabulaciones.

    Retorna
    -------
    pd.DataFrame
        Tabla con los tamaños de cristalita medios, sus incertidumbres
        y la cantidad de reflexiones consideradas para cada muestra.
    """
    return pd.read_csv(archivo, sep='\t')
#%% Importo ciclos y resultados  ordenado por temperatura

nombres = ['M1','M2','M3','M4','M5','M6','M9','M10','M11','M12']
concentracion = [5.7,19.8,48.1,13.0,23.0,20,19.6,24.5,5.2,9.9] # g/L Magnetita
ciclos_M1 = glob("data_ESAR/M1/*ciclo_promedio_H_M*")
resultados_M1 = glob("data_ESAR/M1/*resultados*")
ciclos_M2 = glob("data_ESAR/M2/*ciclo_promedio_H_M*")
resultados_M2 = glob("data_ESAR/M2/*resultados*")
ciclos_M3 = glob("data_ESAR/M3/*ciclo_promedio_H_M*")
resultados_M3 = glob("data_ESAR/M3/*resultados*")
ciclos_M4 = glob("data_ESAR/M4/*ciclo_promedio_H_M*")
resultados_M4 = glob("data_ESAR/M4/*resultados*")
ciclos_M5 = glob("data_ESAR/M5/*ciclo_promedio_H_M*")
resultados_M5 = glob("data_ESAR/M5/*resultados*")
ciclos_M6 = glob("data_ESAR/M6/*ciclo_promedio_H_M*")
resultados_M6 = glob("data_ESAR/M6/*resultados*")
ciclos_M9 = glob("data_ESAR/M9/*ciclo_promedio_H_M*")
resultados_M9 = glob("data_ESAR/M9/*resultados*")
ciclos_M10 = glob("data_ESAR/M10/*ciclo_promedio_H_M*")
resultados_M10 = glob("data_ESAR/M10/*resultados*")
ciclos_M11 = glob("data_ESAR/M11/*ciclo_promedio_H_M*")
resultados_M11 = glob("data_ESAR/M11/*resultados*")
ciclos_M12 = glob("data_ESAR/M12/*ciclo_promedio_H_M*")
resultados_M12 = glob("data_ESAR/M12/*resultados*")

# Ordeno listas
for c in [ciclos_M1, ciclos_M2, ciclos_M3, ciclos_M4, ciclos_M5, ciclos_M6, ciclos_M9, ciclos_M10, ciclos_M11, ciclos_M12]:
    c.sort()
    print(c)
for r in [resultados_M1, resultados_M2, resultados_M3, resultados_M4, resultados_M5, resultados_M6, resultados_M9, resultados_M10, resultados_M11, resultados_M12]:
    r.sort()   
    print(r)

#%% Calcular el campo magnético de cada ciclo

muestras = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12']

ciclos = [ciclos_M1, ciclos_M2, ciclos_M3, ciclos_M4, ciclos_M5,
          ciclos_M6, ciclos_M9, ciclos_M10, ciclos_M11, ciclos_M12]

resultados = [resultados_M1, resultados_M2, resultados_M3, resultados_M4, resultados_M5,
              resultados_M6, resultados_M9, resultados_M10, resultados_M11, resultados_M12]

campo_M1, campo_M2, campo_M3, campo_M4, campo_M5 = [], [], [], [], []
campo_M6, campo_M9, campo_M10, campo_M11, campo_M12 = [], [], [], [], []

campo_muestras = [campo_M1, campo_M2, campo_M3, campo_M4, campo_M5,
                  campo_M6, campo_M9, campo_M10, campo_M11, campo_M12]

for lista_ciclos, campo_lista in zip(ciclos, campo_muestras):
    for archivo in lista_ciclos:
        nombre_archivo = os.path.basename(archivo)
        Idc = float(nombre_archivo.split('kHz_')[1].split('dA')[0])/10
        H0 = (Idc*pendiente_HvsI + ordenada_HvsI)/1000
        campo_lista.append(H0)

campos = campo_muestras
#%% extraigo ESAR, tau y Hc
ESAR_M1, tau_M1, Hc_M1 = extraer_ESAR_tau(resultados_M1)
res_M1=[]
ESAR_M2, tau_M2, Hc_M2 = extraer_ESAR_tau(resultados_M2)
res_M2=[]
ESAR_M3, tau_M3, Hc_M3 = extraer_ESAR_tau(resultados_M3)
res_M3=[]
ESAR_M4, tau_M4, Hc_M4 = extraer_ESAR_tau(resultados_M4)
res_M4=[]
ESAR_M5, tau_M5, Hc_M5 = extraer_ESAR_tau(resultados_M5)
res_M5=[]
ESAR_M6, tau_M6, Hc_M6 = extraer_ESAR_tau(resultados_M6)
res_M6=[]
ESAR_M9, tau_M9, Hc_M9 = extraer_ESAR_tau(resultados_M9)
res_M9=[]
ESAR_M10, tau_M10, Hc_M10 = extraer_ESAR_tau(resultados_M10)
res_M10=[]
ESAR_M11, tau_M11, Hc_M11 = extraer_ESAR_tau(resultados_M11)
res_M11=[]
ESAR_M12, tau_M12, Hc_M12 = extraer_ESAR_tau(resultados_M12)
res_M12=[]
#%% Importar resultados ESAR y calcular warming rates

nombres = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12']
concentracion = [5.7, 19.8, 48.1, 13.0, 23.0, 20.0, 19.6, 24.5, 5.2, 9.9] # g/L Magnetita

ciclos = [ciclos_M1, ciclos_M2, ciclos_M3, ciclos_M4, ciclos_M5,
          ciclos_M6, ciclos_M9, ciclos_M10, ciclos_M11, ciclos_M12]

resultados = [resultados_M1, resultados_M2, resultados_M3, resultados_M4, resultados_M5,
              resultados_M6, resultados_M9, resultados_M10, resultados_M11, resultados_M12]


campos_resultados = [[campo_desde_archivo(r) for r in lista] for lista in resultados]

# Crear objetos ResultadosESAR
res_muestras = []
for nombre, lista_resultados in zip(nombres, resultados):
    print(f'Resultados {nombre}', '='*80, '\n')
    res_muestras.append([ResultadosESAR(r) for r in lista_resultados])

# Extraer ESAR, tau y Hc
datos_ESAR_tau_Hc = [extraer_ESAR_tau(lista) for lista in resultados]
ESAR = [datos[0] for datos in datos_ESAR_tau_Hc]
tau = [datos[1] for datos in datos_ESAR_tau_Hc]
Hc = [datos[2] for datos in datos_ESAR_tau_Hc]



warming_rates = [warming_rates_por_campo(res, campos)
                 for res, campos in zip(res_muestras, campos_resultados)]

#%% Ploteo comparativo de ciclos de histéresis

colores_campo = {20: 'C3', 29: 'C4', 38: 'C0', 47: 'C1', 57: 'C2'}

fig00, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True, sharex=True, sharey=True)

for i, (nombre, lista_ciclos, lista_ESAR, conc, campo_lista) in enumerate(
        zip(muestras, ciclos, ESAR, concentracion, campos)):

    ax = axs.ravel()[i]

    for j, (archivo, H0) in enumerate(zip(lista_ciclos, campo_lista)):
        _, _, _, H, M, meta = lector_ciclos(archivo)

        campo_ref = min(colores_campo, key=lambda c: abs(H0 - c))
        color = colores_campo[campo_ref]

        etiqueta = f'{lista_ESAR[j]:.2uS}' if lista_ESAR[j].s < 100 else f'{lista_ESAR[j]:.3uS}'

        ax.plot(H, M/conc, '-', color=color, label=f'{etiqueta}')

    ax.set_title(f'{nombre} — {conc} g/L', loc='left')
    ax.grid()
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=8, title='ESAR (W/g)')

for ax in axs[-1, :]:
    ax.set_xlabel('H (kA/m)')

for ax in axs[:, 0]:
    ax.set_ylabel('M/[NPM] (Am$^2$/kg)')

plt.suptitle('Ciclos promedio normalizados por concentracion — 300 kHz', fontsize=14)
plt.savefig('00_comparativa_ciclos_norm_all.png', dpi=300)


#%% Templogs — todas las muestras

colores_campo = {20: 'C3', 29: 'C4', 38: 'C0', 47: 'C1', 57: 'C2'}

def color_campo(H0):
    campo_ref = min(colores_campo, key=lambda c: abs(H0 - c))
    return colores_campo[campo_ref]

fig02, axs = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True, sharex=True, sharey=True)

for j, (nombre, res, campos, WR, conc) in enumerate(
        zip(nombres, res_muestras, campos_resultados, warming_rates, concentracion)):

    ax = axs.ravel()[j]
    etiquetados = set()

    for r, H0 in zip(res, campos):
        clave = round(H0, 3)
        color = color_campo(H0)

        etiqueta = None
        if clave not in etiquetados:
            etiqueta = f'{H0:.1f} kA/m — {WR[clave]:.1uS}'
            etiquetados.add(clave)

        ax.plot(r.time, r.temperatura, '.-', color=color, label=etiqueta)

    ax.set_title(f'{nombre} — {conc} g/L', loc='left')
    ax.grid()
    ax.legend(title='Campo — WR (°C/s)', loc='best', frameon=True, shadow=True, fontsize=8)
    ax.set_xlim(0,40)
for ax in axs[-1, :]:
    ax.set_xlabel('t (s)')

for ax in axs[:, 0]:
    ax.set_ylabel('T (°C)')

plt.suptitle('Templogs — 300 kHz', fontsize=16)
plt.savefig('01_templogs_all.png',dpi=300)
plt.show()


#%% Ploteo comparativo de errorbars de ESAR, tau y Hc

categorias = sorted({round(H0, 1) for lista in campos_resultados for H0 in lista})
datos = [ESAR, tau, Hc]
etiquetas_y = ['ESAR (W/g)', r'$\tau$ (ns)', r'$H_c$ (kA/m)']

fig03, axs = plt.subplots(3, len(nombres), figsize=(25, 9),
                          sharey='row', sharex=True, constrained_layout=True)

for j, nombre in enumerate(nombres):
    axs[0, j].set_title(f'{nombre}\n{concentracion[j]} g/L', loc='left')

for fila, datos_fila in enumerate(datos):
    for col, (valores, campos) in enumerate(zip(datos_fila, campos_resultados)):
        ax = axs[fila, col]
        grupos = {}

        for valor, H0 in zip(valores, campos):
            grupos.setdefault(round(H0, 1), []).append(valor)

        for H0, valores_campo in grupos.items():
            x0 = categorias.index(H0)
            color = color_campo(H0)
            n = len(valores_campo)
            ancho = min(0.16, 0.65/n)
            posiciones = x0 + (np.arange(n) - (n-1)/2)*ancho

            for x, s in zip(posiciones, valores_campo):
                ax.bar(x, s.n, yerr=s.s, width=ancho*0.9,
                       capsize=3, color=color)

        ax.grid(axis='y', alpha=0.5)

        if col == 0:
            ax.set_ylabel(etiquetas_y[fila])

for ax in axs[-1, :]:
    ax.set_xticks(np.arange(len(categorias)))
    ax.set_xticklabels([f'{c:.1f}' for c in categorias], rotation=45)
    ax.set_xlabel('$H_0$ (kA/m)')

plt.suptitle('Comparativa ESAR — $\\tau$ — $H_c$ — 300 kHz', fontsize=16)
plt.show()


#%% Salvo figuras
# fig00.savefig('00_metacomparativa_ciclos.png',dpi=300)
# fig01.savefig('01_metacomparativa_ciclos_normalizados.png',dpi=300)
# fig02.savefig('02_metacomparativa_templogs.png',dpi=300)
# fig03.savefig('03_metacomparativa_ESAR_Tau_Hc.png',dpi=300)
#%% Importo resultados SEM
resultados_feret = leer_resultados_feret('Resultados_lognormal_Feret.txt')
print(resultados_feret)

diametros_medios = resultados_feret['Media (nm)'].to_numpy()
incertezas_diametros = resultados_feret['u(Media) (nm)'].to_numpy()

diametros_medios = [ufloat(media, incerteza)
    for media, incerteza in zip(resultados_feret['Media (nm)'],resultados_feret['u(Media) (nm)'])]
#%% Importo resultados DRX
df_cristalitas = leer_resultados_cristalita('Resultados_cristalitas_Scherrer.txt')
print(df_cristalitas)
muestras = df_cristalitas['Muestra'].to_numpy()
D_sin = df_cristalitas['D sin corrección (nm)'].to_numpy()
u_D_sin = df_cristalitas['u(D sin corrección) (nm)'].to_numpy()
D_con = df_cristalitas['D con corrección (nm)'].to_numpy()
u_D_con = df_cristalitas['u(D con corrección) (nm)'].to_numpy()


#%% Preparar datos para ESAR vs diámetros

nombres_ESAR = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M9', 'M10', 'M11', 'M12']

ESAR_max = []
ESAR_max_std = []
ESAR_max_n = []
H_max_muestras = []

for nombre, valores_ESAR, campos in zip(nombres_ESAR, ESAR, campos_resultados):
    grupos = {}
    for valor, H0 in zip(valores_ESAR, campos):
        grupos.setdefault(round(H0, 1), []).append(valor)

    H_max = max(grupos)
    valores_max = grupos[H_max]
    valores_nominales = np.array([s.n for s in valores_max])

    ESAR_max.append(np.mean(valores_nominales))
    ESAR_max_n.append(len(valores_max))
    H_max_muestras.append(H_max)

    if len(valores_max) > 1:
        ESAR_max_std.append(np.std(valores_nominales, ddof=1))
    else:
        ESAR_max_std.append(valores_max[0].s)

df_ESAR = pd.DataFrame({
    'Muestra': nombres_ESAR,
    'H máximo (kA/m)': H_max_muestras,
    'N repeticiones': ESAR_max_n,
    'ESAR (W/g)': ESAR_max,
    'u(ESAR) (W/g)': ESAR_max_std
})

df_Feret = resultados_feret[
    ['Muestra', 'Media (nm)', 'u(Media) (nm)']
].rename(columns={
    'Media (nm)': 'D_Feret (nm)',
    'u(Media) (nm)': 'u(D_Feret) (nm)'
})

df_DRX = df_cristalitas[
    ['Muestra', 'D con corrección (nm)', 'u(D con corrección) (nm)']
].rename(columns={
    'D con corrección (nm)': 'D cristalita (nm)',
    'u(D con corrección) (nm)': 'u(D cristalita) (nm)'
})

df_comparacion = df_ESAR.merge(df_Feret, on='Muestra').merge(df_DRX, on='Muestra')

print(df_comparacion)
# %%
#%% ESAR a campo máximo vs diámetro Feret y tamaño de cristalita

# Calcular el ESAR promedio a partir de todas las mediciones del campo máximo
nombres_ESAR = nombres
ESAR_max = []
ESAR_max_std = []
ESAR_max_n = []
H_max_muestras = []

for nombre, valores_ESAR, campos in zip(nombres_ESAR, ESAR, campos_resultados):
    grupos = {}
    for valor, H0 in zip(valores_ESAR, campos):
        grupos.setdefault(round(H0, 1), []).append(valor)

    H_max = max(grupos)
    valores_max = grupos[H_max]
    valores_nominales = np.array([s.n for s in valores_max])

    ESAR_max.append(np.mean(valores_nominales))
    ESAR_max_n.append(len(valores_max))
    H_max_muestras.append(H_max)

    if len(valores_max) > 1:
        ESAR_max_std.append(np.std(valores_nominales, ddof=1))
    else:
        # Con una sola medición no se puede estimar la dispersión entre repeticiones.
        # Se conserva la incertidumbre propia del resultado individual.
        ESAR_max_std.append(valores_max[0].s)

ESAR_max = np.array(ESAR_max)
ESAR_max_std = np.array(ESAR_max_std)
ESAR_max_n = np.array(ESAR_max_n)
H_max_muestras = np.array(H_max_muestras)

#%% Unificar los datos por nombre de muestra

df_ESAR = pd.DataFrame({
    'Muestra': nombres_ESAR,
    'H máximo (kA/m)': H_max_muestras,
    'N repeticiones': ESAR_max_n,
    'ESAR (W/g)': ESAR_max,
    'u(ESAR) (W/g)': ESAR_max_std
})

df_Feret = resultados_feret[
    ['Muestra', 'Media (nm)', 'u(Media) (nm)']
].rename(columns={
    'Media (nm)': 'D_Feret (nm)',
    'u(Media) (nm)': 'u(D_Feret) (nm)'
})

df_DRX = df_cristalitas[
    ['Muestra', 'D con corrección (nm)', 'u(D con corrección) (nm)']
].rename(columns={
    'D con corrección (nm)': 'D cristalita (nm)',
    'u(D con corrección) (nm)': 'u(D cristalita) (nm)'
})

df_comparacion = df_ESAR.merge(df_Feret, on='Muestra').merge(df_DRX, on='Muestra')

print(df_comparacion[['Muestra', 'H máximo (kA/m)', 'N repeticiones', 'ESAR (W/g)']])

#%% Gráfico comparativo

fig, axs = plt.subplots(3, 1, figsize=(15, 10), constrained_layout=True)

# ESAR vs diámetro Feret
df_plot = df_comparacion.sort_values('D_Feret (nm)')

axs[0].errorbar(df_plot['D_Feret (nm)'],df_plot['ESAR (W/g)'],
    xerr=df_plot['u(D_Feret) (nm)'],yerr=df_plot['u(ESAR) (W/g)'],
    fmt='o',capsize=4)

for _, fila in df_plot.iterrows():
    axs[0].annotate(fila['Muestra'],(fila['D_Feret (nm)'], fila['ESAR (W/g)']),
        xytext=(5, 5), textcoords='offset points')

axs[0].set_xlabel('Diámetro medio Feret (nm)')
axs[0].set_ylabel('ESAR (W/g)')
axs[0].set_title('ESAR vs diámetro Feret (SEM)', loc='left')

# ESAR vs tamaño de cristalita: con y sin corrección instrumental

# Sin corrección instrumental
df_plot = df_comparacion.merge(df_cristalitas[['Muestra', 'D sin corrección (nm)', 'u(D sin corrección) (nm)']],
on='Muestra').sort_values('D sin corrección (nm)')

axs[1].errorbar(
    df_plot['D sin corrección (nm)'],
    df_plot['ESAR (W/g)'],
    xerr=df_plot['u(D sin corrección) (nm)'],
    yerr=df_plot['u(ESAR) (W/g)'],
    fmt='o',
    color='tab:blue',
    capsize=4,
    label='Sin corrección instrumental'
)

for _, fila in df_plot.iterrows():
    axs[1].annotate(
        fila['Muestra'],
        (fila['D sin corrección (nm)'], fila['ESAR (W/g)']),
        xytext=(5, 5),
        textcoords='offset points',
        color='tab:blue'    )

# Con corrección instrumental LaB6
df_plot = df_comparacion.sort_values('D cristalita (nm)')

axs[2].errorbar(
    df_plot['D cristalita (nm)'],
    df_plot['ESAR (W/g)'],
    xerr=df_plot['u(D cristalita) (nm)'],
    yerr=df_plot['u(ESAR) (W/g)'],
    fmt='s',
    color='tab:orange',
    capsize=4,
    label=r'Con corrección LaB$_6$'
)

for _, fila in df_plot.iterrows():
    axs[2].annotate(
        fila['Muestra'],
        (fila['D cristalita (nm)'], fila['ESAR (W/g)']),
        xytext=(5, -10),
        textcoords='offset points',
        color='tab:orange'
    )

axs[2].set_xlabel('Tamaño de cristalita (nm)')
axs[2].set_ylabel('ESAR (W/g)')
axs[1].set_title('ESAR vs tamaño de cristalita (DRX) - sin correccion', loc='left')
axs[2].set_title('ESAR vs tamaño de cristalita (DRX) - con correccion', loc='left')

for a in axs:
    a.grid()    
    a.legend(frameon=True, shadow=True)

# %%
