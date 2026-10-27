#%%clase_resultados.py 
import os
import glob
import numpy as np
from uncertainties import ufloat, unumpy
from lectores import lector_resultados, lector_ciclos  
#%%
"""
clase_resultados_esar.py
Clase principal para manejar resultados de experimentos ESAR.
Depende de: lectores.py (contiene lector_resultados y lector_ciclos)
"""

import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from lectores import lector_resultados, lector_ciclos

class ResultadosESAR:
    """
    Clase para cargar, almacenar y analizar resultados de experimentos ESAR.
    
    Al instanciar con un directorio 'Analisis_[fecha]', automáticamente:
    1. Carga datos de 'resultados.txt'
    2. Carga primer y último ciclo de 'ciclos_H_M/'
    3. Organiza todo en atributos accesibles
    
    Ejemplo:
    ---------
    res = ResultadosESAR('ruta/Analisis_20240215')
    print(res.Hc)           # Array de campos coercitivos
    print(res.SAR)          # Array de valores SAR
    res.plot_ciclos_comparacion()  # Grafica ciclos
    """
    
    def __init__(self, directorio_analisis):
        """
        Inicializa la clase cargando todos los datos del directorio.
        
        Parámetros:
        -----------
        directorio_analisis : str
            Ruta completa al directorio 'Analisis_[fecha]' que contiene:
            - Archivo 'resultados.txt' (o '*resultados.txt')
            - Subdirectorio 'ciclos_H_M/' con archivos '*_ciclo_H_M.txt'
        """
        # Puede recibir un directorio o directamente un archivo resultados.txt
        ruta = os.path.abspath(directorio_analisis)

        if os.path.isfile(ruta):
            # Se recibió directamente el archivo de resultados
            self.ruta_resultados = ruta
            self.directorio = os.path.dirname(ruta)

        elif os.path.isdir(ruta):
            # Se recibió un directorio
            self.directorio = ruta

        else:
            raise FileNotFoundError(f"✗ Ruta no encontrada: {ruta}")

        self.directorio_ciclos = os.path.join(self.directorio, 'ciclos_H_M')
        self.nombre_base = os.path.basename(self.directorio)    
        print(f"\n{'='*60}")
        print(f"CARGANDO RESULTADOS ESAR")
        print(f"{'='*60}")
        print(f"Directorio: {self.directorio}")
        
        # 1. Cargar archivo de resultados principales
        self._cargar_resultados_txt()
        
        # 2. Cargar ciclos de magnetización extremos
        self._cargar_ciclos_extremos()
        
        # 3. Mostrar resumen de lo cargado
        self._mostrar_resumen()
        
        # 4. Calcular estadísticas básicas
        self._calcular_estadisticas()
        #print(f"  - Primer ciclo: {os.path.basename(self.primer_ciclo['ruta_archivo'])}")
        #print(f"  - Último ciclo: {os.path.basename(self.ultimo_ciclo['ruta_archivo'])}")

    
    def _cargar_resultados_txt(self):
        # Si ya se indicó un archivo específico, usarlo
        if hasattr(self, 'ruta_resultados'):
            pass

        else:
            # Buscar cualquier archivo que termine en resultados.txt
            patron_resultados = os.path.join(self.directorio, '*resultados.txt')
            archivos_resultados = glob.glob(patron_resultados)

            if not archivos_resultados:
                raise FileNotFoundError(
                    f"✗ No se encontró archivo resultados.txt en {self.directorio}")

            self.ruta_resultados = archivos_resultados[0]
        nombre_archivo = os.path.basename(self.ruta_resultados)
        print(f"✓ Archivo de resultados: {nombre_archivo}")
        
        try:
            # Usar la función importada de lectores.py
            (self.meta, self.files, self.time, self.temperatura, 
             self.Mr, self.Hc, self.campo_max, self.mag_max, 
             self.xi_M_0, self.frecuencia_fund, self.magnitud_fund, 
             self.dphi_fem, self.SAR, self.tau, self.N) = lector_resultados(self.ruta_resultados)
            
            # Ajustar tiempo para que empiece en 0 si hay datos
            if len(self.time) > 0:
                self.time = self.time - self.time[0]
            
            print(f"✓ Datos cargados: {len(self.files)} mediciones")
            
        except Exception as e:
            print(f"✗ Error en lector_resultados: {e}")
            raise
    
    def _cargar_ciclos_extremos(self):
        """
        Carga el primer y último ciclo de magnetización
        según el orden de self.files (resultados.txt).
        """

        if not os.path.exists(self.directorio_ciclos):
            print(f"⚠ Directorio de ciclos no encontrado: {self.directorio_ciclos}")
            self.primer_ciclo = None
            self.ultimo_ciclo = None
            return

        if not hasattr(self, 'files') or len(self.files) == 0:
            print("⚠ Lista de archivos vacía (files)")
            self.primer_ciclo = None
            self.ultimo_ciclo = None
            return

        def cargar_ciclo_desde_file(nombre_file):
            base = os.path.splitext(nombre_file)[0]
            nombre_ciclo = f"{base}_ciclo_H_M.txt"
            ruta = os.path.join(self.directorio_ciclos, nombre_ciclo)

            if not os.path.exists(ruta):
                raise FileNotFoundError(nombre_ciclo)

            t, H_Vs, M_Vs, H_kAm, M_Am, meta = lector_ciclos(ruta)
            return {
                'tiempo': t,
                'H_Vs': H_Vs,
                'M_Vs': M_Vs,
                'H_kAm': H_kAm,
                'M_Am': M_Am,
                'metadata': meta,
                'ruta_archivo': ruta
            }

        # ---------- PRIMER CICLO ----------
        try:
            self.primer_ciclo = cargar_ciclo_desde_file(self.files[0])
            self.primer_ciclo_metadata = self.primer_ciclo['metadata']
            print(f"✓ Primer ciclo (files[0]): {os.path.basename(self.primer_ciclo['ruta_archivo'])}")
        except Exception as e:
            print(f"✗ Error cargando primer ciclo: {e}")
            self.primer_ciclo = None

        # ---------- ÚLTIMO CICLO ----------
        try:
            self.ultimo_ciclo = cargar_ciclo_desde_file(self.files[-1])
            self.ultimo_ciclo_metadata = self.ultimo_ciclo['metadata']
            print(f"✓ Último ciclo (files[-1]): {os.path.basename(self.ultimo_ciclo['ruta_archivo'])}")
        except Exception as e:
            print(f"✗ Error cargando último ciclo: {e}")
            self.ultimo_ciclo = None

    
    def _calcular_estadisticas(self):
        """Calcula estadísticas básicas de los datos"""
        self.estadisticas = {}
        
        # Lista de atributos numéricos para calcular estadísticas
        atributos_numericos = [
            ('temperatura', 'Temperatura', '°C'),
            ('Hc', 'Campo coercitivo', 'kA/m'),
            ('SAR', 'SAR', 'W/g'),
            ('tau', 'tau', 'ns'),
            ('campo_max', 'Campo máximo', 'kA/m'),
            ('mag_max', 'Magnetización máxima', 'A/m'),
            ('frecuencia_fund', 'Frecuencia fundamental', 'Hz'),
            ('Mr', 'Magnetización remanente', 'A/m')]
        
        for attr, nombre, unidad in atributos_numericos:
            if hasattr(self, attr):
                datos = getattr(self, attr)
                if len(datos) > 0:
                    self.estadisticas[attr] = {
                        'nombre': nombre,
                        'unidad': unidad,
                        'media': float(np.nanmean(datos)),
                        'desviacion': float(np.nanstd(datos)),
                        'min': float(np.nanmin(datos)),
                        'max': float(np.nanmax(datos)),
                        'n_muestras': len(datos)}
    
    def _mostrar_resumen(self):
        """Muestra un resumen de los datos cargados"""
        print(f"\n{'='*60}")
        print(f"RESUMEN DE DATOS CARGADOS")
        print(f"{'='*60}")
        
        # Información básica
        if hasattr(self, 'meta') and 'Archivo_datos' in self.meta:
            print(f"Archivo de datos original: {self.meta['Archivo_datos']}")
        
        print(f"Número total de mediciones: {len(self.files) if hasattr(self, 'files') else 0}")
        
        # Rango de temperaturas
        if hasattr(self, 'temperatura') and len(self.temperatura) > 0:
            temp_min = np.min(self.temperatura)
            temp_max = np.max(self.temperatura)
            print(f"Rango de temperatura: {temp_min:.1f}°C - {temp_max:.1f}°C")
        
        # Valores promedio (si hay estadísticas calculadas)
        if hasattr(self, 'estadisticas'):
            if 'Hc' in self.estadisticas:
                hc_stats = self.estadisticas['Hc']
                print(f"Hc promedio: {hc_stats['media']:.2f} ± {hc_stats['desviacion']:.2f} {hc_stats['unidad']}")
            
            if 'SAR' in self.estadisticas:
                sar_stats = self.estadisticas['SAR']
                print(f"SAR promedio: {sar_stats['media']:.3f} ± {sar_stats['desviacion']:.3f} {sar_stats['unidad']}")
        
        print("Ciclos cargados:")
        print(f"  Directorio RT: {os.path.basename(self.directorio)}")
        print(f"  Primer ciclo -> file base: {self.files[0]}")
        print(f"  Último ciclo -> file base: {self.files[-1]}")

        
        print(f"{'='*60}")
    
    # ==================== MÉTODOS DE VISUALIZACIÓN ====================
    
    def plot_ciclos_comparacion(self, figsize=(8, 6),guardar=False):
        """
        Grafica ambos ciclos de magnetización para comparación.
        
        Parámetros:
        -----------
        figsize : tuple
            Tamaño de la figura (ancho, alto)
        guardar : bool o str
            Si es True, guarda en el directorio. Si es str, usa ese nombre.
        
        Retorna:
        --------
        fig, ax : matplotlib figure and axes
        """
        if not self.primer_ciclo or not self.ultimo_ciclo:
            print("⚠ No se pueden graficar ciclos: faltan datos")
            return None, None
        
        # Añadir información de temperatura
        temp_inicio = self.primer_ciclo['metadata'].get('Temperatura', 'N/A')
        temp_fin = self.ultimo_ciclo['metadata'].get('Temperatura', 'N/A')
                
        fig, ax = plt.subplots(figsize=figsize,constrained_layout=True)
        
        ax.plot(self.primer_ciclo['H_kAm'], self.primer_ciclo['M_Am'], 
                '-', linewidth=2.5,  
                label=f'Primer ciclo\nT$_i$: {temp_inicio}°C', zorder=3)
        
        ax.plot(self.ultimo_ciclo['H_kAm'], self.ultimo_ciclo['M_Am'], 
                '-', linewidth=2.5, 
                label=f'Ultimo ciclo\nT$_f$: {temp_fin}°C', zorder=2)
        
        # Configuración del gráfico
        ax.set_xlabel('H (kA/m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('M (A/m)', fontsize=12, fontweight='bold')
        
        # Título con información
        titulo = 'Primer vs ultimo ciclo de histeresis'
        if 'Archivo_datos' in self.meta:
            titulo += f"\n{os.path.basename(self.meta['Archivo_datos'])}"
        ax.set_title(titulo, fontsize=13, fontweight='bold')
        
        ax.grid(True)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.set_xlim(0,)
        # Ejes centrados en 0 si es apropiado
        x_lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
        y_lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
        ax.set_xlim(-x_lim * 1.1, x_lim * 1.1)
        ax.set_ylim(-y_lim * 1.1, y_lim * 1.1)

        # Guardar si se solicita
        if guardar:
            if isinstance(guardar, str):
                nombre_archivo = guardar
            else:
                nombre_base = os.path.basename(self.directorio)
                nombre_archivo = f"comparacion_ciclos.png"
            
            ruta_guardado = os.path.join(self.directorio, nombre_archivo)
            fig.savefig(ruta_guardado, dpi=300)
            print(f"✓ Gráfico guardado en: {ruta_guardado}")
        
        return fig, ax
    
    def plot_evolucion_temporal(self, figsize=(12, 10), guardar=False):
        """
        Grafica la evolución temporal de parámetros clave.
        
        Parámetros:
        -----------
        figsize : tuple
            Tamaño de la figura
        guardar : bool o str
            Si es True, guarda en el directorio. Si es str, usa ese nombre.
        
        Retorna:
        --------
        fig, axs : matplotlib figure and axes array
        """
        if not hasattr(self, 'time') or len(self.time) == 0:
            print("⚠ No hay datos temporales para graficar")
            return None, None
        
        fig, axs = plt.subplots(3, 2, figsize=figsize,constrained_layout=True,sharex=True)
        fig.suptitle('Evolución temporal de parámetros', fontsize=16, fontweight='bold')
        
        # Parámetros a graficar (título, atributo, unidad, color)
        parametros = [
            ('Temperatura', 'temperatura', '°C', 'tab:red'),
            ('Tau', 'tau', 'ns', 'tab:purple'),
            ('SAR', 'SAR', 'W/g', 'tab:green'),
            ('Campo Coercitivo Hc', 'Hc', 'kA/m', 'tab:blue'),
            ('Magnetizacion remanente', 'Mr', 'A/m', 'tab:orange'),
            ('Magnetización Máxima', 'mag_max', 'A/m', 'tab:orange'),]
        
        for idx, (titulo, attr, unidad, color) in enumerate(parametros):
            ax = axs[idx // 2, idx % 2]
            
            if hasattr(self, attr) and len(getattr(self, attr)) > 0:
                datos = getattr(self, attr)
                tiempo = self.time[:len(datos)]  # Asegurar misma longitud
                
                # Graficar datos
                ax.plot(tiempo, datos, 'o-', color=color, markersize=5,
                        label=f'<{attr}> ={ufloat(self.estadisticas[attr]["media"],self.estadisticas[attr]["desviacion"]):.1uS} {unidad}') 
                                       
                ax.set_ylabel(unidad, fontsize=10)
                ax.set_title(titulo, fontsize=11, loc='left',fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend()
                if idx >= 4:
                    ax.set_xlabel('t (s)', fontsize=10)

        # Guardar si se solicita
        if guardar:
            if isinstance(guardar, str):
                nombre_archivo = guardar
            else:
                nombre_base = os.path.basename(self.directorio)
                nombre_archivo = f"evolucion_temporal.png"
            
            ruta_guardado = os.path.join(self.directorio, nombre_archivo)
            fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado en: {ruta_guardado}")
        
        return fig, axs

# ==================== MÉTODO NUEVO: EVOLUCIÓN VS TEMPERATURA ====================
    def plot_evolucion_temperatura(self, figsize=(12, 8), guardar=False):
        """
        Grafica parámetros clave en función de la temperatura.
        
        Parámetros:
        -----------
        figsize : tuple
            Tamaño de la figura
        guardar : bool o str
            Si es True, guarda en el directorio. Si es str, usa ese nombre.
        
        Retorna:
        --------
        fig, axs : matplotlib figure and axes array
        """
        # Verificar que tenemos datos de temperatura
        if not hasattr(self, 'temperatura') or len(self.temperatura) == 0:
            print("⚠ No hay datos de temperatura para graficar")
            return None, None
        
        # Verificar que tenemos los parámetros requeridos
        parametros_requeridos = ['tau', 'SAR', 'Hc', 'Mr']
        for param in parametros_requeridos:
            if not hasattr(self, param) or len(getattr(self, param)) == 0:
                print(f"⚠ No hay datos de {param} para graficar")
                return None, None
        
        # Crear figura con subgráficos
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex=True)
        fig.suptitle('Evolución de parámetros vs Temperatura', fontsize=16, fontweight='bold')
        
        # Configuración de cada parámetro (título, atributo, unidad, color)
        parametros_config = [
            ('Tau', 'tau', 'ns', 'tab:purple'),
            ('SAR', 'SAR', 'W/g', 'tab:green'),
            ('Campo Coercitivo Hc', 'Hc', 'kA/m', 'tab:blue'),
            ('Magnetización Remanente', 'Mr', 'A/m', 'tab:orange')]
        
        # Obtener datos de temperatura para todos los puntos
        min_len = min(len(self.temperatura), 
                     len(self.tau), 
                     len(self.SAR), 
                     len(self.Hc), 
                     len(self.Mr))
        
        temp_vals = self.temperatura[:min_len]
        
        for idx, (titulo, attr, unidad, color) in enumerate(parametros_config):
            ax = axs[idx // 2, idx % 2]
            
            datos = getattr(self, attr)[:min_len]
            
            # Graficar datos vs temperatura
            ax.plot(temp_vals, datos, 'o-', color=color, markersize=6,
                   linewidth=2, alpha=0.8, label=f'{titulo}')
            
            # Agregar valor medio y desviación estándar como texto
            # if hasattr(self, 'estadisticas') and attr in self.estadisticas:
            #     stats = self.estadisticas[attr]
            #     ax.text(0.05, 0.95, 
            #            f"μ = {stats['media']:.2f} {unidad}\nσ = {stats['desviacion']:.2f} {unidad}",
            #            transform=ax.transAxes, fontsize=9,
            #            verticalalignment='top',
            #            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Configuración del gráfico
            ax.set_ylabel(unidad, fontsize=10, fontweight='bold')
            ax.set_title(titulo, fontsize=12, loc='left', fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='best', fontsize=9)
            
            # Si es la última fila, agregar etiqueta de temperatura
            if idx >= 2:
                ax.set_xlabel('Temperatura (°C)', fontsize=10, fontweight='bold')
        
        # Guardar si se solicita
        if guardar:
            if isinstance(guardar, str):
                nombre_archivo = guardar
            else:
                nombre_base = os.path.basename(self.directorio)
                nombre_archivo = f"evolucion_temperatura.png"
            
            ruta_guardado = os.path.join(self.directorio, nombre_archivo)
            fig.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
            print(f"✓ Gráfico guardado en: {ruta_guardado}")
        
        return fig, axs

    
    # ==================== MÉTODOS DE PERSISTENCIA ====================
    
    def guardar_completo(self, nombre_archivo=None, incluir_ciclos=True):
        """
        Guarda todos los datos en un archivo pickle.
        
        Parámetros:
        -----------
        nombre_archivo : str, opcional
            Nombre del archivo. Si es None, se genera automáticamente.
        incluir_ciclos : bool
            Si True, incluye los datos de ciclos en el guardado.
        
        Retorna:
        --------
        ruta_guardado : str
            Ruta completa del archivo guardado.
        """
        # Crear una copia para no modificar el objeto original
        objeto_guardar = self
        
        # Si no queremos incluir ciclos, crear una versión ligera
        if not incluir_ciclos:
            import copy
            objeto_guardar = copy.copy(self)
            objeto_guardar.primer_ciclo = None
            objeto_guardar.ultimo_ciclo = None
            objeto_guardar.primer_ciclo_metadata = None
            objeto_guardar.ultimo_ciclo_metadata = None
        
        # Generar nombre de archivo si no se proporciona
        if nombre_archivo is None:
            nombre_base = os.path.basename(self.directorio)
            if incluir_ciclos:
                nombre_archivo = f"{nombre_base}_completo.pkl"
            else:
                nombre_archivo = f"{nombre_base}_sin_ciclos.pkl"
        
        ruta_guardado = os.path.join(self.directorio, nombre_archivo)
        
        try:
            with open(ruta_guardado, 'wb') as f:
                pickle.dump(objeto_guardar, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            tamaño_mb = os.path.getsize(ruta_guardado) / (1024 * 1024)
            print(f"✓ Datos guardados en: {ruta_guardado}")
            print(f"  Tamaño: {tamaño_mb:.2f} MB")
            print(f"  Incluye ciclos: {'Sí' if incluir_ciclos else 'No'}")
            
            return ruta_guardado
            
        except Exception as e:
            print(f"✗ Error guardando datos: {e}")
            return None
    
    @classmethod
    def cargar_completo(cls, ruta_archivo_pkl):
        """
        Carga una instancia desde un archivo pickle.
        
        Parámetros:
        -----------
        ruta_archivo_pkl : str
            Ruta completa al archivo .pkl
        
        Retorna:
        --------
        instancia : ResultadosESAR
            Instancia cargada
        """
        if not os.path.exists(ruta_archivo_pkl):
            raise FileNotFoundError(f"Archivo no encontrado: {ruta_archivo_pkl}")
        
        try:
            with open(ruta_archivo_pkl, 'rb') as f:
                instancia = pickle.load(f)
            
            tamaño_mb = os.path.getsize(ruta_archivo_pkl) / (1024 * 1024)
            print(f"✓ Instancia cargada desde: {ruta_archivo_pkl}")
            print(f"  Tamaño: {tamaño_mb:.2f} MB")
            print(f"  Directorio origen: {instancia.directorio}")
            
            return instancia
            
        except Exception as e:
            print(f"✗ Error cargando archivo: {e}")
            raise
    
    # ==================== MÉTODOS DE INFORMACIÓN ====================
    
    def __repr__(self):
        """Representación formal de la instancia"""
        nombre_base = os.path.basename(self.directorio)
        n_mediciones = len(self.files) if hasattr(self, 'files') else 0
        return f"ResultadosESAR('{nombre_base}', mediciones={n_mediciones})"
    
    def __str__(self):
        """Representación informal (para print)"""
        nombre_base = os.path.basename(self.directorio)
        n_mediciones = len(self.files) if hasattr(self, 'files') else 0
        
        output = f"Resultados ESAR: {nombre_base}\n"
        output += "-" * 40 + "\n"
        output += f"Mediciones: {n_mediciones}\n"
        
        if hasattr(self, 'temperatura') and len(self.temperatura) > 0:
            output += f"Temperatura: {self.temperatura.min():.1f}°C a {self.temperatura.max():.1f}°C\n"
        
        if hasattr(self, 'estadisticas'):
            if 'Hc' in self.estadisticas:
                output += f"Hc promedio: {self.estadisticas['Hc']['media']:.2f} ± {self.estadisticas['Hc']['desviacion']:.2f} kA/m\n"
            if 'SAR' in self.estadisticas:
                output += f"SAR promedio: {self.estadisticas['SAR']['media']:.3f} ± {self.estadisticas['SAR']['desviacion']:.3f} W/g\n"
        
        output += f"Ciclos cargados: {'Primer' if self.primer_ciclo else 'No'} / {'Último' if self.ultimo_ciclo else 'No'}"
        
        return output
    
    def info(self):
        """Muestra información detallada de todos los atributos"""
        print(f"\n{'='*60}")
        print(f"INFORMACIÓN DETALLADA - {os.path.basename(self.directorio)}")
        print(f"{'='*60}")
        
        # Atributos principales
        print("\nATRIBUTOS PRINCIPALES:")
        print(f"- directorio: {self.directorio}")
        print(f"- ruta_resultados: {self.ruta_resultados}")
        
        # Metadatos
        if hasattr(self, 'meta'):
            print(f"\nMETADATOS ({len(self.meta)} items):")
            for key, value in list(self.meta.items())[:5]:  # Mostrar primeros 5
                print(f"  {key}: {value}")
            if len(self.meta) > 5:
                print(f"  ... y {len(self.meta) - 5} más")
        
        # Datos numéricos
        print(f"\nDATOS NUMÉRICOS:")
        atributos = [
            ('files', 'Archivos', 'str'),
            ('time', 'Tiempo', 'float'),
            ('temperatura', 'Temperatura', 'float'),
            ('Hc', 'Campo coercitivo', 'float'),
            ('SAR', 'SAR', 'float'),
            ('campo_max', 'Campo máximo', 'float'),
            ('mag_max', 'Magnetización máxima', 'float')
        ]
        
        for attr, nombre, tipo in atributos:
            if hasattr(self, attr):
                datos = getattr(self, attr)
                if hasattr(datos, '__len__'):
                    print(f"  {nombre}: {len(datos)} elementos [{tipo}]")
                    if len(datos) > 0 and hasattr(datos[0], '__len__'):
                        print(f"    Ejemplo: {datos[0][:3] if len(datos[0]) > 3 else datos[0]}...")
                    elif len(datos) > 0:
                        print(f"    Ejemplo: {datos[:3] if len(datos) > 3 else datos}...")
                else:
                    print(f"  {nombre}: {datos} [{tipo}]")
        
        # Estadísticas
        if hasattr(self, 'estadisticas') and self.estadisticas:
            print(f"\nESTADÍSTICAS:")
            for attr, stats in self.estadisticas.items():
                print(f"  {stats['nombre']}:")
                print(f"    Media: {stats['media']:.4f} {stats['unidad']}")
                print(f"    Desviación: {stats['desviacion']:.4f} {stats['unidad']}")
                print(f"    Rango: [{stats['min']:.4f}, {stats['max']:.4f}] {stats['unidad']}")
                print(f"    Muestras: {stats['n_muestras']}")
        
        # Ciclos
        print(f"\nCICLOS DE MAGNETIZACIÓN:")
        for nombre, ciclo in [('Primer ciclo', self.primer_ciclo), 
                            ('Último ciclo', self.ultimo_ciclo)]:
            if ciclo:
                n_puntos = len(ciclo['H_kAm'])
                temp = ciclo['metadata'].get('Temperatura', 'N/A')
                print(f"  {nombre}:")
                print(f"    Puntos: {n_puntos}")
                print(f"    Temperatura: {temp}°C")
                print(f"    Rango H: [{ciclo['H_kAm'].min():.2f}, {ciclo['H_kAm'].max():.2f}] kA/m")
                print(f"    Rango M: [{ciclo['M_Am'].min():.2f}, {ciclo['M_Am'].max():.2f}] A/m")
            else:
                print(f"  {nombre}: No cargado")
        
        print(f"\nMÉTODOS DISPONIBLES:")
        metodos = [m for m in dir(self) if not m.startswith('_') and callable(getattr(self, m))]
        metodos_principales = [m for m in metodos if m.startswith(('plot', 'guardar', 'info'))]
        print(f"  Visualización: {', '.join([m for m in metodos_principales if 'plot' in m])}")
        print(f"  Persistencia: {', '.join([m for m in metodos_principales if 'guardar' in m])}")
        
        print(f"{'='*60}\n")


        