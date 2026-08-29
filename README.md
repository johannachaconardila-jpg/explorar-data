# explorar-data
# Explorador automático de datos

Aplicación web creada con Streamlit para cargar archivos tabulares y ejecutar un análisis exploratorio de datos sin depender de un dataset predeterminado. El archivo se procesa en memoria durante la sesión.

## Funcionalidades

- Carga de CSV, XLSX y XLS.
- Reconocimiento sugerido de fechas a partir de nombres como `fecha` o `date`.
- Indicadores dinámicos de filas, columnas, duplicados y valores faltantes.
- Resumen de tipos Pandas y tipos analíticos.
- Revisión de duplicados y valores faltantes.
- Estadísticas descriptivas numéricas y categóricas.
- Histogramas, diagramas de caja y frecuencias con Plotly.
- Correlaciones Pearson, Spearman y Kendall.
- Detección de valores atípicos mediante IQR.
- Filtros interactivos por fecha, categoría y rango numérico.
- Tabla ordenable con selección de columnas.
- Descarga en CSV UTF-8 con BOM de datos filtrados y valores atípicos.

## Formatos admitidos

- `.csv`: lectura con Pandas y detección de separador.
- `.xlsx`: lectura mediante `openpyxl`.
- `.xls`: lectura mediante `xlrd`.

La aplicación lee la primera hoja de los libros de Excel.

## Estructura del repositorio

```text
explorador-automatico-datos/
├── app.py
├── requirements.txt
└── README.md
```

No se incluye ningún dataset.

## Instalación

Se recomienda Python 3.11 o 3.12.

```bash
git clone https://github.com/TU_USUARIO/explorador-automatico-datos.git
cd explorador-automatico-datos
python -m venv .venv
```

Activa el entorno virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS o Linux
source .venv/bin/activate
```

Instala las dependencias:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución local

```bash
streamlit run app.py
```

Abre la dirección local mostrada por Streamlit, normalmente `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Publica los tres archivos en un repositorio de GitHub.
2. Ingresa a [Streamlit Community Cloud](https://share.streamlit.io/) con GitHub.
3. Selecciona **Create app**.
4. Elige el repositorio, la rama y `app.py` como archivo principal.
5. En configuración avanzada, selecciona una versión compatible de Python, por ejemplo 3.12.
6. Haz clic en **Deploy** y revisa los registros si aparece un error.

La aplicación no necesita secretos, claves ni variables de entorno.

## Privacidad y uso responsable

Los datos se procesan durante la sesión de la aplicación. Evita cargar información personal, confidencial o sensible. El análisis es exploratorio y no reemplaza el criterio de una persona experta. Una correlación no implica causalidad y un valor atípico no necesariamente representa un error.

## Limitaciones conocidas

- Se analiza la primera hoja de cada archivo Excel.
- Los CSV con estructuras irregulares, filas dañadas o codificaciones no contempladas pueden requerir limpieza previa.
- La detección de fechas se intenta solo cuando el nombre contiene `fecha` o `date`, y exige una proporción suficiente de valores interpretables.
- Archivos muy grandes pueden superar la memoria o los límites de carga del entorno de despliegue.
- Las columnas de texto con alta cardinalidad no se tratan como categóricas para evitar filtros poco prácticos.
- El método IQR puede no ser apropiado para todas las distribuciones o áreas de conocimiento.
