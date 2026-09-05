from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Explorador automático de datos",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def leer_dataset(contenido: bytes, nombre_archivo: str) -> pd.DataFrame:
    """Lee un archivo cargado en memoria y normaliza solo los nombres de columnas."""
    extension = nombre_archivo.lower().rsplit(".", 1)[-1]
    buffer = BytesIO(contenido)
    if extension == "csv":
        intentos = (
            {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
            {"encoding": "utf-8", "sep": None, "engine": "python"},
            {"encoding": "latin-1", "sep": None, "engine": "python"},
        )
        ultimo_error = None
        for opciones in intentos:
            try:
                buffer.seek(0)
                df = pd.read_csv(buffer, **opciones)
                break
            except Exception as error:
                ultimo_error = error
        else:
            raise ValueError(f"No fue posible interpretar el CSV: {ultimo_error}")
    elif extension == "xlsx":
        df = pd.read_excel(buffer, engine="openpyxl")
    elif extension == "xls":
        df = pd.read_excel(buffer, engine="xlrd")
    else:
        raise ValueError("Formato no admitido. Use CSV, XLSX o XLS.")

    df.columns = [str(columna).strip() for columna in df.columns]
    if df.columns.duplicated().any():
        raise ValueError(
            "Después de quitar espacios, existen nombres de columnas duplicados. "
            "Renombre esas columnas en el archivo de origen."
        )
    return reconocer_fechas(df)


@st.cache_data(show_spinner=False)
def reconocer_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta convertir columnas cuyo nombre sugiere una fecha, sin forzar otras columnas."""
    resultado = df.copy()
    for columna in resultado.columns:
        nombre = columna.lower()
        if "fecha" in nombre or "date" in nombre:
            serie_original = resultado[columna]
            if pd.api.types.is_datetime64_any_dtype(serie_original):
                continue
            try:
                convertida = pd.to_datetime(serie_original, errors="coerce")
                no_nulos = int(serie_original.notna().sum())
                proporcion_valida = convertida.notna().sum() / no_nulos if no_nulos else 0
                if proporcion_valida >= 0.8:
                    resultado[columna] = convertida
            except (TypeError, ValueError, OverflowError):
                pass
    return resultado


def tipo_analitico(serie: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(serie):
        return "Booleana"
    if pd.api.types.is_datetime64_any_dtype(serie):
        return "Fecha/hora"
    if pd.api.types.is_numeric_dtype(serie):
        return "Numérica"
    if isinstance(serie.dtype, pd.CategoricalDtype):
        return "Categórica"
    no_nulos = serie.dropna()
    unicos = no_nulos.nunique()
    proporcion = unicos / len(no_nulos) if len(no_nulos) else 0
    longitud_media = no_nulos.astype(str).str.len().mean() if len(no_nulos) else 0
    return "Categórica" if unicos <= 30 or proporcion <= 0.2 or longitud_media <= 40 else "Texto"


def columnas_por_tipo(df: pd.DataFrame) -> dict[str, list[str]]:
    grupos = {"Numérica": [], "Categórica": [], "Texto": [], "Booleana": [], "Fecha/hora": []}
    for columna in df.columns:
        grupos[tipo_analitico(df[columna])].append(columna)
    return grupos


def resumen_tipos(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": df.columns,
            "Tipo de dato Pandas": [str(df[c].dtype) for c in df.columns],
            "Tipo analítico": [tipo_analitico(df[c]) for c in df.columns],
            "Valores no nulos": [int(df[c].notna().sum()) for c in df.columns],
            "Valores únicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    faltantes = df.isna().sum()
    porcentaje = faltantes.div(len(df)).mul(100) if len(df) else faltantes.astype(float)
    return (
        pd.DataFrame(
            {"Variable": df.columns, "Valores faltantes": faltantes.values, "Porcentaje faltante": porcentaje.values}
        )
        .sort_values(["Valores faltantes", "Variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def a_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def aplicar_filtros(df: pd.DataFrame) -> pd.DataFrame:
    filtrado = df.copy()
    tipos = columnas_por_tipo(df)
    st.sidebar.header("Filtros interactivos")

    if tipos["Fecha/hora"]:
        fechas_elegidas = st.sidebar.multiselect("Variables de fecha", tipos["Fecha/hora"])
        for columna in fechas_elegidas:
            validas = df[columna].dropna()
            if validas.empty:
                st.sidebar.info(f"{columna}: no contiene fechas válidas.")
                continue
            minimo, maximo = validas.min().date(), validas.max().date()
            rango = st.sidebar.date_input(
                f"Rango de {columna}", value=(minimo, maximo), min_value=minimo, max_value=maximo, key=f"fecha_{columna}"
            )
            if isinstance(rango, (tuple, list)) and len(rango) == 2:
                inicio, fin = pd.Timestamp(rango[0]), pd.Timestamp(rango[1]) + pd.Timedelta(days=1)
                filtrado = filtrado[filtrado[columna].isna() | ((filtrado[columna] >= inicio) & (filtrado[columna] < fin))]

    candidatas_cat = tipos["Categórica"] + tipos["Booleana"]
    if candidatas_cat:
        cat_elegidas = st.sidebar.multiselect("Variables categóricas", candidatas_cat)
        for columna in cat_elegidas:
            opciones = sorted(df[columna].dropna().unique().tolist(), key=lambda valor: str(valor))
            seleccion = st.sidebar.multiselect(f"Categorías de {columna}", opciones, default=opciones, key=f"cat_{columna}")
            filtrado = filtrado[filtrado[columna].isin(seleccion)]

    numericas_validas = [c for c in tipos["Numérica"] if df[c].dropna().size and np.isfinite(pd.to_numeric(df[c], errors="coerce").dropna()).any()]
    if numericas_validas:
        num_elegidas = st.sidebar.multiselect("Variables numéricas", numericas_validas)
        for columna in num_elegidas:
            serie = pd.to_numeric(df[columna], errors="coerce").replace([np.inf, -np.inf], np.nan)
            minimo, maximo = float(serie.min()), float(serie.max())
            if minimo == maximo:
                st.sidebar.caption(f"{columna}: valor constante {minimo:g}")
                continue
            rango = st.sidebar.slider(f"Rango de {columna}", minimo, maximo, (minimo, maximo), key=f"num_{columna}")
            filtrado = filtrado[filtrado[columna].isna() | filtrado[columna].between(rango[0], rango[1])]

    st.sidebar.metric("Registros resultantes", f"{len(filtrado):,}")
    return filtrado


def detectar_atipicos(df: pd.DataFrame, variables: Iterable[str], factor: float) -> pd.DataFrame:
    resultados = []
    for columna in variables:
        serie = pd.to_numeric(df[columna], errors="coerce").replace([np.inf, -np.inf], np.nan)
        validos = serie.dropna()
        if validos.empty:
            continue
        q1, q3 = validos.quantile([0.25, 0.75])
        iqr = q3 - q1
        inferior, superior = q1 - factor * iqr, q3 + factor * iqr
        mascara = serie.notna() & ((serie < inferior) | (serie > superior))
        if mascara.any():
            bloque = df.loc[mascara].copy()
            bloque.insert(0, "Fila original", bloque.index)
            bloque.insert(1, "Variable con valor atípico", columna)
            bloque.insert(2, "Valor atípico", serie.loc[mascara].values)
            bloque.insert(3, "Límite inferior", inferior)
            bloque.insert(4, "Límite superior", superior)
            resultados.append(bloque)
    columnas_extra = ["Fila original", "Variable con valor atípico", "Valor atípico", "Límite inferior", "Límite superior"]
    return pd.concat(resultados, ignore_index=True) if resultados else pd.DataFrame(columns=columnas_extra + list(df.columns))


st.title("📊 Explorador automático de datos")
st.write(
    "Carga un archivo y obtén un análisis exploratorio interactivo de su estructura, calidad, "
    "estadísticas, distribuciones, correlaciones y valores atípicos."
)
st.info(
    "Privacidad y uso responsable: los datos se procesan durante la sesión. Evita cargar información personal, "
    "confidencial o sensible. Este análisis no reemplaza la interpretación experta. Una correlación no implica "
    "causalidad y un valor atípico no necesariamente es un error."
)

archivo = st.sidebar.file_uploader("Cargar dataset", type=["csv", "xlsx", "xls"], help="Formatos admitidos: CSV, XLSX y XLS")

if archivo is None:
    st.info("Bienvenido. Para comenzar, carga un archivo CSV, XLSX o XLS desde la barra lateral.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("### 1. Cargar\nSelecciona tu archivo desde el computador.")
    c2.markdown("### 2. Explorar\nAplica filtros y revisa los análisis automáticos.")
    c3.markdown("### 3. Descargar\nExporta los datos filtrados y los valores atípicos.")
    st.markdown(
        """### Análisis disponibles
- Dimensiones, tipos analíticos e indicadores generales.
- Duplicados y valores faltantes.
- Estadísticas descriptivas.
- Distribuciones numéricas y categóricas.
- Correlaciones Pearson, Spearman y Kendall.
- Detección de valores atípicos mediante IQR.
- Filtros y tabla interactiva ordenable.
"""
    )
    st.stop()

try:
    df_original = leer_dataset(archivo.getvalue(), archivo.name)
except Exception as error:
    st.error(f"No se pudo procesar el archivo. Detalle: {error}")
    st.stop()

if df_original.empty or len(df_original.columns) == 0:
    st.warning("El archivo está vacío o no contiene una tabla utilizable.")
    st.stop()

st.sidebar.success(f"Archivo cargado: {archivo.name}")
df = aplicar_filtros(df_original)
if df.empty:
    st.warning("Los filtros no producen registros. Ajusta o elimina uno o más filtros para continuar.")
    st.stop()

st.caption(f"Archivo: **{archivo.name}** | Dimensiones filtradas: **{df.shape[0]:,} filas × {df.shape[1]:,} columnas**")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Filas", f"{df.shape[0]:,}")
m2.metric("Columnas", f"{df.shape[1]:,}")
m3.metric("Duplicados completos", f"{int(df.duplicated().sum()):,}")
m4.metric("Celdas faltantes", f"{int(df.isna().sum().sum()):,}")

st.download_button("Descargar datos filtrados", a_csv(df), "datos_filtrados.csv", "text/csv")

tipos = columnas_por_tipo(df)
tab_resumen, tab_calidad, tab_est, tab_dist, tab_corr, tab_atip, tab_tabla = st.tabs(
    ["Resumen y tipos", "Calidad de datos", "Estadísticas", "Distribuciones", "Correlaciones", "Valores atípicos", "Tabla ordenable"]
)

with tab_resumen:
    st.subheader("Dimensiones y tipos de variables")
    a, b, c = st.columns(3)
    a.metric("Cantidad de filas", f"{df.shape[0]:,}")
    b.metric("Cantidad de columnas", f"{df.shape[1]:,}")
    c.metric("Archivo", archivo.name)
    st.dataframe(resumen_tipos(df), use_container_width=True, hide_index=True)

with tab_calidad:
    st.subheader("Registros duplicados")
    duplicados = df[df.duplicated(keep=False)]
    if duplicados.empty:
        st.success("No existen registros completamente duplicados en los datos filtrados.")
    else:
        st.warning(f"Se encontraron {int(df.duplicated().sum()):,} duplicados adicionales y {len(duplicados):,} filas involucradas.")
        st.dataframe(duplicados, use_container_width=True)

    st.subheader("Valores faltantes")
    faltantes = resumen_faltantes(df)
    st.dataframe(
        faltantes.style.format({"Porcentaje faltante": "{:.2f}%"}), use_container_width=True, hide_index=True
    )
    fig_faltantes = px.bar(
        faltantes, x="Variable", y="Porcentaje faltante", title="Porcentaje de valores faltantes por variable", text_auto=".2f"
    )
    fig_faltantes.update_yaxes(range=[0, 100], title="Porcentaje (%)")
    st.plotly_chart(fig_faltantes, use_container_width=True)

with tab_est:
    st.subheader("Estadísticas descriptivas")
    opcion = st.radio("Variables que se incluirán", ["Todas las variables", "Solo variables numéricas", "Solo variables categóricas"], horizontal=True)
    nombres = {"count": "Conteo", "mean": "Media", "std": "Desviación estándar", "min": "Mínimo", "25%": "Primer cuartil", "50%": "Mediana", "75%": "Tercer cuartil", "max": "Máximo", "unique": "Valores únicos", "top": "Categoría más frecuente", "freq": "Frecuencia dominante"}
    try:
        if opcion == "Solo variables numéricas":
            if not tipos["Numérica"]:
                raise ValueError("El dataset filtrado no contiene variables numéricas.")
            estadisticas = df[tipos["Numérica"]].describe().T
        elif opcion == "Solo variables categóricas":
            categoricas = tipos["Categórica"] + tipos["Texto"] + tipos["Booleana"]
            if not categoricas:
                raise ValueError("El dataset filtrado no contiene variables categóricas o de texto.")
            estadisticas = df[categoricas].describe(include="all").T
        else:
            estadisticas = df.describe(include="all", datetime_is_numeric=True).T
        st.dataframe(estadisticas.rename(columns=nombres), use_container_width=True)
    except TypeError:
        # Compatibilidad con versiones de Pandas sin datetime_is_numeric.
        st.dataframe(df.describe(include="all").T.rename(columns=nombres), use_container_width=True)
    except Exception as error:
        st.info(str(error))

with tab_dist:
    st.subheader("Distribuciones")
    variable = st.selectbox("Selecciona una variable", df.columns)
    clase = tipo_analitico(df[variable])
    if clase == "Numérica":
        intervalos = st.slider("Número de intervalos", 5, 100, 30)
        fig_hist = px.histogram(df, x=variable, nbins=intervalos, title=f"Histograma de {variable}")
        st.plotly_chart(fig_hist, use_container_width=True)
        agrupadoras = ["Sin agrupación"] + tipos["Categórica"] + tipos["Booleana"]
        grupo = st.selectbox("Agrupar diagrama de caja por", agrupadoras)
        fig_caja = px.box(df, x=None if grupo == "Sin agrupación" else grupo, y=variable, points="outliers", title=f"Diagrama de caja de {variable}")
        st.plotly_chart(fig_caja, use_container_width=True)
    else:
        etiquetas = df[variable].astype("string").fillna("(Faltante)")
        frecuencias = etiquetas.value_counts(dropna=False).rename_axis("Categoría").reset_index(name="Frecuencia")
        if len(frecuencias) > 30:
            st.info("Se muestran las 30 categorías más frecuentes.")
            frecuencias = frecuencias.head(30)
        fig_cat = px.bar(frecuencias, x="Categoría", y="Frecuencia", title=f"Frecuencias de {variable}")
        st.plotly_chart(fig_cat, use_container_width=True)

with tab_corr:
    st.subheader("Correlaciones")
    if len(tipos["Numérica"]) < 2:
        st.info("Se necesitan al menos dos variables numéricas para calcular correlaciones.")
    else:
        seleccion_corr = st.multiselect("Variables numéricas", tipos["Numérica"], default=tipos["Numérica"])
        metodo = st.selectbox("Método", ["Pearson", "Spearman", "Kendall"])
        if len(seleccion_corr) < 2:
            st.warning("Selecciona al menos dos variables numéricas.")
        else:
            matriz = df[seleccion_corr].corr(method=metodo.lower())
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=matriz.values, x=matriz.columns, y=matriz.index, zmin=-1, zmax=1,
                    colorscale="RdBu", reversescale=True, text=np.round(matriz.values, 2), texttemplate="%{text}",
                    hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
                )
            )
            fig_corr.update_layout(title=f"Matriz de correlación de {metodo}")
            st.plotly_chart(fig_corr, use_container_width=True)
            st.dataframe(matriz.style.format("{:.3f}"), use_container_width=True)

with tab_atip:
    st.subheader("Valores atípicos mediante rango intercuartílico")
    if not tipos["Numérica"]:
        st.info("El dataset filtrado no contiene variables numéricas.")
        atipicos = pd.DataFrame()
    else:
        seleccion_atip = st.multiselect("Variables para analizar", tipos["Numérica"], default=tipos["Numérica"])
        factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1)
        atipicos = detectar_atipicos(df, seleccion_atip, factor)
        st.metric("Detecciones", f"{len(atipicos):,}")
        if atipicos.empty:
            st.success("No se detectaron valores atípicos con la configuración seleccionada.")
        else:
            conteo = atipicos["Variable con valor atípico"].value_counts().rename_axis("Variable").reset_index(name="Cantidad")
            st.plotly_chart(px.bar(conteo, x="Variable", y="Cantidad", title="Cantidad de atípicos por variable", text_auto=True), use_container_width=True)
            st.dataframe(atipicos, use_container_width=True, hide_index=True)
        st.download_button("Descargar valores atípicos", a_csv(atipicos), "valores_atipicos.csv", "text/csv")

with tab_tabla:
    st.subheader("Tabla interactiva y ordenable")
    visibles = st.multiselect("Columnas visibles", df.columns, default=list(df.columns))
    if not visibles:
        st.warning("Selecciona al menos una columna para visualizar la tabla.")
    else:
        st.dataframe(df[visibles], use_container_width=True, hide_index=True, height=520)
