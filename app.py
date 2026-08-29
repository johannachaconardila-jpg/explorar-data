"""Aplicacion Streamlit para analisis exploratorio automatico de datos."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Explorador automatico de datos",
    page_icon="📊",
    layout="wide",
)

DATE_HINTS = ("fecha", "date")
MISSING_LABEL = "(Faltante)"


def make_unique_column_names(columns: Iterable[object]) -> list[str]:
    """Limpia espacios y evita nombres repetidos despues de la limpieza."""
    result: list[str] = []
    counts: dict[str, int] = {}
    for position, column in enumerate(columns, start=1):
        base = str(column).strip() or f"columna_{position}"
        counts[base] = counts.get(base, 0) + 1
        result.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return result


def recognize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas sugeridas por su nombre solo cuando hay fechas validas."""
    result = df.copy()
    for column in result.columns:
        if any(hint in column.casefold() for hint in DATE_HINTS):
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                continue
            try:
                converted = pd.to_datetime(result[column], errors="coerce", format="mixed")
            except (TypeError, ValueError):
                converted = pd.to_datetime(result[column], errors="coerce")
            non_null_original = int(result[column].notna().sum())
            valid_ratio = converted.notna().sum() / non_null_original if non_null_original else 0
            if valid_ratio >= 0.60:
                result[column] = converted
    return result


@st.cache_data(show_spinner=False)
def read_dataset(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Lee CSV o Excel desde memoria; no escribe el archivo cargado en disco."""
    extension = Path(file_name).suffix.lower()
    if extension == ".csv":
        last_error: Exception | None = None
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                return pd.read_csv(
                    BytesIO(file_bytes), encoding=encoding, sep=None, engine="python"
                )
            except (UnicodeDecodeError, pd.errors.ParserError) as error:
                last_error = error
        raise ValueError(f"No fue posible interpretar el CSV: {last_error}")
    if extension == ".xlsx":
        return pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    if extension == ".xls":
        return pd.read_excel(BytesIO(file_bytes), engine="xlrd")
    raise ValueError("Formato no admitido. Use CSV, XLSX o XLS.")


def analytical_type(series: pd.Series) -> str:
    """Asigna un tipo analitico sin alterar los valores."""
    if pd.api.types.is_bool_dtype(series):
        return "Booleana"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "Fecha/hora"
    if pd.api.types.is_numeric_dtype(series):
        return "Numérica"
    non_null = series.dropna()
    if pd.api.types.is_categorical_dtype(series):
        return "Categórica"
    if non_null.empty:
        return "Texto"
    unique_ratio = non_null.nunique(dropna=True) / len(non_null)
    mean_length = non_null.astype(str).str.len().mean()
    return "Texto" if unique_ratio > 0.50 or mean_length > 40 else "Categórica"


def type_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Variable": df.columns,
            "Tipo Pandas": [str(df[c].dtype) for c in df.columns],
            "Tipo analítico": [analytical_type(df[c]) for c in df.columns],
            "Valores no nulos": [int(df[c].notna().sum()) for c in df.columns],
            "Valores únicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )


def columns_by_kind(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    dates = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical = [
        c for c in df.columns if analytical_type(df[c]) in ("Categórica", "Booleana")
    ]
    return numeric, categorical, dates


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    denominator = len(df)
    percentage = missing.div(denominator).mul(100) if denominator else missing.astype(float)
    return (
        pd.DataFrame(
            {
                "Variable": df.columns,
                "Valores faltantes": missing.values.astype(int),
                "Porcentaje faltante": percentage.values,
            }
        )
        .sort_values(["Valores faltantes", "Variable"], ascending=[False, True])
        .reset_index(drop=True)
    )


def numeric_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    stats = df[columns].describe().T.rename(
        columns={
            "count": "Conteo", "mean": "Media", "std": "Desviación estándar",
            "min": "Mínimo", "25%": "Primer cuartil", "50%": "Mediana",
            "75%": "Tercer cuartil", "max": "Máximo",
        }
    )
    stats.index.name = "Variable"
    return stats.reset_index()


def categorical_statistics(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = df[column].dropna()
        counts = series.value_counts(dropna=True)
        rows.append(
            {
                "Variable": column,
                "Conteo": int(series.count()),
                "Valores únicos": int(series.nunique()),
                "Categoría más frecuente": counts.index[0] if not counts.empty else np.nan,
                "Frecuencia dominante": int(counts.iloc[0]) if not counts.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def find_outliers(df: pd.DataFrame, columns: list[str], factor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve detalle por variable y resumen de detecciones IQR."""
    parts: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for column in columns:
        values = df[column].dropna()
        if values.empty:
            lower = upper = np.nan
            mask = pd.Series(False, index=df.index)
        else:
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            mask = df[column].notna() & ((df[column] < lower) | (df[column] > upper))
        count = int(mask.sum())
        summary.append({"Variable": column, "Cantidad de atípicos": count})
        if count:
            part = df.loc[mask].copy()
            part.insert(0, "Fila original", part.index)
            part.insert(1, "Variable con valor atípico", column)
            part.insert(2, "Valor atípico", df.loc[mask, column].values)
            part.insert(3, "Límite inferior", lower)
            part.insert(4, "Límite superior", upper)
            parts.append(part)
    detail_columns = [
        "Fila original", "Variable con valor atípico", "Valor atípico",
        "Límite inferior", "Límite superior", *df.columns,
    ]
    detail = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=detail_columns)
    return detail, pd.DataFrame(summary)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Construye filtros laterales y conserva faltantes en fechas y numeros."""
    filtered = df.copy()
    numeric, categorical, dates = columns_by_kind(df)
    st.sidebar.header("Filtros interactivos")

    if dates:
        with st.sidebar.expander("Filtros por fecha"):
            selected_dates = st.multiselect("Variables de fecha", dates, key="date_filter_cols")
            for column in selected_dates:
                valid = df[column].dropna()
                if valid.empty:
                    st.caption(f"{column}: sin fechas válidas.")
                    continue
                min_date, max_date = valid.min().date(), valid.max().date()
                start, end = st.date_input(
                    f"Rango de {column}", value=(min_date, max_date),
                    min_value=min_date, max_value=max_date, key=f"date_{column}",
                )
                if start > end:
                    start, end = end, start
                normalized = filtered[column].dt.date
                filtered = filtered[filtered[column].isna() | normalized.between(start, end)]

    if categorical:
        with st.sidebar.expander("Filtros categóricos"):
            chosen = st.multiselect("Variables categóricas", categorical, key="cat_filter_cols")
            for column in chosen:
                options = df[column].dropna().unique().tolist()
                options = sorted(options, key=lambda value: str(value))
                selected = st.multiselect(
                    f"Categorías de {column}", options, default=options, key=f"cat_{column}"
                )
                include_missing = st.checkbox(
                    f"Incluir faltantes en {column}", value=True, key=f"cat_na_{column}"
                )
                mask = filtered[column].isin(selected)
                if include_missing:
                    mask |= filtered[column].isna()
                filtered = filtered[mask]

    if numeric:
        with st.sidebar.expander("Filtros numéricos"):
            chosen = st.multiselect("Variables numéricas", numeric, key="num_filter_cols")
            for column in chosen:
                valid = df[column].dropna()
                if valid.empty:
                    st.caption(f"{column}: sin valores numéricos válidos.")
                    continue
                minimum, maximum = float(valid.min()), float(valid.max())
                if np.isclose(minimum, maximum):
                    st.caption(f"{column}: valor constante ({minimum:g}).")
                    continue
                selected_min, selected_max = st.slider(
                    f"Rango de {column}", minimum, maximum, (minimum, maximum),
                    key=f"num_{column}",
                )
                filtered = filtered[
                    filtered[column].isna() | filtered[column].between(selected_min, selected_max)
                ]
    return filtered


st.title("📊 Explorador automático de datos")
st.write(
    "Carga un archivo para obtener un análisis exploratorio interactivo, aplicar filtros "
    "y descargar resultados sin depender de un conjunto de datos predeterminado."
)
st.info(
    "**Uso responsable:** los datos se procesan durante la sesión. Evita cargar información "
    "personal, confidencial o sensible. Este análisis no reemplaza la interpretación experta. "
    "Una correlación no implica causalidad y un valor atípico no necesariamente es un error."
)

uploaded_file = st.sidebar.file_uploader(
    "Cargar conjunto de datos", type=["csv", "xlsx", "xls"],
    help="Formatos admitidos: CSV, XLSX y XLS.",
)

if uploaded_file is None:
    st.info("Para comenzar, carga un archivo desde la barra lateral.")
    left, right = st.columns(2)
    with left:
        st.subheader("Etapas de uso")
        st.markdown("1. **Cargar** un CSV, XLSX o XLS.\n2. **Explorar** las pestañas y aplicar filtros.\n3. **Descargar** los resultados.")
    with right:
        st.subheader("Análisis disponibles")
        st.markdown(
            "- Dimensiones, tipos e indicadores\n- Duplicados y valores faltantes\n"
            "- Estadísticas y distribuciones\n- Correlaciones y valores atípicos\n"
            "- Filtros, tabla y descargas"
        )
    st.warning("No se generan datos ficticios. El análisis empieza únicamente con tu archivo.")
    st.stop()

try:
    raw_df = read_dataset(uploaded_file.getvalue(), uploaded_file.name)
except Exception as error:
    st.error(
        "No fue posible procesar el archivo. Verifica su formato, extensión, codificación "
        f"y que no esté dañado. Detalle: {error}"
    )
    st.stop()

if raw_df.empty or raw_df.shape[1] == 0:
    st.warning("El archivo está vacío o no contiene columnas utilizables.")
    st.stop()

raw_df.columns = make_unique_column_names(raw_df.columns)
raw_df = recognize_dates(raw_df)
st.sidebar.success(f"Archivo cargado: {uploaded_file.name}")
filtered_df = apply_filters(raw_df)
st.sidebar.metric("Registros resultantes", len(filtered_df))

if filtered_df.empty:
    st.warning("Los filtros no producen registros. Ajusta o restablece los filtros para continuar.")
    st.stop()

numeric_columns, categorical_columns, date_columns = columns_by_kind(filtered_df)
rows, columns = filtered_df.shape
duplicates_count = int(filtered_df.duplicated().sum())
missing_count = int(filtered_df.isna().sum().sum())

metric_columns = st.columns(4)
metric_columns[0].metric("Filas", rows)
metric_columns[1].metric("Columnas", columns)
metric_columns[2].metric("Duplicados completos", duplicates_count)
metric_columns[3].metric("Celdas faltantes", missing_count)
st.caption(f"Archivo: **{uploaded_file.name}** | Dimensiones filtradas: **{rows} filas × {columns} columnas**")
st.sidebar.download_button(
    "Descargar datos filtrados", to_csv_bytes(filtered_df), "datos_filtrados.csv",
    "text/csv", use_container_width=True,
)

tabs = st.tabs([
    "Resumen y tipos", "Calidad de datos", "Estadísticas", "Distribuciones",
    "Correlaciones", "Valores atípicos", "Tabla ordenable",
])

with tabs[0]:
    st.subheader("Dimensiones y tipos de variables")
    st.write(f"El conjunto filtrado contiene **{rows} filas** y **{columns} columnas**.")
    st.dataframe(type_summary(filtered_df), hide_index=True, use_container_width=True)

with tabs[1]:
    st.subheader("Registros duplicados")
    duplicated_rows = filtered_df[filtered_df.duplicated(keep=False)]
    if duplicated_rows.empty:
        st.success("No se encontraron registros completamente duplicados.")
    else:
        st.warning(f"Hay {duplicates_count} duplicados adicionales y {len(duplicated_rows)} filas involucradas.")
        st.dataframe(duplicated_rows, use_container_width=True)

    st.subheader("Valores faltantes")
    missing = missing_summary(filtered_df)
    st.dataframe(
        missing.style.format({"Porcentaje faltante": "{:.2f}%"}),
        hide_index=True, use_container_width=True,
    )
    missing_chart = missing[missing["Valores faltantes"] > 0]
    if missing_chart.empty:
        st.success("No se encontraron valores faltantes.")
    else:
        figure = px.bar(
            missing_chart, x="Porcentaje faltante", y="Variable", orientation="h",
            title="Porcentaje de valores faltantes por variable",
            labels={"Porcentaje faltante": "Porcentaje (%)"},
        )
        figure.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(figure, use_container_width=True)

with tabs[2]:
    st.subheader("Estadísticas descriptivas")
    scope = st.selectbox(
        "Variables a resumir", ["Todas las variables", "Solo variables numéricas", "Solo variables categóricas"]
    )
    try:
        if scope in ("Todas las variables", "Solo variables numéricas"):
            st.markdown("#### Variables numéricas")
            if not numeric_columns:
                st.info("El conjunto filtrado no tiene variables numéricas.")
            else:
                st.dataframe(numeric_statistics(filtered_df, numeric_columns), hide_index=True, use_container_width=True)
        if scope in ("Todas las variables", "Solo variables categóricas"):
            st.markdown("#### Variables categóricas")
            if not categorical_columns:
                st.info("El conjunto filtrado no tiene variables categóricas.")
            else:
                st.dataframe(categorical_statistics(filtered_df, categorical_columns), hide_index=True, use_container_width=True)
    except (TypeError, ValueError, IndexError) as error:
        st.warning(f"No fue posible calcular el resumen seleccionado: {error}")

with tabs[3]:
    st.subheader("Distribuciones")
    distribution_columns = numeric_columns + [c for c in categorical_columns if c not in numeric_columns]
    if not distribution_columns:
        st.info("No hay variables numéricas o categóricas disponibles.")
    else:
        selected_variable = st.selectbox("Variable", distribution_columns)
        if selected_variable in numeric_columns:
            bins = st.slider("Número de intervalos", 5, 100, 30)
            histogram = px.histogram(
                filtered_df, x=selected_variable, nbins=bins,
                title=f"Histograma de {selected_variable}", marginal=None,
            )
            st.plotly_chart(histogram, use_container_width=True)
            group_options = ["Sin agrupar"] + categorical_columns
            group = st.selectbox("Agrupar diagrama de caja por", group_options)
            box = px.box(
                filtered_df, x=None if group == "Sin agrupar" else group,
                y=selected_variable, points="outliers",
                title=f"Diagrama de caja de {selected_variable}",
            )
            st.plotly_chart(box, use_container_width=True)
        else:
            frequencies = (
                filtered_df[selected_variable].astype("object").where(
                    filtered_df[selected_variable].notna(), MISSING_LABEL
                ).value_counts(dropna=False).rename_axis("Categoría").reset_index(name="Frecuencia")
            )
            if len(frequencies) > 30:
                st.info("Se muestran las 30 categorías más frecuentes.")
                frequencies = frequencies.head(30)
            frequencies["Categoría"] = frequencies["Categoría"].astype(str)
            chart = px.bar(
                frequencies, x="Categoría", y="Frecuencia",
                title=f"Frecuencias de {selected_variable}",
            )
            st.plotly_chart(chart, use_container_width=True)

with tabs[4]:
    st.subheader("Correlaciones")
    if len(numeric_columns) < 2:
        st.info("Se necesitan al menos dos variables numéricas para calcular correlaciones.")
    else:
        selected_corr = st.multiselect(
            "Variables numéricas", numeric_columns, default=numeric_columns
        )
        method_label = st.selectbox("Método", ["Pearson", "Spearman", "Kendall"])
        if len(selected_corr) < 2:
            st.warning("Selecciona al menos dos variables numéricas.")
        else:
            correlation = filtered_df[selected_corr].corr(method=method_label.lower())
            heatmap = go.Figure(
                data=go.Heatmap(
                    z=correlation.values, x=correlation.columns, y=correlation.index,
                    zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                    text=np.round(correlation.values, 2), texttemplate="%{text}",
                    hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                )
            )
            heatmap.update_layout(title=f"Correlación de {method_label}")
            st.plotly_chart(heatmap, use_container_width=True)
            st.dataframe(correlation.style.format("{:.3f}"), use_container_width=True)
            st.caption("Recuerda: una correlación no implica causalidad.")

with tabs[5]:
    st.subheader("Detección de valores atípicos por IQR")
    if not numeric_columns:
        st.info("No hay variables numéricas para analizar.")
    else:
        selected_outliers = st.multiselect(
            "Variables numéricas", numeric_columns, default=numeric_columns
        )
        factor = st.slider("Factor IQR", 1.0, 3.0, 1.5, 0.1)
        if not selected_outliers:
            st.warning("Selecciona al menos una variable numérica.")
        else:
            outlier_detail, outlier_summary = find_outliers(filtered_df, selected_outliers, factor)
            st.metric("Detecciones de valores atípicos", len(outlier_detail))
            outlier_chart = px.bar(
                outlier_summary, x="Variable", y="Cantidad de atípicos",
                title="Cantidad de atípicos por variable",
            )
            st.plotly_chart(outlier_chart, use_container_width=True)
            if outlier_detail.empty:
                st.success("No se detectaron valores atípicos con la configuración actual.")
            else:
                st.dataframe(outlier_detail, hide_index=True, use_container_width=True)
            st.download_button(
                "Descargar valores atípicos", to_csv_bytes(outlier_detail),
                "valores_atipicos.csv", "text/csv", use_container_width=True,
            )
            st.caption("Una fila puede aparecer varias veces si es atípica en varias variables.")

with tabs[6]:
    st.subheader("Tabla interactiva y ordenable")
    visible_columns = st.multiselect(
        "Columnas visibles", filtered_df.columns.tolist(), default=filtered_df.columns.tolist()
    )
    if not visible_columns:
        st.info("Selecciona al menos una columna para visualizar la tabla.")
    else:
        st.dataframe(
            filtered_df[visible_columns], hide_index=True, use_container_width=True,
            height=550,
        )
