import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

@st.cache_data
def cargar_y_limpiar_datos(url):
    """
    Carga y limpia los datos de deforestación desde un archivo CSV.
    Realiza interpolación para valores faltantes y convierte a un GeoDataFrame.

    Args:
        url (str): URL o ruta del archivo CSV con los datos.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos limpios e interpolados.
    """
    datos = pd.read_csv(url)
    
    # Interpolación para columnas numéricas y fechas
    for col in datos.columns:
        if pd.api.types.is_numeric_dtype(datos[col]):
            datos[col] = datos[col].interpolate(method='linear')
        elif pd.api.types.is_datetime64_any_dtype(datos[col]):
            datos[col] = pd.to_datetime(datos[col], errors='coerce')
            datos[col] = datos[col].interpolate(method='time')
    
    # Interpolación para columnas categóricas
    modos_categoricos = datos.select_dtypes(include=["object"]).mode().iloc[0]
    datos[datos.select_dtypes(include=["object"]).columns] = datos[datos.select_dtypes(include=["object"]).columns].fillna(modos_categoricos)
    
    # Crear GeoDataFrame
    return gpd.GeoDataFrame(
        datos, geometry=gpd.points_from_xy(datos["Longitud"], datos["Latitud"])
    )

@st.cache_data
def cargar_mapa_base(url_geopackage):
    """
    Carga un mapa base mundial desde un GeoPackage.

    Args:
        url_geopackage (str): URL del GeoPackage con los datos de mapa base.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con el mapa base.
    """
    mapa_base = gpd.read_file(url_geopackage)
    return mapa_base

def analizar_datos_deforestacion(gdf):
    """
    Realiza un análisis descriptivo de los datos de deforestación.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.

    Returns:
        dict: Resultados del análisis descriptivo, incluyendo estadísticas clave.
    """
    superficie_total = gdf["Superficie_Deforestada"].sum()
    tasa_promedio = gdf["Tasa_Deforestacion"].mean()
    distribucion_vegetacion = (
        gdf.groupby("Tipo_Vegetacion")["Superficie_Deforestada"]
        .sum()
        .sort_values(ascending=False)
    )
    resumen_estadisticas = gdf.describe()

    resultados = {
        "superficie_total_deforestada": superficie_total,
        "tasa_promedio_deforestacion": tasa_promedio,
        "distribucion_por_vegetacion": distribucion_vegetacion,
        "resumen_estadistico": resumen_estadisticas,
    }

    return resultados

def crear_mapa_deforestacion(gdf, columna, mapa_base, titulo):
    """
    Crea y muestra un mapa de las zonas deforestadas basado en una columna especificada.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.
        columna (str): Nombre de la columna para categorizar los datos en el mapa.
        mapa_base (gpd.GeoDataFrame): GeoDataFrame con el mapa base mundial.
        titulo (str): Título del mapa.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    mapa_base.plot(ax=ax, color="lightgrey", edgecolor="black")
    gdf.plot(ax=ax, column=columna, legend=True, markersize=10, cmap="viridis")
    ax.set_title(titulo, fontsize=16)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    st.pyplot(fig)

def crear_mapa_personalizado(gdf, mapa_base, filtros):
    """
    Crea y muestra un mapa personalizado basado en los filtros seleccionados por el usuario.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.
        mapa_base (gpd.GeoDataFrame): GeoDataFrame con el mapa base mundial.
        filtros (dict): Diccionario con las variables, sus rangos y/o categorías seleccionadas.

    Returns:
        None
    """
    # Aplicar filtros con vectorización
    gdf_filtrado = gdf
    for columna, rango in filtros.items():
        if gdf[columna].dtype == "O":
            gdf_filtrado = gdf_filtrado[gdf_filtrado[columna].isin(rango)]
        else:
            gdf_filtrado = gdf_filtrado[gdf_filtrado[columna].between(rango[0], rango[1])]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    mapa_base.plot(ax=ax, color="lightgrey", edgecolor="black")
    gdf_filtrado.plot(ax=ax, color="red", markersize=10, alpha=0.7)
    ax.set_title("Mapa Personalizado de Zonas Deforestadas", fontsize=16)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    st.pyplot(fig)

def realizar_analisis_clusters(gdf):
    """
    Realiza un análisis de clústeres de superficies deforestadas utilizando KMeans.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.

    Returns:
        pd.DataFrame: Datos con las etiquetas de clúster asignadas.
    """
    X = gdf[["Latitud", "Longitud", "Superficie_Deforestada"]].dropna()
    kmeans = KMeans(n_clusters=3, random_state=42)
    gdf["Cluster"] = kmeans.fit_predict(X)
    return gdf

def crear_grafico_torta(gdf):
    """
    Crea un gráfico de torta con la distribución de la superficie deforestada por tipo de vegetación.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.

    Returns:
        None
    """
    distribucion = gdf.groupby("Tipo_Vegetacion")["Superficie_Deforestada"].sum()
    fig, ax = plt.subplots()
    ax.pie(distribucion, labels=distribucion.index, autopct='%1.1f%%', startangle=90)
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

# URL de los datos
url_datos = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/deforestacion.csv"
url_mapa_base = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# Cargar y limpiar los datos
gdf_datos_limpios = cargar_y_limpiar_datos(url_datos)
mapa_base = cargar_mapa_base(url_mapa_base)

# Crear la app en Streamlit
st.title("Análisis de Datos de Deforestación")

# Análisis descriptivo
datos_analizados = analizar_datos_deforestacion(gdf_datos_limpios)

st.header("Análisis Descriptivo")
st.metric("Superficie Total Deforestada (ha)", f"{datos_analizados['superficie_total_deforestada']:.2f}")
st.metric("Tasa Promedio de Deforestación (%)", f"{datos_analizados['tasa_promedio_deforestacion']:.2f}")

st.subheader("Distribución de la Superficie Deforestada por Tipo de Vegetación")
st.dataframe(datos_analizados["distribucion_por_vegetacion"])

st.subheader("Resumen Estadístico")
st.dataframe(datos_analizados["resumen_estadistico"])

# Mapas de zonas deforestadas por diferentes variables
st.header("Mapas de zonas deforestadas")

st.subheader("Por tipo de vegetación")
crear_mapa_deforestacion(
    gdf_datos_limpios, "Tipo_Vegetacion", mapa_base, "Zonas Deforestadas por Tipo de Vegetación"
)

st.subheader("Por altitud")
crear_mapa_deforestacion(
    gdf_datos_limpios, "Altitud", mapa_base, "Zonas Deforestadas por Altitud"
)

st.subheader("Por precipitación")
crear_mapa_deforestacion(
    gdf_datos_limpios, "Precipitacion", mapa_base, "Zonas Deforestadas por Precipitación"
)

# Análisis de clústeres
st.header("Análisis de Clústeres de Superficies Deforestadas")
gdf_clusterizado = realizar_analisis_clusters(gdf_datos_limpios)
st.subheader("Zonas Deforestadas por Clúster")
crear_mapa_deforestacion(gdf_clusterizado, "Cluster", mapa_base, "Clústeres de Zonas Deforestadas")

# Gráfico de torta por tipo de vegetación
st.header("Distribución de la Superficie Deforestada por Tipo de Vegetación")
crear_grafico_torta(gdf_datos_limpios)

# Mapas personalizados
st.header("Mapa Personalizado de Zonas Deforestadas")

columnas = gdf_datos_limpios.columns.tolist()
filtros = {}

st.write("Selecciona hasta 4 variables para filtrar el mapa:")
for i in range(1, 5):
    columna = st.selectbox(f"Variable {i}", [None] + columnas, key=f"var_{i}")
    if columna:
        if gdf_datos_limpios[columna].dtype == "O":
            categorias = st.multiselect(f"Categorías para {columna}", gdf_datos_limpios[columna].unique(), key=f"cat_{i}")
            if categorias:
                filtros[columna] = categorias
        else:
            minimo, maximo = st.slider(
                f"Rango para {columna}",
                float(gdf_datos_limpios[columna].min()),
                float(gdf_datos_limpios[columna].max()),
                (float(gdf_datos_limpios[columna].min()), float(gdf_datos_limpios[columna].max())),
                key=f"rng_{i}"
            )
            filtros[columna] = (minimo, maximo)

if st.button("Generar Mapa Personalizado"):
    crear_mapa_personalizado(gdf_datos_limpios, mapa_base, filtros)
