import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

@st.cache_data
def cargar_y_limpiar_datos(url):
    """
    Carga y limpia los datos de deforestación desde un archivo CSV.
    Realiza interpolación para valores faltantes y convierte a un GeoDataFrame.

    Args:
        url (str): URL del archivo CSV con los datos.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos limpios e interpolados.
    """
    datos = pd.read_csv(url)

    datos.iloc[:, 3:] = datos.iloc[:, 3:].interpolate(method="linear", axis=0, limit_direction="both")

    modos_categoricos = datos.select_dtypes(include=["object"]).mode().iloc[0]
    datos[datos.select_dtypes(include=["object"]).columns] = datos[datos.select_dtypes(include=["object"]).columns].fillna(modos_categoricos)

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

def cargar_mapa_sudamerica(url_geopackage):
    """
    Carga un mapa de Sudamérica desde un GeoPackage y recorta la geometría.
    
    Args:
        url_geopackage (str): URL del GeoPackage con los datos de mapa base.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame con el mapa de Sudamérica.
    """
    mapa_base = gpd.read_file(url_geopackage)
    
    # Filtramos solo Sudamérica por su nombre en el campo 'name' (verifica que este campo exista en tu archivo)
    sudamerica = mapa_base[mapa_base["CONTINENT"] == "South America"]
    
    return sudamerica

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
    gdf_filtrado = gdf.copy()
    for columna, rango in filtros.items():
        if gdf[columna].dtype == "O":
            gdf_filtrado = gdf_filtrado[gdf_filtrado[columna].isin(rango)]
        else:
            gdf_filtrado = gdf_filtrado[(gdf_filtrado[columna] >= rango[0]) & (gdf_filtrado[columna] <= rango[1])]

    fig, ax = plt.subplots(figsize=(12, 8))
    mapa_base.plot(ax=ax, color="lightgrey", edgecolor="black")
    gdf_filtrado.plot(ax=ax, color="red", markersize=10, alpha=0.7)
    ax.set_title("Mapa Personalizado de Zonas Deforestadas", fontsize=16)
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    st.pyplot(fig)

# URL de los datos
url_datos = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/deforestacion.csv"
url_mapa_base = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# Cargar y limpiar los datos
gdf_datos_limpios = cargar_y_limpiar_datos(url_datos)

# Cargar el mapa base de Sudamérica
mapa_base = cargar_mapa_sudamerica(url_mapa_base)

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
