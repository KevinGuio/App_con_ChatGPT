import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

@st.cache_data
def cargar_y_limpiar_datos(url_o_archivo):
    """
    Carga y limpia los datos de deforestación desde un archivo CSV (puede ser URL o archivo local).
    Realiza interpolación para valores faltantes y convierte a un GeoDataFrame.

    Args:
        url_o_archivo (str or file): URL o archivo local con los datos CSV.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos limpios e interpolados.
    """
    if isinstance(url_o_archivo, str):
        datos = pd.read_csv(url_o_archivo)
    else:
        # Si se carga un archivo desde el dispositivo
        datos = pd.read_csv(url_o_archivo)

    # Interpolación para los valores numéricos
    datos.iloc[:, 3:] = datos.iloc[:, 3:].interpolate(method="linear", axis=0, limit_direction="both")

    # Rellenar valores nulos de tipo string con la moda
    modos_categoricos = datos.select_dtypes(include=["object"]).mode().iloc[0]
    datos[datos.select_dtypes(include=["object"]).columns] = datos[datos.select_dtypes(include=["object"]).columns].fillna(modos_categoricos)

    # Convertir a GeoDataFrame usando las columnas de latitud y longitud
    return gpd.GeoDataFrame(
        datos, geometry=gpd.points_from_xy(datos["Longitud"], datos["Latitud"])
    )

def cargar_datos():
    """
    Permite al usuario cargar un archivo CSV desde su dispositivo o ingresar una URL.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con los datos de deforestación.
    """
    st.title("Cargar Datos de Deforestación")

    # Opción para cargar archivo desde el dispositivo
    archivo_subido = st.file_uploader("Sube un archivo CSV", type="csv")

    # Opción para cargar desde una URL
    url_archivo = st.text_input("O ingresa la URL del archivo CSV")

    if archivo_subido:
        # Si el usuario sube un archivo
        st.write("Archivo cargado desde tu dispositivo")
        gdf = cargar_y_limpiar_datos(archivo_subido)
        return gdf

    elif url_archivo:
        # Si el usuario ingresa una URL
        st.write(f"Archivo cargado desde la URL: {url_archivo}")
        gdf = cargar_y_limpiar_datos(url_archivo)
        return gdf

    else:
        st.warning("Por favor, sube un archivo o ingresa una URL.")
        return None

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

def crear_grafico_torta(gdf):
    """
    Crea un gráfico de torta según el tipo de vegetación.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.

    Returns:
        None
    """
    distribucion_vegetacion = gdf["Tipo_Vegetacion"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 8))
    distribucion_vegetacion.plot.pie(autopct='%1.1f%%', startangle=90, ax=ax, cmap="Set3")
    ax.set_title("Distribución de Deforestación por Tipo de Vegetación")
    st.pyplot(fig)

def realizar_analisis_clusters(gdf):
    """
    Realiza un análisis de clústeres de las zonas deforestadas utilizando KMeans.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame con los datos de deforestación.

    Returns:
        None
    """
    # Asegurarse de que las columnas necesarias no tengan valores nulos
    datos_clustering = gdf[["Latitud", "Longitud", "Superficie_Deforestada"]].dropna()

    if datos_clustering.shape[0] == 0:
        st.warning("No hay suficientes datos no nulos para realizar el análisis de clústeres.")
        return

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(datos_clustering)

    # Asegurarse de que el índice de `gdf` y `datos_clustering` coincidan
    gdf = gdf.loc[datos_clustering.index].copy()  # Alineamos el índice

    # Asignar la columna "Cluster" a `gdf`
    gdf["Cluster"] = clusters

    # Visualizar los resultados del clustering
    fig, ax = plt.subplots(figsize=(12, 8))
    gdf.plot(ax=ax, column="Cluster", legend=True, cmap="viridis", markersize=10)
    ax.set_title("Análisis de Clústeres de Superficies Deforestadas")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    st.pyplot(fig)


# URL base del mapa mundial
url_mapa_base = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

# Cargar y limpiar los datos
gdf_datos_limpios = cargar_datos()
if gdf_datos_limpios is None:
    st.stop()  # Detiene la ejecución si no se cargaron los datos

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

# Gráfico de torta
crear_grafico_torta(gdf_datos_limpios)

# Análisis de clústeres
realizar_analisis_clusters(gdf_datos_limpios)
