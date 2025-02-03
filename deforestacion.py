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


# URL de los datos
url_datos = "https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/aplicaciones/deforestacion.csv"

# Cargar y limpiar los datos
gdf_datos_limpios = cargar_y_limpiar_datos(url_datos)

# Crear la app en Streamlit
st.title("Análisis de Datos de Deforestación")

# Mostrar resultados del análisis
st.header("Resultados del análisis descriptivo")
resultados = analizar_datos_deforestacion(gdf_datos_limpios)

st.metric("Superficie Total Deforestada (ha)", f"{resultados['superficie_total_deforestada']:.2f}")
st.metric("Tasa Promedio de Deforestación (%)", f"{resultados['tasa_promedio_deforestacion']:.2f}")

st.subheader("Distribución de la Superficie Deforestada por Tipo de Vegetación")
st.bar_chart(resultados["distribucion_por_vegetacion"])

st.subheader("Resumen Estadístico")
st.dataframe(resultados["resumen_estadistico"])

# Gráficos adicionales
st.subheader("Gráfico de Distribución por Tipo de Vegetación")
fig, ax = plt.subplots()
resultados["distribucion_por_vegetacion"].plot(kind="pie", autopct='%1.1f%%', ax=ax, startangle=90)
ax.set_ylabel("")
ax.set_title("Distribución por Tipo de Vegetación")
st.pyplot(fig)
