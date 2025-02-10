import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from urllib.error import URLError
from shapely.geometry import Point
from shapely.ops import nearest_points

# URL fija del GeoPackage (Natural Earth) que contiene los países
# Se filtrará para mostrar solo Sudamérica en la función cargar_mapa_base.
URL_GEOPACKAGE = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"


@st.cache_data
def load_data(source_type, file=None, url=None):
    """Carga datos desde un archivo o URL.

    Args:
        source_type (str): 'file' para carga por archivo o 'url' para carga desde URL.
        file (UploadedFile, optional): Archivo subido por el usuario.
        url (str, optional): URL proporcionada por el usuario.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    if source_type == "file" and file is not None:
        return pd.read_csv(file)
    elif source_type == "url" and url is not None:
        return pd.read_csv(url)
    else:
        raise ValueError("Tipo de fuente inválido o entrada faltante.")


@st.cache_data
def fill_missing_values(df):
    """Rellena los valores faltantes del DataFrame usando binning para agrupación.

    Se crean bins para las columnas numéricas usadas como llaves de agrupación y
    se aplica agregación grupal para rellenar los valores faltantes.

    Args:
        df (pd.DataFrame): DataFrame de entrada con valores faltantes.

    Returns:
        pd.DataFrame: DataFrame con los valores faltantes rellenados.
    """
    df_filled = df.copy()

    # Rellenar columnas de texto
    df_filled["Nombre"] = df_filled["Nombre"].fillna("N/A")
    df_filled["Género"] = df_filled["Género"].fillna("No binario")

    # Crear bins para columnas numéricas
    df_filled["Ingreso_bin"] = pd.qcut(df_filled["Ingreso_Anual_USD"], q=4, duplicates="drop")
    df_filled["Historial_bin"] = pd.qcut(df_filled["Historial_Compras"], q=4, duplicates="drop")
    df_filled["Edad_bin"] = pd.qcut(df_filled["Edad"], q=4, duplicates="drop")

    # Para 'Frecuencia_Compra': si es numérica se usa qcut; de lo contrario se convierte a códigos
    df_filled["Frecuencia_bin"] = (
        pd.qcut(df_filled["Frecuencia_Compra"], q=4, duplicates="drop")
        if pd.api.types.is_numeric_dtype(df_filled["Frecuencia_Compra"])
        else df_filled["Frecuencia_Compra"].astype("category").cat.codes
    )

    # Imputar valores faltantes usando agrupaciones
    df_filled["Edad"] = df_filled["Edad"].fillna(
        df_filled.groupby(["Género", "Ingreso_bin", "Historial_bin", "Frecuencia_bin"])["Edad"]
        .transform("median")
    )
    df_filled["Ingreso_Anual_USD"] = df_filled["Ingreso_Anual_USD"].fillna(
        df_filled.groupby(["Edad_bin", "Género", "Historial_bin", "Frecuencia_bin"])["Ingreso_Anual_USD"]
        .transform("median")
    )
    df_filled["Historial_Compras"] = df_filled["Historial_Compras"].fillna(
        df_filled.groupby(["Edad_bin", "Género", "Ingreso_bin", "Frecuencia_bin"])["Historial_Compras"]
        .transform("median")
    )
    df_filled["Frecuencia_Compra"] = df_filled["Frecuencia_Compra"].fillna(
        df_filled.groupby(["Edad_bin", "Género", "Ingreso_bin", "Historial_bin"])["Frecuencia_Compra"]
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    df_filled["Latitud"] = df_filled["Latitud"].fillna(
        df_filled.groupby(["Edad_bin", "Género", "Ingreso_bin", "Historial_bin", "Frecuencia_bin"])["Latitud"]
        .transform("median")
    )
    df_filled["Longitud"] = df_filled["Longitud"].fillna(
        df_filled.groupby(["Edad_bin", "Género", "Ingreso_bin", "Historial_bin", "Frecuencia_bin"])["Longitud"]
        .transform("median")
    )

    # Eliminar columnas temporales de bins
    df_filled = df_filled.drop(columns=["Ingreso_bin", "Historial_bin", "Frecuencia_bin", "Edad_bin"])

    # Respaldo global para cualquier valor faltante restante
    df_filled = df_filled.fillna({
        "Edad": df["Edad"].median(),
        "Ingreso_Anual_USD": df["Ingreso_Anual_USD"].median(),
        "Historial_Compras": df["Historial_Compras"].median(),
        "Latitud": df["Latitud"].median(),
        "Longitud": df["Longitud"].median()
    })

    return df_filled


def cargar_mapa_base(url_geopackage):
    """
    Carga un mapa base mundial desde un GeoPackage y filtra para mostrar solo Centro y Suramérica.

    Args:
        url_geopackage (str): URL del GeoPackage con los datos de mapa base.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame con el mapa base filtrado a Centro y Suramérica, o None si ocurre un error.
    """
    try:
        mapa_base = gpd.read_file(url_geopackage)
        # Filtrar para incluir países cuyo continente sea "South America" 
        # o cuyo subregión sea "Central America".
        mapa_base = mapa_base[
            (mapa_base["CONTINENT"] == "South America") |
            (mapa_base["SUBREGION"] == "Central America")
        ]
        return mapa_base
    except URLError as e:
        st.error(f"Error al cargar el mapa base (URLError): {e}")
    except Exception as e:
        st.error(f"Error al cargar el mapa base: {e}")
    return None



def snap_to_land(point, land_gdf):
    """
    Dado un punto (shapely Point) y un GeoDataFrame con polígonos de tierra,
    devuelve el mismo punto si ya se encuentra sobre tierra;
    en caso contrario, retorna el punto de la tierra más cercano.

    Args:
        point (shapely.geometry.Point): Punto a evaluar.
        land_gdf (gpd.GeoDataFrame): GeoDataFrame con la geometría de la tierra.

    Returns:
        shapely.geometry.Point: Punto sobre tierra (original o corregido).
    """
    if land_gdf.contains(point).any():
        return point
    else:
        land_union = land_gdf.unary_union
        nearest_pt = nearest_points(point, land_union)[1]
        return nearest_pt


def corregir_coordenadas(df, land_gdf):
    """
    Corrige las coordenadas que caen en el mar moviéndolas al punto de tierra más cercano.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas 'Longitud' y 'Latitud'.
        land_gdf (gpd.GeoDataFrame): GeoDataFrame con los polígonos de tierra.

    Returns:
        pd.DataFrame: DataFrame con las coordenadas corregidas.
    """
    def snap_row(row):
        pt = Point(row["Longitud"], row["Latitud"])
        pt_corrected = snap_to_land(pt, land_gdf)
        return pt_corrected

    df["geometry"] = df.apply(snap_row, axis=1)
    # Usamos .apply para extraer las coordenadas de cada punto
    df["Longitud"] = df["geometry"].apply(lambda g: g.x)
    df["Latitud"] = df["geometry"].apply(lambda g: g.y)
    return df


def plot_on_basemap(df_points, title="Mapa de Clientes"):
    """Grafica las ubicaciones de los clientes sobre un mapa base cargado desde un GeoPackage.

    Args:
        df_points (pd.DataFrame): DataFrame con las columnas 'Latitud' y 'Longitud'.
        title (str, optional): Título del gráfico.
    """
    base_map = cargar_mapa_base(URL_GEOPACKAGE)
    if base_map is None:
        st.error("No se pudo cargar el mapa base. Verifica la URL o la conexión.")
        return
    # Convertir los datos de clientes a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points["Longitud"], df_points["Latitud"]),
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    base_map.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_points.plot(ax=ax, color="red", markersize=50)
    ax.set_title(title)
    st.pyplot(fig)


def plot_correlation_segmented(df):
    """Grafica la correlación entre Edad e Ingreso Anual segmentada por Género y Frecuencia de Compra.

    Utiliza seaborn lmplot para crear un FacetGrid donde cada panel corresponde a un valor de Frecuencia_Compra
    y los puntos se colorean según Género.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    g = sns.lmplot(
        data=df,
        x="Edad",
        y="Ingreso_Anual_USD",
        hue="Género",
        col="Frecuencia_Compra",
        aspect=1.2,
        height=4,
        markers="o",
        scatter_kws={'s': 50, 'alpha': 0.7},
        ci=None
    )
    g.fig.suptitle(
        "Correlación entre Edad e Ingreso Anual segmentado por Género y Frecuencia de Compra",
        y=1.05
    )
    st.pyplot(g.fig)


def analyze_correlation(df):
    """Analiza la correlación entre Edad e Ingreso Anual.

    Calcula la correlación global y la segmentada por Género y Frecuencia de Compra.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.

    Returns:
        dict: Diccionario con resultados de correlación.
    """
    global_corr = df["Edad"].corr(df["Ingreso_Anual_USD"])
    by_gender = df.groupby("Género").apply(lambda x: x["Edad"].corr(x["Ingreso_Anual_USD"]))
    by_frequency = df.groupby("Frecuencia_Compra").apply(lambda x: x["Edad"].corr(x["Ingreso_Anual_USD"]))
    return {"global": global_corr, "by_gender": by_gender, "by_frequency": by_frequency}


def map_global(df):
    """Muestra un mapa global de las ubicaciones de los clientes usando el mapa base de Sudamérica.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    st.subheader("Mapa Global de Clientes")
    plot_on_basemap(df, title="Clientes Globales")


def map_by_gender(df):
    """Muestra un mapa de clientes filtrado por Género usando el mapa base.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    gender = st.selectbox("Seleccione Género", df["Género"].unique())
    filtered = df[df["Género"] == gender]
    st.subheader(f"Mapa de Clientes - Género: {gender}")
    plot_on_basemap(filtered, title=f"Clientes - Género: {gender}")


def map_by_frequency(df):
    """Muestra un mapa de clientes filtrado por Frecuencia de Compra usando el mapa base.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    freq = st.selectbox("Seleccione Frecuencia de Compra", df["Frecuencia_Compra"].unique())
    filtered = df[df["Frecuencia_Compra"] == freq]
    st.subheader(f"Mapa de Clientes - Frecuencia de Compra: {freq}")
    plot_on_basemap(filtered, title=f"Clientes - Frecuencia: {freq}")


def custom_map(df):
    """Muestra un mapa personalizado basado en rangos de variables numéricas seleccionados por el usuario.

    El usuario puede seleccionar hasta 4 variables numéricas y definir su rango para filtrar los datos.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    st.subheader("Mapa Personalizado")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    selected = st.multiselect("Seleccione hasta 4 variables para filtrar", numeric_cols)
    filters = pd.Series(True, index=df.index)
    if selected:
        for col in selected:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            range_val = st.slider(
                f"Rango para {col}",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max)
            )
            filters &= df[col].between(range_val[0], range_val[1])
    filtered = df[filters]
    plot_on_basemap(filtered, title="Mapa Personalizado")
    st.dataframe(filtered)


def cluster_analysis(df):
    """Realiza el análisis de clúster basado en Frecuencia de Compra.

    Convierte 'Frecuencia_Compra' a numérico si es necesario, aplica KMeans y muestra los resultados.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.

    Returns:
        pd.DataFrame: DataFrame con una columna adicional 'cluster'.
    """
    df_cluster = df.copy()
    if not pd.api.types.is_numeric_dtype(df_cluster["Frecuencia_Compra"]):
        df_cluster["Frecuencia_Compra_num"] = df_cluster["Frecuencia_Compra"].astype("category").cat.codes
    else:
        df_cluster["Frecuencia_Compra_num"] = df_cluster["Frecuencia_Compra"]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster["cluster"] = kmeans.fit_predict(df_cluster[["Frecuencia_Compra_num"]])
    st.subheader("Análisis de Clúster")
    st.dataframe(df_cluster[["ID_Cliente", "Nombre", "Género", "Frecuencia_Compra", "cluster"]])
    return df_cluster


def plot_cluster_on_basemap(df):
    """Grafica los resultados del análisis de clúster sobre el mapa base de Sudamérica.

    Los puntos se colorean según el clúster al que pertenecen.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    df_cluster = cluster_analysis(df)
    base_map = cargar_mapa_base(URL_GEOPACKAGE)
    if base_map is None:
        st.error("No se pudo cargar el mapa base.")
        return
    gdf = gpd.GeoDataFrame(
        df_cluster,
        geometry=gpd.points_from_xy(df_cluster["Longitud"], df_cluster["Latitud"]),
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    base_map.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf.plot(ax=ax, column="cluster", cmap="viridis", markersize=50, legend=True)
    ax.set_title("Mapa de Clústeres basado en Frecuencia de Compra")
    st.pyplot(fig)


def bar_chart_gender_frequency(df):
    """Muestra un gráfico de barras segmentado por Género y Frecuencia de Compra.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    st.subheader("Gráfico de Barra por Género y Frecuencia de Compra")
    chart_data = df.groupby(["Género", "Frecuencia_Compra"]).size().unstack(fill_value=0)
    st.bar_chart(chart_data)


def heatmap_income(df):
    """Muestra un mapa de calor basado en Ingreso Anual usando pydeck.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    st.subheader("Mapa de Calor según Ingresos")
    heat_data = df[["Latitud", "Longitud", "Ingreso_Anual_USD"]].dropna()
    layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position="[Longitud, Latitud]",
        get_weight="Ingreso_Anual_USD",
        radius=50,
        aggregation=pdk.types.String("SUM")
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=heat_data["Latitud"].mean(),
            longitude=heat_data["Longitud"].mean(),
            zoom=4,
            pitch=50
        )
    )
    st.pydeck_chart(deck)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia de Haversine entre dos puntos en kilómetros.

    Args:
        lat1, lon1, lat2, lon2 (float o np.array): Coordenadas en grados.

    Returns:
        float o np.array: Distancia en kilómetros.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def distance_calculation(df):
    """Calcula las distancias entre el comprador con mayores ingresos global y grupos segmentados.

    Para cada Género y Frecuencia de Compra, calcula la distancia al comprador global.

    Args:
        df (pd.DataFrame): DataFrame con datos rellenados.
    """
    global_max = df.loc[df["Ingreso_Anual_USD"].idxmax()]
    by_gender = df.groupby("Género").apply(lambda x: x.loc[x["Ingreso_Anual_USD"].idxmax()]).reset_index(drop=True)
    by_frequency = df.groupby("Frecuencia_Compra").apply(lambda x: x.loc[x["Ingreso_Anual_USD"].idxmax()]).reset_index(drop=True)
    by_gender["distance_to_global"] = haversine_distance(
        global_max["Latitud"], global_max["Longitud"],
        by_gender["Latitud"], by_gender["Longitud"]
    )
    by_frequency["distance_to_global"] = haversine_distance(
        global_max["Latitud"], global_max["Longitud"],
        by_frequency["Latitud"], by_frequency["Longitud"]
    )
    st.subheader("Distancias (por Género) desde el comprador de mayores ingresos global")
    st.dataframe(by_gender[["Género", "Ingreso_Anual_USD", "distance_to_global"]])
    st.subheader("Distancias (por Frecuencia de Compra) desde el comprador de mayores ingresos global")
    st.dataframe(by_frequency[["Frecuencia_Compra", "Ingreso_Anual_USD", "distance_to_global"]])


def main():
    st.title("Análisis de Datos de Clientes")
    st.sidebar.header("Carga de datos")
    source_type = st.sidebar.radio("Seleccione la fuente de datos:", ("Archivo", "URL"))

    if source_type == "Archivo":
        file = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")
        if file:
            df = load_data("file", file=file)
    elif source_type == "URL":
        url = st.sidebar.text_input("Introduce la URL del archivo CSV")
        if url:
            df = load_data("url", url=url)

    if "df" in locals():
        st.subheader("Datos originales")
        st.dataframe(df)
        df_filled = fill_missing_values(df)
        st.subheader("Datos con valores faltantes rellenados")
        st.dataframe(df_filled)

        # Corregir coordenadas que caen en el mar (se corrige usando el mapa base de Sudamérica)
        land_gdf = cargar_mapa_base(URL_GEOPACKAGE)
        if land_gdf is not None:
            df_filled = corregir_coordenadas(df_filled, land_gdf)
            st.subheader("Datos con coordenadas corregidas")
            st.dataframe(df_filled)

        funcionalidad = st.sidebar.selectbox(
            "Seleccione funcionalidad:",
            [
                "Análisis de Correlación",
                "Mapa Global",
                "Mapa por Género",
                "Mapa por Frecuencia de Compra",
                "Mapa Personalizado",
                "Análisis de Clúster",
                "Grafica Clúster",
                "Gráfico de Barra",
                "Mapa de Calor",
                "Cálculo de Distancias"
            ]
        )

        if funcionalidad == "Análisis de Correlación":
            corr = analyze_correlation(df_filled)
            st.subheader("Correlaciones numéricas")
            st.write("Global:", corr["global"])
            st.write("Por Género:", corr["by_gender"])
            st.write("Por Frecuencia de Compra:", corr["by_frequency"])
            st.subheader("Gráfica de Correlación Segmentada")
            plot_correlation_segmented(df_filled)
        elif funcionalidad == "Mapa Global":
            map_global(df_filled)
        elif funcionalidad == "Mapa por Género":
            map_by_gender(df_filled)
        elif funcionalidad == "Mapa por Frecuencia de Compra":
            map_by_frequency(df_filled)
        elif funcionalidad == "Mapa Personalizado":
            custom_map(df_filled)
        elif funcionalidad == "Análisis de Clúster":
            cluster_analysis(df_filled)
        elif funcionalidad == "Grafica Clúster":
            plot_cluster_on_basemap(df_filled)
        elif funcionalidad == "Gráfico de Barra":
            bar_chart_gender_frequency(df_filled)
        elif funcionalidad == "Mapa de Calor":
            heatmap_income(df_filled)
        elif funcionalidad == "Cálculo de Distancias":
            distance_calculation(df_filled)


if __name__ == "__main__":
    main()
