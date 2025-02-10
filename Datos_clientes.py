import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import pydeck as pdk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# URL fija del GeoPackage (reemplaza por la URL deseada)
URL_GEOPACKAGE = "https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip"

@st.cache_data
def load_data(source_type, file=None, url=None):
    """Load data from a file or URL.

    Args:
        source_type (str): 'file' for file upload or 'url' for URL source.
        file (UploadedFile, optional): File uploaded by the user.
        url (str, optional): URL provided by the user.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    if source_type == 'file' and file is not None:
        return pd.read_csv(file)
    elif source_type == 'url' and url is not None:
        return pd.read_csv(url)
    else:
        raise ValueError("Invalid source type or missing input.")

@st.cache_data
def fill_missing_values(df):
    """Fill missing values in the DataFrame using binning for grouping.

    The method creates bins for numeric columns used in grouping keys to better handle
    imputation, and then applies group-based aggregation to fill the missing values.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()

    # Rellenar columnas de texto
    df_filled['Nombre'] = df_filled['Nombre'].fillna('N/A')
    df_filled['Género'] = df_filled['Género'].fillna('No binario')

    # Crear bins para columnas numéricas
    df_filled['Ingreso_bin'] = pd.qcut(df_filled['Ingreso_Anual_USD'], q=4, duplicates='drop')
    df_filled['Historial_bin'] = pd.qcut(df_filled['Historial_Compras'], q=4, duplicates='drop')
    df_filled['Edad_bin'] = pd.qcut(df_filled['Edad'], q=4, duplicates='drop')

    # Para 'Frecuencia_Compra': si es numérica se usa qcut; de lo contrario se convierte a códigos
    df_filled['Frecuencia_bin'] = (
        pd.qcut(df_filled['Frecuencia_Compra'], q=4, duplicates='drop')
        if pd.api.types.is_numeric_dtype(df_filled['Frecuencia_Compra'])
        else df_filled['Frecuencia_Compra'].astype('category').cat.codes
    )

    # Imputar valores faltantes usando agrupaciones
    df_filled['Edad'] = df_filled['Edad'].fillna(
        df_filled.groupby(['Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin'])['Edad']
        .transform('median')
    )
    df_filled['Ingreso_Anual_USD'] = df_filled['Ingreso_Anual_USD'].fillna(
        df_filled.groupby(['Edad_bin', 'Género', 'Historial_bin', 'Frecuencia_bin'])['Ingreso_Anual_USD']
        .transform('median')
    )
    df_filled['Historial_Compras'] = df_filled['Historial_Compras'].fillna(
        df_filled.groupby(['Edad_bin', 'Género', 'Ingreso_bin', 'Frecuencia_bin'])['Historial_Compras']
        .transform('median')
    )
    df_filled['Frecuencia_Compra'] = df_filled['Frecuencia_Compra'].fillna(
        df_filled.groupby(['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin'])['Frecuencia_Compra']
        .transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    df_filled['Latitud'] = df_filled['Latitud'].fillna(
        df_filled.groupby(['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin'])['Latitud']
        .transform('median')
    )
    df_filled['Longitud'] = df_filled['Longitud'].fillna(
        df_filled.groupby(['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin'])['Longitud']
        .transform('median')
    )

    # Eliminar columnas temporales de bins
    df_filled = df_filled.drop(columns=['Ingreso_bin', 'Historial_bin', 'Frecuencia_bin', 'Edad_bin'])

    # Respaldo global para cualquier valor faltante restante
    df_filled = df_filled.fillna({
        'Edad': df['Edad'].median(),
        'Ingreso_Anual_USD': df['Ingreso_Anual_USD'].median(),
        'Historial_Compras': df['Historial_Compras'].median(),
        'Latitud': df['Latitud'].median(),
        'Longitud': df['Longitud'].median()
    })

    return df_filled

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

def plot_on_basemap(df_points, title="Mapa de Clientes"):
    """Plot client locations on a base map loaded from a GeoPackage.

    Args:
        df_points (pd.DataFrame): DataFrame with columns 'Latitud' y 'Longitud'.
        title (str, optional): Título del gráfico.
    """
    base_map = cargar_mapa_base(URL_GEOPACKAGE)
    # Convertir los datos de clientes a GeoDataFrame
    gdf_points = gpd.GeoDataFrame(
        df_points,
        geometry=gpd.points_from_xy(df_points['Longitud'], df_points['Latitud']),
        crs="EPSG:4326"
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    base_map.plot(ax=ax, color="lightgray", edgecolor="black")
    gdf_points.plot(ax=ax, color="red", markersize=50)
    ax.set_title(title)
    st.pyplot(fig)

def analyze_correlation(df):
    """Analyze correlation between Edad and Ingreso_Anual_USD.

    Computes global correlation, and correlations segmented by Género and Frecuencia_Compra.

    Args:
        df (pd.DataFrame): DataFrame with filled data.

    Returns:
        dict: Dictionary with correlation results.
    """
    global_corr = df['Edad'].corr(df['Ingreso_Anual_USD'])
    by_gender = df.groupby('Género').apply(lambda x: x['Edad'].corr(x['Ingreso_Anual_USD']))
    by_frequency = df.groupby('Frecuencia_Compra').apply(lambda x: x['Edad'].corr(x['Ingreso_Anual_USD']))
    return {'global': global_corr, 'by_gender': by_gender, 'by_frequency': by_frequency}

def plot_correlation_segmented(df):
    """Plot correlation between Edad and Ingreso_Anual_USD segmented by Género and Frecuencia_Compra.

    This function uses seaborn's lmplot to create a facet grid where each panel corresponds
    to a value of Frecuencia_Compra, and within each panel points are colored by Género.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
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

def map_global(df):
    """Display a global map of client locations using a base map from a GeoPackage.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    st.subheader("Mapa Global de Clientes")
    plot_on_basemap(df, title="Clientes Globales")

def map_by_gender(df):
    """Display a map of client locations filtered by Género using the base map.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    gender = st.selectbox("Seleccione Género", df['Género'].unique())
    filtered = df[df['Género'] == gender]
    st.subheader(f"Mapa de Clientes - Género: {gender}")
    plot_on_basemap(filtered, title=f"Clientes - Género: {gender}")

def map_by_frequency(df):
    """Display a map of client locations filtered by Frecuencia de Compra using the base map.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    freq = st.selectbox("Seleccione Frecuencia de Compra", df['Frecuencia_Compra'].unique())
    filtered = df[df['Frecuencia_Compra'] == freq]
    st.subheader(f"Mapa de Clientes - Frecuencia de Compra: {freq}")
    plot_on_basemap(filtered, title=f"Clientes - Frecuencia: {freq}")

def custom_map(df):
    """Display a custom map based on user-selected variable ranges using the base map.

    The user can select up to four numeric variables and specify their range to filter the data.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
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
    """Perform clustering analysis based on Frecuencia_Compra.

    Converts 'Frecuencia_Compra' to numeric if needed, applies KMeans clustering, and displays the results.

    Args:
        df (pd.DataFrame): DataFrame with filled data.

    Returns:
        pd.DataFrame: DataFrame with an added 'cluster' column.
    """
    df_cluster = df.copy()
    if not pd.api.types.is_numeric_dtype(df_cluster['Frecuencia_Compra']):
        df_cluster['Frecuencia_Compra_num'] = df_cluster['Frecuencia_Compra'].astype('category').cat.codes
    else:
        df_cluster['Frecuencia_Compra_num'] = df_cluster['Frecuencia_Compra']
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(df_cluster[['Frecuencia_Compra_num']])
    st.subheader("Análisis de Clúster")
    st.dataframe(df_cluster[['ID_Cliente', 'Nombre', 'Género', 'Frecuencia_Compra', 'cluster']])
    return df_cluster

def bar_chart_gender_frequency(df):
    """Display a bar chart segmented by Género and Frecuencia de Compra.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    st.subheader("Gráfico de Barra por Género y Frecuencia de Compra")
    chart_data = df.groupby(['Género', 'Frecuencia_Compra']).size().unstack(fill_value=0)
    st.bar_chart(chart_data)

def heatmap_income(df):
    """Display a heat map based on Ingreso_Anual_USD using pydeck.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    st.subheader("Mapa de Calor según Ingresos")
    heat_data = df[['Latitud', 'Longitud', 'Ingreso_Anual_USD']].dropna()
    layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_data,
        get_position='[Longitud, Latitud]',
        get_weight='Ingreso_Anual_USD',
        radius=50,
        aggregation=pdk.types.String("SUM")
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=heat_data['Latitud'].mean(),
            longitude=heat_data['Longitud'].mean(),
            zoom=4,
            pitch=50
        )
    )
    st.pydeck_chart(deck)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance between two points in kilometers.

    Args:
        lat1, lon1, lat2, lon2 (float or np.array): Coordinates in degrees.

    Returns:
        float or np.array: Distance in kilometers.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

def distance_calculation(df):
    """Calculate distances between the highest income buyer globally and segmented groups.

    Identifies the global highest income buyer, and for each Género and Frecuencia de Compra,
    calculates the distance to that comprador.

    Args:
        df (pd.DataFrame): DataFrame with filled data.
    """
    global_max = df.loc[df['Ingreso_Anual_USD'].idxmax()]
    by_gender = df.groupby('Género').apply(lambda x: x.loc[x['Ingreso_Anual_USD'].idxmax()]).reset_index(drop=True)
    by_frequency = df.groupby('Frecuencia_Compra').apply(lambda x: x.loc[x['Ingreso_Anual_USD'].idxmax()]).reset_index(drop=True)
    by_gender['distance_to_global'] = haversine_distance(
        global_max['Latitud'], global_max['Longitud'],
        by_gender['Latitud'], by_gender['Longitud']
    )
    by_frequency['distance_to_global'] = haversine_distance(
        global_max['Latitud'], global_max['Longitud'],
        by_frequency['Latitud'], by_frequency['Longitud']
    )
    st.subheader("Distancias (por Género) desde el comprador de mayores ingresos global")
    st.dataframe(by_gender[['Género', 'Ingreso_Anual_USD', 'distance_to_global']])
    st.subheader("Distancias (por Frecuencia de Compra) desde el comprador de mayores ingresos global")
    st.dataframe(by_frequency[['Frecuencia_Compra', 'Ingreso_Anual_USD', 'distance_to_global']])

def main():
    st.title("Análisis de Datos de Clientes")
    st.sidebar.header("Carga de datos")
    source_type = st.sidebar.radio("Seleccione la fuente de datos:", ("Archivo", "URL"))

    if source_type == "Archivo":
        file = st.sidebar.file_uploader("Sube un archivo CSV", type="csv")
        if file:
            df = load_data('file', file=file)
    elif source_type == "URL":
        url = st.sidebar.text_input("Introduce la URL del archivo CSV")
        if url:
            df = load_data('url', url=url)

    if 'df' in locals():
        st.subheader("Datos originales")
        st.dataframe(df)
        df_filled = fill_missing_values(df)
        st.subheader("Datos con valores faltantes rellenados")
        st.dataframe(df_filled)

        funcionalidad = st.sidebar.selectbox(
            "Seleccione funcionalidad:",
            ["Análisis de Correlación", "Mapa Global", "Mapa por Género",
             "Mapa por Frecuencia de Compra", "Mapa Personalizado", "Análisis de Clúster",
             "Gráfico de Barra", "Mapa de Calor", "Cálculo de Distancias"]
        )

        if funcionalidad == "Análisis de Correlación":
            corr = analyze_correlation(df_filled)
            st.subheader("Correlaciones numéricas")
            st.write("Global:", corr['global'])
            st.write("Por Género:", corr['by_gender'])
            st.write("Por Frecuencia de Compra:", corr['by_frequency'])
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
        elif funcionalidad == "Gráfico de Barra":
            bar_chart_gender_frequency(df_filled)
        elif funcionalidad == "Mapa de Calor":
            heatmap_income(df_filled)
        elif funcionalidad == "Cálculo de Distancias":
            distance_calculation(df_filled)

if __name__ == "__main__":
    main()
