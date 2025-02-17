import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pydeck as pdk
from unidecode import unidecode

def load_data(uploaded_file=None, url=None):
    """Carga datos desde un archivo CSV subido o una URL."""
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file).copy()
        if url:
            return pd.read_csv(url).copy()
        raise ValueError("Debe proporcionar un archivo o URL")
    except Exception as e:
        raise Exception(f"Error cargando datos: {str(e)}") from e

def handle_missing_values(df):
    """Rellena valores faltantes usando interpolación vectorizada."""
    df_filled = df.copy()
    numeric_cols = df_filled.select_dtypes(include=np.number).columns
    
    integer_cols = df_filled[numeric_cols].select_dtypes(include='integer').columns
    float_cols = df_filled[numeric_cols].select_dtypes(include='float').columns
    
    if not integer_cols.empty:
        df_filled[integer_cols] = df_filled[integer_cols].interpolate(method='nearest').copy()
    if not float_cols.empty:
        df_filled[float_cols] = df_filled[float_cols].interpolate(method='linear').copy()
    
    return df_filled

def get_top_species(df):
    """Identifica las 5 especies más comunes y sus volúmenes por departamento."""
    df = df.copy()
    required_columns = {'DPTO', 'ESPECIE', 'VOLUMEN M3'}
    if not required_columns.issubset(df.columns):
        raise ValueError("El dataset no contiene las columnas requeridas")
    
    grouped = df.groupby(['DPTO', 'ESPECIE'], observed=True)['VOLUMEN M3']\
                .sum()\
                .reset_index()\
                .copy()
    
    grouped['Rank'] = grouped.groupby('DPTO')['VOLUMEN M3']\
                            .rank(method='dense', ascending=False)
    top_species = grouped[grouped['Rank'] <= 5]\
                    .sort_values(['DPTO', 'Rank'])\
                    .reset_index(drop=True)\
                    .copy()
    
    return top_species.drop(columns='Rank')

def prepare_geo_data(df):
    """Prepara y fusiona datos de volumen con geometrías municipales."""
    df = df.copy()
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url).copy()
    
    # Normalización de nombres
    df['DPTO'] = df['DPTO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_DPTO'] = geo_df['NOM_DPTO'].apply(lambda x: unidecode(x).upper().strip())
    
    # Calcular volumen por departamento
    volume_by_dept = df.groupby('DPTO')['VOLUMEN M3'].sum().reset_index().copy()
    
    # Fusionar datos
    merged = geo_df.merge(volume_by_dept, 
                        left_on='NOM_DPTO',
                        right_on='DPTO',
                        how='inner').copy()
    
    # Crear GeoDataFrame
    return gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged.LONGITUD, merged.LATITUD)
    ).copy()

def create_colombia_heatmap(gdf):
    """Genera mapa de calor interactivo para Colombia."""
    gdf = gdf.copy()
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=gdf,
        get_position=["LONGITUD", "LATITUD"],
        get_weight="VOLUMEN M3",
        opacity=0.7,
        threshold=0.05,
        radius_pixels=30,
        pickable=True
    )
    
    view_state = pdk.ViewState(
        latitude=4.5709,
        longitude=-74.2973,
        zoom=4.5,
        pitch=40
    )
    
    tooltip = {
        "html": "<b>Departamento:</b> {NOM_DPTO}<br><b>Volumen:</b> {VOLUMEN M3} m³",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    return pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='light'
    )

def main():
    """Función principal para la aplicación Streamlit."""
    st.title("Análisis de Producción Maderera")

    # Sección de carga de datos
    uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    url = st.text_input("O ingresar URL de datos CSV")

    df = None
    if uploaded_file or url:
        try:
            df = load_data(uploaded_file, url)
            df_clean = handle_missing_values(df)
            
            # Mostrar datos originales y procesados
            st.header("Datos Originales")
            st.dataframe(df)
            
            st.header("Datos Procesados")
            st.dataframe(df_clean)

            # Análisis de especies
            st.header("Análisis de Especies")
            top_species = get_top_species(df_clean)
            st.subheader("Top 5 Especies por Volumen y Departamento")
            st.dataframe(top_species)

            # Mapa de calor
            st.header("Mapa de Calor por Volumen Maderero")
            gdf = prepare_geo_data(df_clean)
            heatmap = create_colombia_heatmap(gdf)
            st.pydeck_chart(heatmap)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
