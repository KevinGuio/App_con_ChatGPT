import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
from unidecode import unidecode

def load_data(uploaded_file=None, url=None):
    """Carga y normaliza datos desde CSV."""
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif url:
            df = pd.read_csv(url)
        else:
            raise ValueError("Debe proporcionar un archivo o URL")
        
        # Normalizar nombres de columnas
        df.columns = [unidecode(col).strip().upper().replace(' ', '_') for col in df.columns]
        return df.copy()
    
    except Exception as e:
        raise Exception(f"Error cargando datos: {str(e)}") from e

def handle_missing_values(df):
    """Interpolación segura con copias explícitas."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Crear copias independientes de las columnas
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='nearest' if np.issubdtype(df[col].dtype, np.integer) else 'linear').copy()
    
    return df

def get_top_species(df):
    """Identifica top 5 especies con validación de columnas."""
    df = df.copy()
    required = {'DPTO', 'ESPECIE', 'VOLUMEN_M3'}
    
    # Verificar existencia de columnas normalizadas
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Agrupar con operaciones vectorizadas
    grouped = df.groupby(['DPTO', 'ESPECIE'], observed=False, as_index=False).agg(
        VOLUMEN_TOTAL=('VOLUMEN_M3', 'sum')
    )
    
    # Filtrar top 5 por departamento
    grouped['RANK'] = grouped.groupby('DPTO')['VOLUMEN_TOTAL'].rank(ascending=False, method='dense')
    return grouped[grouped['RANK'] <= 5].sort_values(['DPTO', 'RANK']).drop(columns='RANK')

def create_heatmap(df):
    """Crea mapa de calor con Plotly Express."""
    # Cargar datos geográficos
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url)
    
    # Normalizar nombres en ambos datasets
    df['DPTO'] = df['DPTO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_DPTO'] = geo_df['NOM_DPTO'].apply(lambda x: unidecode(x).upper().strip())
    
    # Calcular centroides por departamento
    dept_volumes = df.groupby('DPTO', observed=False)['VOLUMEN_M3'].sum().reset_index()
    merged = geo_df.merge(dept_volumes, left_on='NOM_DPTO', right_on='DPTO')
    
    # Crear mapa interactivo
    fig = px.density_mapbox(
        merged,
        lat='LATITUD',
        lon='LONGITUD',
        z='VOLUMEN_M3',
        hover_name='NOM_DPTO',
        radius=20,
        zoom=4,
        mapbox_style="carto-positron",
        title='Distribución de Volúmenes por Departamento'
    )
    
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    return fig

def main():
    st.title("🌳 Análisis de Producción Maderera")
    
    # Carga de datos
    uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    url = st.text_input("O ingresar URL de datos CSV")
    
    if uploaded_file or url:
        try:
            # Procesamiento inicial
            df = load_data(uploaded_file, url)
            df_clean = handle_missing_values(df)
            
            # Sección de datos
            with st.expander("📥 Datos Originales"):
                st.dataframe(df)
            
            with st.expander("🧹 Datos Procesados"):
                st.dataframe(df_clean)
            
            # Análisis de especies
            st.header("🔝 Top 5 Especies por Departamento")
            try:
                top_species = get_top_species(df_clean)
                st.dataframe(top_species)
                
                # Métricas
                cols = st.columns(3)
                cols[0].metric("📦 Volumen Total", f"{top_species['VOLUMEN_TOTAL'].sum():,.0f} m³")
                cols[1].metric("🌳 Especies Únicas", top_species['ESPECIE'].nunique())
                cols[2].metric("🗺️ Departamentos", top_species['DPTO'].nunique())
                
            except ValueError as e:
                st.error(str(e))
            
            # Mapa de calor
            st.header("🌎 Mapa de Distribución")
            fig = create_heatmap(df_clean)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()
