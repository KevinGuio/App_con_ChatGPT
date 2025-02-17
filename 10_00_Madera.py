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
    
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='nearest' if np.issubdtype(df[col].dtype, np.integer) else 'linear').copy()
    
    return df

def plot_top_species(df):
    """Crea gráfico de barras de las 10 especies con mayor volumen."""
    df = df.copy()
    
    if 'ESPECIE' not in df.columns or 'VOLUMEN_M3' not in df.columns:
        raise ValueError("Dataset no contiene las columnas requeridas")
    
    # Calcular y ordenar los datos
    top_species = df.groupby('ESPECIE', observed=False)['VOLUMEN_M3']\
                   .sum()\
                   .nlargest(10)\
                   .reset_index()
    
    # Crear gráfico interactivo
    fig = px.bar(top_species, 
                 x='VOLUMEN_M3', 
                 y='ESPECIE', 
                 orientation='h',
                 title='Top 10 Especies con Mayor Volumen Movilizado',
                 labels={'VOLUMEN_M3': 'Volumen Total (m³)', 'ESPECIE': 'Especie'},
                 color='VOLUMEN_M3',
                 color_continuous_scale='Greens')
    
    fig.update_layout(showlegend=False,
                    yaxis={'categoryorder':'total ascending'},
                    height=500)
    return fig

def get_top_species(df):
    """Identifica top 5 especies con validación de columnas."""
    df = df.copy()
    required = {'DPTO', 'ESPECIE', 'VOLUMEN_M3'}
    
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    grouped = df.groupby(['DPTO', 'ESPECIE'], observed=False, as_index=False).agg(
        VOLUMEN_TOTAL=('VOLUMEN_M3', 'sum')
    )
    
    grouped['RANK'] = grouped.groupby('DPTO')['VOLUMEN_TOTAL'].rank(ascending=False, method='dense')
    return grouped[grouped['RANK'] <= 5].sort_values(['DPTO', 'RANK']).drop(columns='RANK')

def create_heatmap(df):
    """Crea mapa de calor con Plotly Express."""
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url)
    
    df['DPTO'] = df['DPTO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_DPTO'] = geo_df['NOM_DPTO'].apply(lambda x: unidecode(x).upper().strip())
    
    dept_volumes = df.groupby('DPTO', observed=False)['VOLUMEN_M3'].sum().reset_index()
    merged = geo_df.merge(dept_volumes, left_on='NOM_DPTO', right_on='DPTO')
    
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


def get_top_municipalities(df):
    """Identifica los 10 municipios con mayor volumen movilizado."""
    df = df.copy()
    
    # Verificar columnas requeridas
    required = {'DPTO', 'MUNICIPIO', 'VOLUMEN_M3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Agrupar y sumar volúmenes
    top_municipios = df.groupby(['DPTO', 'MUNICIPIO'], observed=False, as_index=False)\
                      .agg(VOLUMEN_TOTAL=('VOLUMEN_M3', 'sum'))\
                      .nlargest(10, 'VOLUMEN_TOTAL')
    
    return top_municipios

def plot_municipality_map(df):
    """Crea mapa interactivo de los municipios con mayor movilización."""
    # Cargar datos geográficos
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url)
    
    # Normalizar nombres
    df['MUNICIPIO'] = df['MUNICIPIO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_MPIO'] = geo_df['NOM_MPIO'].apply(lambda x: unidecode(x).upper().strip())
    
    # Fusionar datos
    merged = geo_df.merge(df, 
                        left_on=['NOM_DPTO', 'NOM_MPIO'],
                        right_on=['DPTO', 'MUNICIPIO'],
                        how='inner')
    
    # Crear mapa interactivo
    fig = px.scatter_mapbox(merged,
                           lat='LATITUD',
                           lon='LONGITUD',
                           size='VOLUMEN_TOTAL',
                           color='VOLUMEN_TOTAL',
                           hover_name='NOM_MPIO',
                           hover_data={'DPTO': True, 'VOLUMEN_TOTAL': ':.2f'},
                           zoom=4.5,
                           height=600,
                           color_continuous_scale=px.colors.sequential.Viridis,
                           title='Top 10 Municipios con Mayor Movilización de Madera')
    
    fig.update_layout(mapbox_style="carto-positron",
                    margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def get_temporal_evolution(df):
    """Analiza la evolución temporal por especie y tipo de producto."""
    df = df.copy()
    
    # Verificar columnas requeridas
    required = {'ANO', 'ESPECIE', 'TIPO_PRODUCTO', 'VOLUMEN_M3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Agrupar y sumar volúmenes
    evolution = df.groupby(['ANO', 'ESPECIE', 'TIPO_PRODUCTO'], observed=False)\
                .agg(VOLUMEN_TOTAL=('VOLUMEN_M3', 'sum'))\
                .reset_index()
    
    return evolution

def plot_temporal_evolution(df):
    """Crea gráfico interactivo de evolución temporal."""
    fig = px.line(df,
                 x='AÑO',
                 y='VOLUMEN_TOTAL',
                 color='ESPECIE',
                 line_dash='TIPO_PRODUCTO',
                 markers=True,
                 title='Evolución Temporal del Volumen por Especie y Tipo de Producto',
                 labels={'VOLUMEN_TOTAL': 'Volumen (m³)', 'AÑO': 'Año'},
                 height=600)
    
    fig.update_layout(hovermode='x unified',
                    xaxis={'type': 'category'},
                    legend={'title': None})
    
    return fig
    

def main():
    st.title("🌳 Análisis de Producción Maderera")
    
    uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    url = st.text_input("O ingresar URL de datos CSV")
    
    if uploaded_file or url:
        try:
            df = load_data(uploaded_file, url)
            df_clean = handle_missing_values(df)
            
            with st.expander("📥 Datos Originales"):
                st.dataframe(df)
            
            with st.expander("🧹 Datos Procesados"):
                st.dataframe(df_clean)

            # Sección de Top 5 por Departamento
            st.header("📊 Análisis por Departamento")
            try:
                top_species = get_top_species(df_clean)
                st.dataframe(top_species)
                
            except ValueError as e:
                st.error(str(e))

            
            # Sección de Top 10 Especies
            st.header("🏆 Top 10 Especies a Nivel Nacional")
            try:
                fig_top10 = plot_top_species(df_clean)
                st.plotly_chart(fig_top10, use_container_width=True)
                
                # Calcular métricas
                top_data = df_clean.groupby('ESPECIE')['VOLUMEN_M3'].sum().nlargest(10)
                total = top_data.sum()
                avg = top_data.mean()
                
                cols = st.columns(3)
                cols[0].metric("📦 Volumen Total Top 10", f"{total:,.0f} m³")
                cols[1].metric("📊 Promedio por Especie", f"{avg:,.0f} m³")
                cols[2].metric("🌿 Especies Únicas", len(top_data))
                
            except ValueError as e:
                st.error(str(e))
            
            
            # Mapa de Calor
            st.header("🌎 Mapa de Distribución")
            try:
                fig_map = create_heatmap(df_clean)
                st.plotly_chart(fig_map, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error en el mapa: {str(e)}")

            st.header("🏙️ Top 10 Municipios")
            try:
                top_municipios = get_top_municipalities(df_clean)
                
                # Mostrar métricas
                cols = st.columns(3)
                cols[0].metric("📦 Volumen Total", f"{top_municipios['VOLUMEN_TOTAL'].sum():,.0f} m³")
                cols[1].metric("📍 Municipios Únicos", top_municipios['MUNICIPIO'].nunique())
                cols[2].metric("🗺️ Departamentos", top_municipios['DPTO'].nunique())
                
                # Mostrar tabla y mapa
                st.dataframe(top_municipios.sort_values('VOLUMEN_TOTAL', ascending=False))
                fig_municipios = plot_municipality_map(top_municipios)
                st.plotly_chart(fig_municipios, use_container_width=True)
                
            except ValueError as e:
                st.error(str(e))


            # Nueva sección de análisis temporal
            st.header("📅 Evolución Temporal")
            try:
                temporal_data = get_temporal_evolution(df_clean)
                
                # Filtros interactivos
                cols = st.columns(2)
                selected_species = cols[0].multiselect(
                    'Seleccionar especies:',
                    options=temporal_data['ESPECIE'].unique(),
                    default=temporal_data['ESPECIE'].unique()[:3]
                )
                
                selected_products = cols[1].multiselect(
                    'Seleccionar tipos de producto:',
                    options=temporal_data['TIPO_PRODUCTO'].unique(),
                    default=temporal_data['TIPO_PRODUCTO'].unique()[:2]
                )
                
                # Filtrar datos
                filtered_data = temporal_data[
                    (temporal_data['ESPECIE'].isin(selected_species)) &
                    (temporal_data['TIPO_PRODUCTO'].isin(selected_products))
                ]
                
                # Mostrar métricas
                total_volume = filtered_data['VOLUMEN_TOTAL'].sum()
                year_range = f"{filtered_data['AÑO'].min()} - {filtered_data['AÑO'].max()}"
                
                st.metric("📦 Volumen Total en Período Seleccionado", 
                         f"{total_volume:,.0f} m³", 
                         f"Período: {year_range}")
                
                # Mostrar gráfico
                fig_temporal = plot_temporal_evolution(filtered_data)
                st.plotly_chart(fig_temporal, use_container_width=True)
                
                # Mostrar datos subyacentes
                with st.expander("🔍 Ver datos detallados"):
                    st.dataframe(filtered_data.sort_values(['AÑO', 'VOLUMEN_TOTAL'], ascending=False))
                
            except ValueError as e:
                st.error(str(e))

        
        except Exception as e:
            st.error(f"🚨 Error general: {str(e)}")

if __name__ == "__main__":
    main()
