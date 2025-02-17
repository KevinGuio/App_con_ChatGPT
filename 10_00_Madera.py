import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import geopandas as gpd
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

        # Correcciones específicas de nombres
        rename_pairs = [
            ('ANO', 'AÑO'),
            ('DEPARTAMENTO', 'DPTO'),  # Nueva línea clave
            ('DEPT', 'DPTO')           # Otras posibles variantes
        ]
        
        for old_name, new_name in rename_pairs:
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
            
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
    required = {'AÑO', 'ESPECIE', 'TIPO_PRODUCTO', 'VOLUMEN_M3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Agrupar y sumar volúmenes
    evolution = df.groupby(['AÑO', 'ESPECIE', 'TIPO_PRODUCTO'], observed=False)\
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
    

def detect_outliers(df, method='zscore'):
    """Detecta outliers usando diferentes métodos estadísticos.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        method (str): Método a usar ('zscore' o 'iqr')
    
    Returns:
        tuple: (DataFrame con outliers, métricas relevantes)
    """
    df = df.copy()
    
    if 'VOLUMEN_M3' not in df.columns:
        raise ValueError("Columna VOLUMEN_M3 no encontrada")
    
    # Calcular métricas base
    stats = {
        'media': df['VOLUMEN_M3'].mean(),
        'mediana': df['VOLUMEN_M3'].median(),
        'std': df['VOLUMEN_M3'].std(),
        'q1': df['VOLUMEN_M3'].quantile(0.25),
        'q3': df['VOLUMEN_M3'].quantile(0.75)
    }
    
    # Detección de outliers
    if method == 'zscore':
        df['zscore'] = np.abs((df['VOLUMEN_M3'] - stats['media']) / stats['std'])
        outliers = df[df['zscore'] > 3]
    elif method == 'iqr':
        iqr = stats['q3'] - stats['q1']
        lower_bound = stats['q1'] - 1.5 * iqr
        upper_bound = stats['q3'] + 1.5 * iqr
        outliers = df[(df['VOLUMEN_M3'] < lower_bound) | (df['VOLUMEN_M3'] > upper_bound)]
    else:
        raise ValueError("Método no válido. Usar 'zscore' o 'iqr'")
    
    # Calcular métricas adicionales
    stats.update({
        'total_outliers': len(outliers),
        'porcentaje_outliers': (len(outliers) / len(df)) * 100,
        'min_outlier': outliers['VOLUMEN_M3'].min() if not outliers.empty else None,
        'max_outlier': outliers['VOLUMEN_M3'].max() if not outliers.empty else None
    })
    
    return outliers, stats

def plot_outliers(df, stats):
    """Crea visualizaciones interactivas para los outliers."""
    fig = px.box(df, y='VOLUMEN_M3', title='Distribución de Volúmenes con Outliers',
                labels={'VOLUMEN_M3': 'Volumen (m³)'})
    
    fig.add_annotation(x=0, y=stats['mediana'], text=f"Mediana: {stats['mediana']:.2f}",
                      showarrow=False, yshift=10)
    
    fig2 = px.histogram(df, x='VOLUMEN_M3', nbins=50, 
                       title='Histograma de Frecuencias con Outliers',
                       labels={'VOLUMEN_M3': 'Volumen (m³)'})
    
    return fig, fig2


def calculate_municipality_volumes(df):
    """Calcula el volumen total por municipio con geolocalización.
    
    Args:
        df (pd.DataFrame): DataFrame con datos procesados
        
    Returns:
        pd.DataFrame: Datos agrupados con información geográfica
    """
    df = df.copy()
    
    # Verificar columnas requeridas
    required = {'MUNICIPIO', 'DPTO', 'VOLUMEN_M3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Cargar datos geográficos
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url)
    
    # Normalizar nombres
    df['MUNICIPIO'] = df['MUNICIPIO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_MPIO'] = geo_df['NOM_MPIO'].apply(lambda x: unidecode(x).upper().strip())
    
    # Agrupar y sumar volúmenes
    municipios = df.groupby(['DPTO', 'MUNICIPIO'], observed=False).agg(
        VOLUMEN_TOTAL=('VOLUMEN_M3', 'sum')
    ).reset_index()
    
    # Fusionar con datos geográficos
    merged = geo_df.merge(municipios, 
                        left_on=['NOM_DPTO', 'NOM_MPIO'],
                        right_on=['DPTO', 'MUNICIPIO'],
                        how='right')
    
    return merged[['DPTO', 'MUNICIPIO', 'LATITUD', 'LONGITUD', 'VOLUMEN_TOTAL']]

def plot_municipality_volumes(gdf):
    """Crea visualización interactiva de volúmenes por municipio."""
    fig = px.scatter_mapbox(gdf,
                          lat='LATITUD',
                          lon='LONGITUD',
                          size='VOLUMEN_TOTAL',
                          color='VOLUMEN_TOTAL',
                          hover_name='MUNICIPIO',
                          hover_data={'DPTO': True, 'VOLUMEN_TOTAL': ':.2f'},
                          zoom=5,
                          height=600,
                          color_continuous_scale=px.colors.sequential.Viridis,
                          title='Volumen de Madera por Municipio')
    
    fig.update_layout(mapbox_style="carto-positron",
                    margin={"r":0,"t":40,"l":0,"b":0})
    return fig


def get_low_volume_species(df, top_n=5):
    """Identifica las especies con menor volumen movilizado."""
    df = df.copy()
    
    if 'ESPECIE' not in df.columns or 'VOLUMEN_M3' not in df.columns:
        raise ValueError("Columnas requeridas no encontradas")
    
    # Calcular volumen total por especie
    species_vol = df.groupby('ESPECIE', observed=False)['VOLUMEN_M3']\
                   .sum()\
                   .reset_index()\
                   .sort_values('VOLUMEN_M3', ascending=True)
    
    # Filtrar las N especies con menor volumen
    return species_vol.head(top_n)

def get_species_geo_distribution(df, low_species):
    """Obtiene la distribución geográfica de especies de bajo volumen."""
    df = df.copy()
    
    # Filtrar datos para las especies seleccionadas
    filtered = df[df['ESPECIE'].isin(low_species['ESPECIE'])]
    
    # Cargar y combinar datos geográficos
    geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
    geo_df = pd.read_csv(geo_url)
    
    # Normalizar nombres
    filtered['MUNICIPIO'] = filtered['MUNICIPIO'].apply(lambda x: unidecode(x).upper().strip())
    geo_df['NOM_MPIO'] = geo_df['NOM_MPIO'].apply(lambda x: unidecode(x).upper().strip())
    
    # Combinar datos
    merged = geo_df.merge(filtered,
                        left_on=['NOM_DPTO', 'NOM_MPIO'],
                        right_on=['DPTO', 'MUNICIPIO'],
                        how='right')
    
    return merged[['ESPECIE', 'DPTO', 'MUNICIPIO', 'LATITUD', 'LONGITUD', 'VOLUMEN_M3']]

def plot_low_volume_distribution(gdf):
    """Crea mapa interactivo de distribución de especies de bajo volumen."""
    fig = px.scatter_mapbox(gdf,
                          lat='LATITUD',
                          lon='LONGITUD',
                          color='ESPECIE',
                          size='VOLUMEN_M3',
                          hover_name='MUNICIPIO',
                          hover_data={'DPTO': True, 'ESPECIE': True, 'VOLUMEN_M3': ':.2f'},
                          zoom=4.5,
                          height=600,
                          title='Distribución Geográfica de Especies con Bajo Volumen',
                          mapbox_style="carto-positron")
    
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
                    legend={'title': None, 'orientation': 'h', 'y': -0.2})
    return fig


def compare_species_distribution(df):
    """Compara la distribución de especies entre departamentos."""
    df = df.copy()
    
    # Verificar columnas requeridas
    required = {'DPTO', 'ESPECIE', 'VOLUMEN_M3'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {', '.join(missing)}")
    
    # Calcular volumen total por departamento y especie
    dept_species = df.groupby(['DPTO', 'ESPECIE'], observed=False)['VOLUMEN_M3']\
                   .sum()\
                   .reset_index()
    
    return dept_species

def plot_comparison_chart(df, top_n=5):
    """Crea visualización interactiva de comparación entre departamentos."""
    # Obtener top N especies por departamento
    top_species = df.groupby('DPTO', observed=False)\
                  .apply(lambda x: x.nlargest(top_n, 'VOLUMEN_M3'))\
                  .reset_index(drop=True)
    
    # Crear gráfico
    fig = px.bar(top_species,
                x='DPTO',
                y='VOLUMEN_M3',
                color='ESPECIE',
                barmode='group',
                title=f'Distribución de Top {top_n} Especies por Departamento',
                labels={'VOLUMEN_M3': 'Volumen (m³)', 'DPTO': 'Departamento'},
                height=600)
    
    fig.update_layout(xaxis={'categoryorder':'total descending'},
                    hovermode='x unified')
    return fig

def prepare_geo_data(df):
    """Prepara datos geográficos fusionando con ubicaciones de municipios."""
    try:
        # Cargar dataset geográfico
        geo_url = "https://raw.githubusercontent.com/KevinGuio/App_con_ChatGPT/main/DIVIPOLA-_C_digos_municipios_geolocalizados_20250217.csv"
        geo_df = pd.read_csv(geo_url)
        
        # Normalizar nombres
        df['DPTO'] = df['DPTO'].apply(lambda x: unidecode(x).upper().strip())
        df['MUNICIPIO'] = df['MUNICIPIO'].apply(lambda x: unidecode(x).upper().strip())
        geo_df['NOM_DPTO'] = geo_df['NOM_DPTO'].apply(lambda x: unidecode(x).upper().strip())
        geo_df['NOM_MPIO'] = geo_df['NOM_MPIO'].apply(lambda x: unidecode(x).upper().strip())
        
        # Fusionar datos
        merged = geo_df.merge(df.groupby(['DPTO', 'MUNICIPIO']).agg({'VOLUMEN_M3': 'sum'}),
                            left_on=['NOM_DPTO', 'NOM_MPIO'],
                            right_on=['DPTO', 'MUNICIPIO'],
                            how='right')
        
        return gpd.GeoDataFrame(
            merged,
            geometry=gpd.points_from_xy(merged.LONGITUD, merged.LATITUD)
        )
    except Exception as e:
        raise ValueError(f"Error preparando datos geográficos: {str(e)}")
        

def prepare_clustering_data(df, level='department'):
    """Prepara los datos para clustering agrupando por nivel geográfico."""
    df = df.copy()
    
    # Agrupar datos según nivel seleccionado
    if level == 'department':
        group_col = 'DPTO'
    else:
        group_col = 'MUNICIPIO'
    
    # Crear características para el clustering
    features = df.groupby(group_col, observed=False).agg({
        'VOLUMEN_M3': ['sum', 'mean', 'std'],
        'ESPECIE': pd.Series.nunique,
        'TIPO_PRODUCTO': pd.Series.nunique
    }).reset_index()
    
    # Renombrar columnas
    features.columns = [
        group_col,
        'volumen_total',
        'volumen_promedio',
        'volumen_std',
        'especies_unicas',
        'productos_unicos'
    ]
    
    # Manejar valores NaN
    features.fillna(0, inplace=True)
    
    return features

def perform_clustering(data, n_clusters=3):
    """Realiza el clustering usando K-means con reducción dimensional."""
    # Escalar características
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Reducción dimensional
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Crear DataFrame con resultados
    results = pd.DataFrame({
        'cluster': clusters,
        'PC1': principal_components[:, 0],
        'PC2': principal_components[:, 1]
    })
    
    return results, pca, scaler, kmeans

    

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

        # Nueva sección de análisis de outliers
            st.header("📊 Detección de Outliers")
            
            try:
                # Selección de método
                method = st.selectbox('Seleccionar método de detección:', 
                                    ['zscore', 'iqr'], index=0)
                
                # Detectar outliers
                outliers, stats = detect_outliers(df_clean, method)
                
                # Mostrar métricas en columnas
                col1, col2, col3 = st.columns(3)
                col1.metric("🔍 Outliers Detectados", stats['total_outliers'])
                col2.metric("📈 Porcentaje de Outliers", f"{stats['porcentaje_outliers']:.2f}%")
                col3.metric("📏 Rango de Outliers", 
                           f"{stats['min_outlier']:.2f} - {stats['max_outlier']:.2f}" 
                           if outliers.any().any() else "N/A")
                
                # Visualizaciones
                fig_box, fig_hist = plot_outliers(df_clean, stats)
                st.plotly_chart(fig_box, use_container_width=True)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # Mostrar tabla de outliers
                with st.expander("🔍 Ver detalles de outliers"):
                    st.dataframe(outliers.sort_values('VOLUMEN_M3', ascending=False))
                    
                    # Botón de descarga
                    csv = outliers.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar outliers como CSV",
                        csv,
                        "outliers.csv",
                        "text/csv",
                        key='download-outliers'
                    )
                
                # Explicación estadística
                with st.expander("📚 Explicación técnica"):
                    st.markdown(f"""
                    **Método usado:** {'Z-Score' if method == 'zscore' else 'IQR'}
                    - **Media:** {stats['media']:.2f}
                    - **Desviación estándar:** {stats['std']:.2f}
                    - **Rango intercuartílico (IQR):** {stats['q3'] - stats['q1']:.2f}
                    - **Límite inferior:** {stats.get('q1', 0) - 1.5*(stats['q3'] - stats['q1']):.2f}
                    - **Límite superior:** {stats.get('q3', 0) + 1.5*(stats['q3'] - stats['q1']):.2f}
                    """)
                
            except ValueError as e:
                st.error(str(e))

        # Nueva sección de análisis por municipio
            st.header("🏘️ Volumen por Municipio")
            
            try:
                municipios_df = calculate_municipality_volumes(df_clean)
                
                # Mostrar métricas rápidas
                cols = st.columns(3)
                cols[0].metric("🗺️ Municipios Únicos", municipios_df['MUNICIPIO'].nunique())
                cols[1].metric("📦 Volumen Total", f"{municipios_df['VOLUMEN_TOTAL'].sum():,.0f} m³")
                cols[2].metric("📌 Municipio con Mayor Volumen", 
                              municipios_df.loc[municipios_df['VOLUMEN_TOTAL'].idxmax(), 'MUNICIPIO'])
                
                # Selector de departamento
                selected_dept = st.selectbox('Filtrar por Departamento:', 
                                            options=['Todos'] + sorted(municipios_df['DPTO'].unique()))
                
                # Aplicar filtro
                if selected_dept != 'Todos':
                    filtered_df = municipios_df[municipios_df['DPTO'] == selected_dept]
                else:
                    filtered_df = municipios_df
                
                # Mostrar tabla interactiva
                st.dataframe(filtered_df.sort_values('VOLUMEN_TOTAL', ascending=False),
                            column_order=['DPTO', 'MUNICIPIO', 'VOLUMEN_TOTAL'],
                            column_config={
                                'VOLUMEN_TOTAL': st.column_config.NumberColumn(
                                    format="%,.2f m³"
                                )
                            })
                
                # Visualización en mapa
                st.subheader("Mapa de Calor por Municipio")
                fig = plot_municipality_volumes(filtered_df)
                st.plotly_chart(fig, use_container_width=True)
                
                # Descarga de datos
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar datos por municipio",
                    csv,
                    "volumen_municipios.csv",
                    "text/csv",
                    key='download-municipios'
                )
                
            except ValueError as e:
                st.error(str(e))

        # Nueva sección de especies de bajo volumen
            st.header("🔍 Especies con Menor Movilización")
            
            try:
                # Selector de cantidad de especies
                n_species = st.slider('Número de especies a analizar:', 
                                     min_value=1, 
                                     max_value=10, 
                                     value=3)
                
                # Obtener especies de bajo volumen
                low_species = get_low_volume_species(df_clean, n_species)
                
                # Mostrar métricas
                cols = st.columns(3)
                cols[0].metric("🌿 Especies Identificadas", len(low_species))
                cols[1].metric("📦 Volumen Total", f"{low_species['VOLUMEN_M3'].sum():,.2f} m³")
                cols[2].metric("🏙️ Municipios Afectados", 
                              get_species_geo_distribution(df_clean, low_species)['MUNICIPIO'].nunique())
                
                # Gráfico de barras
                st.subheader(f"Top {n_species} Especies con Menor Volumen")
                fig_bar = px.bar(low_species,
                                x='VOLUMEN_M3',
                                y='ESPECIE',
                                orientation='h',
                                labels={'VOLUMEN_M3': 'Volumen Total (m³)', 'ESPECIE': ''},
                                color='VOLUMEN_M3',
                                color_continuous_scale='Reds')
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Mapa de distribución
                st.subheader("Distribución Geográfica")
                geo_data = get_species_geo_distribution(df_clean, low_species)
                fig_map = plot_low_volume_distribution(geo_data)
                st.plotly_chart(fig_map, use_container_width=True)
                
                # Tabla detallada
                with st.expander("📋 Ver datos detallados"):
                    st.dataframe(geo_data.sort_values(['VOLUMEN_M3', 'DPTO'], ascending=[True, True]))
                    
                    # Botón de descarga
                    csv = geo_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Descargar datos geográficos",
                        csv,
                        f"especies_bajo_volumen_{n_species}.csv",
                        "text/csv",
                        key='download-low-volume'
                    )
                
            except ValueError as e:
                st.error(str(e))

        # Nueva sección de comparación entre departamentos
            st.header("📊 Comparación entre Departamentos")
            
            try:
                comparison_data = compare_species_distribution(df_clean)
                
                # Controles interactivos
                col1, col2 = st.columns(2)
                selected_depts = col1.multiselect(
                    'Seleccionar departamentos:',
                    options=comparison_data['DPTO'].unique(),
                    default=comparison_data['DPTO'].unique()[:3]
                )
                
                top_n = col2.slider('Número de especies a mostrar:', 
                                   min_value=1, 
                                   max_value=10, 
                                   value=5)
                
                # Filtrar datos
                filtered_data = comparison_data[comparison_data['DPTO'].isin(selected_depts)]
                
                # Mostrar métricas
                cols = st.columns(3)
                cols[0].metric("🗺️ Departamentos Seleccionados", len(selected_depts))
                cols[1].metric("🌿 Especies Únicas", filtered_data['ESPECIE'].nunique())
                cols[2].metric("📦 Volumen Total", f"{filtered_data['VOLUMEN_M3'].sum():,.0f} m³")
                
                # Gráfico principal
                st.subheader("Distribución Comparativa")
                fig = plot_comparison_chart(filtered_data, top_n)
                st.plotly_chart(fig, use_container_width=True)
                
                # Vista de tabla
                st.subheader("Tabla Comparativa")
                pivot_table = filtered_data.pivot_table(
                    index='DPTO',
                    columns='ESPECIE',
                    values='VOLUMEN_M3',
                    aggfunc='sum'
                ).fillna(0)
                
                # Convertir a porcentaje relativo opcional
                if st.checkbox('Mostrar como porcentaje relativo por departamento'):
                    pivot_table = pivot_table.div(pivot_table.sum(axis=1), axis=0) * 100
                
                st.dataframe(pivot_table.style.format("{:.1f}"),
                            use_container_width=True)
                
                # Descarga de datos
                csv = pivot_table.to_csv().encode('utf-8')
                st.download_button(
                    "📥 Descargar tabla comparativa",
                    csv,
                    "comparacion_departamentos.csv",
                    "text/csv",
                    key='download-comparison'
                )
                
            except ValueError as e:
                st.error(str(e))

        # Nueva sección de clustering
            st.header("🔍 Clustering Geográfico")
            
            try:
                # Selección de parámetros
                col1, col2 = st.columns(2)
                level = col1.radio("Nivel de análisis:", 
                                 ['department', 'municipality'], 
                                 format_func=lambda x: 'Departamento' if x == 'department' else 'Municipio')
                n_clusters = col2.slider('Número de clusters:', 2, 6, 3)
                
                # Preparar datos
                clustering_data = prepare_clustering_data(df_clean, level)
                geo_data = prepare_geo_data(df_clean)
                
                # Realizar clustering
                cluster_results, pca, scaler, model = perform_clustering(
                    clustering_data.drop(columns=[clustering_data.columns[0]]), 
                    n_clusters
                )
                
                # Combinar resultados
                full_results = pd.concat([
                    clustering_data,
                    cluster_results
                ], axis=1)
                
                # Mostrar análisis
                st.subheader("Resultados del Clustering")
                
                # Gráfico PCA
                fig_pca = px.scatter(full_results, 
                                    x='PC1', 
                                    y='PC2', 
                                    color='cluster',
                                    hover_name=clustering_data.columns[0],
                                    title='Visualización de Clusters en Espacio PCA')
                st.plotly_chart(fig_pca, use_container_width=True)
                
                
            except ValueError as e:
                st.error(str(e))
                
        
        except Exception as e:
            st.error(f"🚨 Error general: {str(e)}")

if __name__ == "__main__":
    main()
