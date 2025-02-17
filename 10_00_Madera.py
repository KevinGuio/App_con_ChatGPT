import streamlit as st
import pandas as pd
import numpy as np

def load_data(uploaded_file=None, url=None):
    """Carga datos desde un archivo CSV subido o una URL.

    Args:
        uploaded_file (UploadedFile, optional): Archivo cargado mediante Streamlit.
            Defaults to None.
        url (str, optional): URL para cargar datos remotos. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        ValueError: Si no se proporciona ningún método de carga.
        Exception: Para errores generales de carga.
    """
    try:
        if uploaded_file is not None:
            return pd.read_csv(uploaded_file)
        if url:
            return pd.read_csv(url)
        raise ValueError("Debe proporcionar un archivo o URL")
    except Exception as e:
        raise Exception(f"Error cargando datos: {str(e)}") from e

def handle_missing_values(df):
    """Rellena valores faltantes usando interpolación según tipo de dato de la columna.

    Args:
        df (pd.DataFrame): DataFrame original con datos faltantes.

    Returns:
        pd.DataFrame: DataFrame con valores faltantes interpolados.
    """
    df_filled = df.copy()
    
    # Detectar columnas numéricas
    numeric_cols = df_filled.select_dtypes(include=np.number).columns
    
    # Separar columnas enteras y decimales
    integer_cols = df_filled[numeric_cols].select_dtypes(include='integer').columns
    float_cols = df_filled[numeric_cols].select_dtypes(include='float').columns
    
    # Aplicar interpolación específica por tipo de dato
    if not integer_cols.empty:
        df_filled[integer_cols] = df_filled[integer_cols].interpolate(method='nearest')
    if not float_cols.empty:
        df_filled[float_cols] = df_filled[float_cols].interpolate(method='linear')
    
    return df_filled

def get_top_species(df):
    """Identifica las 5 especies más comunes y sus volúmenes por departamento.

    Args:
        df (pd.DataFrame): DataFrame con los datos procesados

    Returns:
        pd.DataFrame: DataFrame con ranking de especies por departamento
    """
    # Verificar existencia de columnas requeridas
    required_columns = {'DPTO', 'ESPECIE', 'VOLUMEN M3'}
    if not required_columns.issubset(df.columns):
        raise ValueError("El dataset no contiene las columnas requeridas")
    
    # Agrupar y sumar volúmenes
    grouped = df.groupby(['DPTO', 'ESPECIE'], observed=True)['VOLUMEN M3']\
                .sum()\
                .reset_index()
    
    # Ordenar y seleccionar top 5 por departamento
    grouped['Rank'] = grouped.groupby('DPTO')['VOLUMEN M3']\
                            .rank(method='dense', ascending=False)
    top_species = grouped[grouped['Rank'] <= 5]\
                    .sort_values(['DPTO', 'Rank'])\
                    .reset_index(drop=True)
    
    return top_species.drop(columns='Rank')

def plot_top_10_species(df):
    """Genera gráfico de barras con las 10 especies con mayor volumen total.
    
    Args:
        df (pd.DataFrame): DataFrame con datos procesados
        
    Returns:
        matplotlib.figure.Figure: Gráfico de barras
    """
    # Verificar columnas requeridas
    if 'ESPECIE' not in df.columns or 'VOLUMEN M3' not in df.columns:
        raise ValueError("Dataset no contiene las columnas requeridas")
    
    # Calcular totales por especie
    species_volume = df.groupby('ESPECIE', observed=True)['VOLUMEN M3']\
                      .sum()\
                      .nlargest(10)\
                      .sort_values(ascending=True)  # Para mejor visualización
    
    # Crear gráfico
    fig = plt.figure(figsize=(10, 6))
    species_volume.plot(kind='barh', color='#1f77b4')
    plt.title('Top 10 Especies con Mayor Volumen Movilizado')
    plt.xlabel('Volumen Total (m³)')
    plt.ylabel('Especie')
    plt.tight_layout()
    
    return fig

def main():
    """Función principal para la aplicación Streamlit."""
    st.title("Análisis de Producción Maderera")

    # Sección de carga de datos
    st.header("Carga de Datos")
    uploaded_file = st.file_uploader("Subir archivo CSV", type="csv")
    url = st.text_input("O ingresar URL de datos CSV")

    df = None
    if uploaded_file or url:
        try:
            df = load_data(uploaded_file, url)
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return

    if df is not None:
        # Mostrar datos originales
        st.header("Datos Originales")
        st.dataframe(df)

        # Procesamiento automático
        try:
            df_clean = handle_missing_values(df)
            
            # Mostrar resultados limpieza
            st.header("Datos Procesados")
            st.dataframe(df_clean)

            # Comparativa de valores faltantes
            st.subheader("Resumen de Valores Faltantes")
            original_nulls = df.select_dtypes(include=np.number).isnull().sum()
            cleaned_nulls = df_clean.select_dtypes(include=np.number).isnull().sum()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Valores faltantes originales:")
                st.write(original_nulls)
            with col2:
                st.write("Valores faltantes después de interpolación:")
                st.write(cleaned_nulls)

            # Análisis de especies
            st.header("Análisis de Especies")
            
            try:
                # Top 5 por departamento
                top_species = get_top_species(df_clean)
                
                st.subheader("Top 5 Especies por Volumen y Departamento")
                st.dataframe(top_species)
                
                # Métricas
                total_volume = top_species['VOLUMEN M3'].sum()
                avg_volume = top_species['VOLUMEN M3'].mean()
                total_species = top_species['ESPECIE'].nunique()
                
                cols = st.columns(3)
                cols[0].metric("Volumen Total Top 5 (m³)", f"{total_volume:,.2f}")
                cols[1].metric("Promedio por Especie", f"{avg_volume:,.2f}")
                cols[2].metric("Especies Únicas Top 5", total_species)
                
                # Nuevo gráfico top 10 global
                st.subheader("Top 10 Especies a Nivel Nacional")
                top_10 = df_clean.groupby('ESPECIE')['VOLUMEN M3']\
                                .sum()\
                                .nlargest(10)\
                                .reset_index()
                                
                st.bar_chart(top_10.set_index('ESPECIE'))
                
            except ValueError as e:
                st.warning(str(e))

        except Exception as e:
            st.error(f"Error en procesamiento: {str(e)}")

if __name__ == "__main__":
    main()
