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


def main():
    """Función principal para la aplicación Streamlit."""
    st.title("Interpolación Automática de Valores Faltantes")

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
            
            # Mostrar resultados
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

        except Exception as e:
            st.error(f"Error en procesamiento: {str(e)}")


if __name__ == "__main__":
    main()
