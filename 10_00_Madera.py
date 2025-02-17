import streamlit as st
import pandas as pd


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


def handle_missing_values(df, numeric_cols, method):
    """Rellena valores faltantes usando interpolación vectorizada.

    Args:
        df (pd.DataFrame): DataFrame original con datos faltantes.
        numeric_cols (list): Lista de columnas numéricas a procesar.
        method (str): Método de interpolación a utilizar.

    Returns:
        pd.DataFrame: DataFrame con valores faltantes interpolados.
    """
    df_filled = df.copy()
    df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method=method)
    return df_filled


def main():
    """Función principal para la aplicación Streamlit."""
    st.title("Interpolación de Valores Faltantes")

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

        # Detectar columnas numéricas
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if not numeric_cols:
            st.warning("No se encontraron columnas numéricas en el dataset")
        else:
            # Selección de parámetros de interpolación
            st.header("Parámetros de Interpolación")
            selected_cols = st.multiselect(
                "Columnas a interpolar", options=numeric_cols
            )

            method = st.selectbox(
                "Método de interpolación",
                options=[
                    "linear",
                    "time",
                    "index",
                    "nearest",
                    "zero",
                    "slinear",
                    "quadratic",
                    "cubic",
                ],
            )

            if selected_cols and method:
                # Aplicar interpolación
                try:
                    df_clean = handle_missing_values(df, selected_cols, method)
                    
                    # Mostrar resultados
                    st.header("Datos Procesados")
                    st.dataframe(df_clean)

                    # Comparativa de valores faltantes
                    st.subheader("Resumen de Valores Faltantes")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Original:")
                        st.write(df[selected_cols].isnull().sum())
                    
                    with col2:
                        st.write("Procesado:")
                        st.write(df_clean[selected_cols].isnull().sum())

                except Exception as e:
                    st.error(f"Error en interpolación: {str(e)}")


if __name__ == "__main__":
    main()
