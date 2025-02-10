import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd

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
    """Fill missing values in the DataFrame based on logical rules.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()

    # Fill missing 'Nombre' with 'N/A'
    df_filled['Nombre'] = df_filled['Nombre'].fillna('N/A')

    # Fill missing 'Género' with 'No binario'
    df_filled['Género'] = df_filled['Género'].fillna('No binario')

    # Fill missing 'Edad' using median grouped by other columns
    df_filled['Edad'] = df_filled['Edad'].fillna(
        df.groupby(['Género', 'Frecuencia_Compra', 'Ingreso_Anual_USD'])['Edad'].transform('median')
    )

    # Fill missing 'Ingreso_Anual_USD' using median grouped by other columns
    df_filled['Ingreso_Anual_USD'] = df_filled['Ingreso_Anual_USD'].fillna(
        df.groupby(['Edad', 'Género', 'Frecuencia_Compra'])['Ingreso_Anual_USD'].transform('median')
    )

    # Fill missing 'Historial_Compras' using median grouped by other columns
    df_filled['Historial_Compras'] = df_filled['Historial_Compras'].fillna(
        df.groupby(['Edad', 'Género', 'Frecuencia_Compra'])['Historial_Compras'].transform('median')
    )

    # Fill missing 'Frecuencia_Compra' with the mode
    df_filled['Frecuencia_Compra'] = df_filled['Frecuencia_Compra'].fillna(
        df['Frecuencia_Compra'].mode()[0]
    )

    # Fill missing 'Latitud' and 'Longitud' using median grouped by other columns
    df_filled['Latitud'] = df_filled['Latitud'].fillna(
        df.groupby(['Edad', 'Género', 'Ingreso_Anual_USD'])['Latitud'].transform('median')
    )
    df_filled['Longitud'] = df_filled['Longitud'].fillna(
        df.groupby(['Edad', 'Género', 'Ingreso_Anual_USD'])['Longitud'].transform('median')
    )

    return df_filled

def main():
    st.title("Análisis de Datos de Clientes")

    # File or URL upload
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

        # Fill missing values
        df_filled = fill_missing_values(df)

        st.subheader("Datos con valores faltantes rellenados")
        st.dataframe(df_filled)

if __name__ == "__main__":
    main()
