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
    """Fill missing values in the DataFrame using binning for improved grouping.

    The method creates temporary bins (cuartiles) for the numeric columns used in
    the grouping keys (Ingreso_Anual_USD, Historial_Compras, Frecuencia_Compra y Edad)
    para poder agrupar de forma más eficiente y rellenar los valores faltantes.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()

    # Rellenar 'Nombre' y 'Género' con valores fijos
    df_filled['Nombre'] = df_filled['Nombre'].fillna('N/A')
    df_filled['Género'] = df_filled['Género'].fillna('No binario')

    # Crear bins para columnas numéricas utilizadas en la agrupación
    df_filled['Ingreso_bin'] = pd.qcut(
        df_filled['Ingreso_Anual_USD'], q=4, duplicates='drop'
    )
    df_filled['Historial_bin'] = pd.qcut(
        df_filled['Historial_Compras'], q=4, duplicates='drop'
    )
    df_filled['Frecuencia_bin'] = pd.qcut(
        df_filled['Frecuencia_Compra'], q=4, duplicates='drop'
    )

    # Rellenar 'Edad' utilizando agrupación por Género y los bins de Ingreso, Historial y Frecuencia
    df_filled['Edad'] = df_filled['Edad'].fillna(
        df_filled.groupby(
            ['Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin']
        )['Edad'].transform('median')
    )

    # Una vez rellenada la Edad, se genera un bin para ésta
    df_filled['Edad_bin'] = pd.qcut(
        df_filled['Edad'], q=4, duplicates='drop'
    )

    # Rellenar 'Ingreso_Anual_USD' agrupando por Edad_bin, Género, Historial_bin y Frecuencia_bin
    df_filled['Ingreso_Anual_USD'] = df_filled['Ingreso_Anual_USD'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Historial_bin', 'Frecuencia_bin']
        )['Ingreso_Anual_USD'].transform('median')
    )

    # Rellenar 'Historial_Compras' agrupando por Edad_bin, Género, Ingreso_bin y Frecuencia_bin
    df_filled['Historial_Compras'] = df_filled['Historial_Compras'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Frecuencia_bin']
        )['Historial_Compras'].transform('median')
    )

    # Rellenar 'Frecuencia_Compra' agrupando por Edad_bin, Género, Ingreso_bin y Historial_bin
    df_filled['Frecuencia_Compra'] = df_filled['Frecuencia_Compra'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin']
        )['Frecuencia_Compra'].transform(lambda x: x.mode().iloc[0])
    )

    # Rellenar 'Latitud' y 'Longitud' agrupando por Edad_bin, Género, Ingreso_bin, Historial_bin y Frecuencia_bin
    df_filled['Latitud'] = df_filled['Latitud'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin']
        )['Latitud'].transform('median')
    )
    df_filled['Longitud'] = df_filled['Longitud'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin']
        )['Longitud'].transform('median')
    )

    # Eliminar columnas temporales de bins
    df_filled = df_filled.drop(
        columns=['Ingreso_bin', 'Historial_bin', 'Frecuencia_bin', 'Edad_bin']
    )

    # Paso global de respaldo para cualquier valor faltante restante
    df_filled = df_filled.fillna({
        'Edad': df['Edad'].median(),
        'Ingreso_Anual_USD': df['Ingreso_Anual_USD'].median(),
        'Historial_Compras': df['Historial_Compras'].median(),
        'Latitud': df['Latitud'].median(),
        'Longitud': df['Longitud'].median()
    })

    return df_filled


def main():
    st.title("Análisis de Datos de Clientes")

    # Carga de datos desde archivo o URL
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

        # Rellenar valores faltantes usando la técnica de binning
        df_filled = fill_missing_values(df)

        st.subheader("Datos con valores faltantes rellenados")
        st.dataframe(df_filled)


if __name__ == "__main__":
    main()
