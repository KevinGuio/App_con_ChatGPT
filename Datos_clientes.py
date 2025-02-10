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
    """Fill missing values in the DataFrame using binning for grouping.

    Se crean bins para las columnas numéricas 'Ingreso_Anual_USD',
    'Historial_Compras' y 'Edad' utilizando pd.qcut. Para la columna
    'Frecuencia_Compra', que es categórica, se convierte a códigos numéricos.
    Luego se rellenan los valores faltantes mediante agrupaciones basadas
    en dichos bins.

    Args:
        df (pd.DataFrame): Input DataFrame with missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.
    """
    df_filled = df.copy()

    # Rellenar columnas de texto con valores fijos
    df_filled['Nombre'] = df_filled['Nombre'].fillna('N/A')
    df_filled['Género'] = df_filled['Género'].fillna('No binario')

    # Crear bins para columnas numéricas
    df_filled['Ingreso_bin'] = pd.qcut(
        df_filled['Ingreso_Anual_USD'], q=4, duplicates='drop'
    )
    df_filled['Historial_bin'] = pd.qcut(
        df_filled['Historial_Compras'], q=4, duplicates='drop'
    )
    df_filled['Edad_bin'] = pd.qcut(
        df_filled['Edad'], q=4, duplicates='drop'
    )

    # Para 'Frecuencia_Compra': si es numérica se utiliza qcut;
    # de lo contrario se convierte a código numérico.
    df_filled['Frecuencia_bin'] = (
        pd.qcut(df_filled['Frecuencia_Compra'], q=4, duplicates='drop')
        if pd.api.types.is_numeric_dtype(df_filled['Frecuencia_Compra'])
        else df_filled['Frecuencia_Compra'].astype('category').cat.codes
    )

    # Rellenar 'Edad' usando grupo por Género y los bins de Ingreso, Historial y Frecuencia
    df_filled['Edad'] = df_filled['Edad'].fillna(
        df_filled.groupby(
            ['Género', 'Ingreso_bin', 'Historial_bin', 'Frecuencia_bin']
        )['Edad'].transform('median')
    )

    # Rellenar 'Ingreso_Anual_USD' usando grupo por Edad, Género, Historial y Frecuencia
    df_filled['Ingreso_Anual_USD'] = df_filled['Ingreso_Anual_USD'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Historial_bin', 'Frecuencia_bin']
        )['Ingreso_Anual_USD'].transform('median')
    )

    # Rellenar 'Historial_Compras' usando grupo por Edad, Género, Ingreso y Frecuencia
    df_filled['Historial_Compras'] = df_filled['Historial_Compras'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Frecuencia_bin']
        )['Historial_Compras'].transform('median')
    )

    # Rellenar 'Frecuencia_Compra' usando la moda en grupos formados por Edad, Género, Ingreso y Historial
    df_filled['Frecuencia_Compra'] = df_filled['Frecuencia_Compra'].fillna(
        df_filled.groupby(
            ['Edad_bin', 'Género', 'Ingreso_bin', 'Historial_bin']
        )['Frecuencia_Compra'].transform(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    # Rellenar 'Latitud' y 'Longitud' usando grupo por Edad, Género, Ingreso, Historial y Frecuencia
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

    # Eliminar las columnas temporales de bins
    df_filled = df_filled.drop(
        columns=['Ingreso_bin', 'Historial_bin', 'Frecuencia_bin', 'Edad_bin']
    )

    # Paso global de respaldo para cualquier valor faltante restante
    df_filled = df_filled.fillna({
        'Edad': df['Edad'].median() if pd.api.types.is_numeric_dtype(df['Edad']) else None,
        'Ingreso_Anual_USD': df['Ingreso_Anual_USD'].median() if pd.api.types.is_numeric_dtype(df['Ingreso_Anual_USD']) else None,
        'Historial_Compras': df['Historial_Compras'].median() if pd.api.types.is_numeric_dtype(df['Historial_Compras']) else None,
        'Latitud': df['Latitud'].median() if pd.api.types.is_numeric_dtype(df['Latitud']) else None,
        'Longitud': df['Longitud'].median() if pd.api.types.is_numeric_dtype(df['Longitud']) else None
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

        # Rellenar valores faltantes usando binning y agrupaciones
        df_filled = fill_missing_values(df)

        st.subheader("Datos con valores faltantes rellenados")
        st.dataframe(df_filled)

if __name__ == "__main__":
    main()
