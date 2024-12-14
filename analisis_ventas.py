import streamlit as st
import numpy as np
import pandas as pd
import requests
from io import StringIO

# Función para cargar los datos desde el repositorio de GitHub
def load_data(url):
    """
    Carga el archivo CSV desde un repositorio de GitHub.
    
    Args:
    url (str): URL del archivo CSV en el repositorio de GitHub.
    
    Returns:
    pd.DataFrame: DataFrame con los datos cargados.
    """
    response = requests.get(url)
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    return df

# Función para filtrar pedidos por estado
def filter_by_status(df, status):
    """
    Filtra los pedidos según el estado (pendiente, enviado, entregado).
    
    Args:
    df (pd.DataFrame): DataFrame con los datos de los pedidos.
    status (str): Estado de los pedidos a filtrar.
    
    Returns:
    pd.DataFrame: DataFrame filtrado con los pedidos en el estado solicitado.
    """
    return df[df['Estado'] == status]

# Función para calcular el tiempo promedio de entrega en días
def average_delivery_time(df):
    """
    Calcula el tiempo promedio de entrega de los pedidos ya entregados.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos de los pedidos.
    
    Returns:
    float: Promedio de días de entrega.
    """
    # Convertir las fechas a formato datetime
    df['Fecha_Pedido'] = pd.to_datetime(df['Fecha_Pedido'])
    df['Fecha_Entrega'] = pd.to_datetime(df['Fecha_Entrega'], errors='coerce')  # Manejo de errores en la conversión
    
    # Filtrar solo los pedidos entregados
    delivered_orders = df[df['Estado'] == 'Entregado']
    
    # Calcular la diferencia en días
    delivery_days = (delivered_orders['Fecha_Entrega'] - delivered_orders['Fecha_Pedido']).dt.days
    
    # Retornar el promedio
    return np.mean(delivery_days) if len(delivery_days) > 0 else 0

# Función para generar el informe descargable
def generate_report(df):
    """
    Genera un informe descargable con los datos de los pedidos.
    
    Args:
    df (pd.DataFrame): DataFrame con los datos de los pedidos.
    
    Returns:
    str: Enlace para descargar el archivo CSV.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="Descargar informe",
        data=csv,
        file_name="informe_pedidos.csv",
        mime="text/csv"
    )

# Aplicación Streamlit
def app():
    # URL del archivo CSV en el repositorio de GitHub
    url = 'https://raw.githubusercontent.com/gabrielawad/programacion-para-ingenieria/refs/heads/main/archivos-datos/pandas/pedidos_ecommerce.csv'

    # Cargar los datos
    df = load_data(url)

    # Título de la aplicación
    st.title("Seguimiento de Pedidos de Ecommerce")

    # Mostrar los primeros registros
    st.subheader("Primeros 5 pedidos:")
    st.write(df.head())

    # Filtrar pedidos por estado
    status = st.selectbox("Selecciona el estado de los pedidos", ['Pendiente', 'Enviado', 'Entregado'])
    filtered_df = filter_by_status(df, status)
    
    # Mostrar los pedidos filtrados
    st.subheader(f"Pedidos {status}:")
    st.write(filtered_df)

    # Calcular y mostrar el tiempo promedio de entrega
    avg_delivery_time = average_delivery_time(df)
    st.subheader("Tiempo promedio de entrega (en días):")
    st.write(f"{avg_delivery_time:.2f} días")

    # Generar el informe descargable
    generate_report(df)

    # Créditos
    st.markdown("""
    ---
    **Esta aplicación fue programada por Kevin Guio**
    """)

# Ejecutar la aplicación
if __name__ == '__main__':
    app()
