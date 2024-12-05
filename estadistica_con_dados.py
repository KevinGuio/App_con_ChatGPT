import numpy as np
import pandas as pd
import streamlit as st

def lanzar_dado(n_lanzamientos: int) -> np.ndarray:
    """
    Simula el lanzamiento de un dado `n_lanzamientos` veces.

    Args:
        n_lanzamientos (int): Número de lanzamientos a realizar.

    Returns:
        np.ndarray: Resultados de los lanzamientos.

    Example:
        >>> lanzar_dado(5)
        array([1, 6, 3, 2, 4])
    """
    return np.random.randint(1, 7, size=n_lanzamientos)

def analizar_resultados(resultados: np.ndarray) -> pd.DataFrame:
    """
    Calcula estadísticas descriptivas y frecuencias de los resultados.

    Args:
        resultados (np.ndarray): Resultados de los lanzamientos.

    Returns:
        pd.DataFrame: Tabla con estadísticas descriptivas y frecuencias.

    Example:
        >>> analizar_resultados(np.array([1, 2, 2, 3]))
        Estadísticas descriptivas y frecuencias.
    """
    # Estadísticas descriptivas
    media = np.mean(resultados)
    mediana = np.median(resultados)
    moda = np.bincount(resultados).argmax()
    varianza = np.var(resultados)
    desviacion_estandar = np.std(resultados)
    
    # Frecuencias
    valores, frecuencias = np.unique(resultados, return_counts=True)
    tabla_frecuencias = pd.DataFrame({
        "Número": valores,
        "Frecuencia": frecuencias
    })
    tabla_frecuencias["Porcentaje"] = (frecuencias / resultados.size) * 100
    
    # Consolidar datos
    resumen = pd.DataFrame({
        "Estadística": ["Media", "Mediana", "Moda", "Varianza", "Desv. Est."],
        "Valor": [media, mediana, moda, varianza, desviacion_estandar]
    })
    
    return resumen, tabla_frecuencias

def main():
    """
    Interfaz de usuario para simular lanzamientos de un dado y analizar resultados.
    """
    st.title("Simulación de Lanzamientos de un Dado 🎲")
    st.markdown("### Programado por Kevin Guio")
    
    # Número de lanzamientos
    n_lanzamientos = st.slider(
        "Número de lanzamientos:", min_value=10, max_value=100, value=20
    )
    
    # Simular lanzamientos
    resultados = lanzar_dado(n_lanzamientos)
    st.subheader("Resultados de los lanzamientos:")
    st.write(resultados)
    
    # Analizar resultados
    resumen, tabla_frecuencias = analizar_resultados(resultados)
    
    st.subheader("Estadísticas descriptivas:")
    st.table(resumen)
    
    st.subheader("Tabla de frecuencias:")
    st.table(tabla_frecuencias)
    
    # Descargar resultados
    st.download_button(
        label="Descargar análisis en CSV",
        data=tabla_frecuencias.to_csv(index=False),
        file_name="analisis_dados.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    main()
