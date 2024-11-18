import streamlit as st
import pandas as pd

# Descripción de la app
st.title("Cálculo del Promedio Académico Ponderado Acumulado (PAPA)")
st.write("Esta app fue elaborada por Kevin Guio.")
st.write("""
Esta aplicación permite calcular el **Promedio Académico Ponderado Acumulado (PAPA)** de un semestre.
Puedes ingresar las materias vistas, las calificaciones obtenidas y los créditos de cada asignatura.
También podrás calcular el PAPA global y por tipología de asignatura.
""")

# Tipologías de asignatura
tipologias = [
    "Disciplinar Optativa",
    "Fundamentación Obligatoria",
    "Fundamentación Optativa",
    "Disciplinar Obligatoria",
    "Libre Elección",
    "Trabajo de Grado",
    "Nivelación"
]

# Función para calcular el PAPA global
def calcular_papa_global(df):
    """
    Calcula el Promedio Académico Ponderado Acumulado global (PAPA) usando los créditos y las calificaciones.
    
    Parameters:
        df (DataFrame): DataFrame que contiene las materias, sus calificaciones y créditos.
    
    Returns:
        float: El PAPA global.
    """
    # Validar si el DataFrame tiene los datos necesarios
    if df.empty:
        return 0.0
    
    # Calcular el total de créditos
    total_creditos = df["Créditos"].sum()
    
    # Calcular el total ponderado (calificación * créditos)
    total_ponderado = (df["Calificación"] * df["Créditos"]).sum()
    
    # Calcular el PAPA global
    papa_global = total_ponderado / total_creditos if total_creditos > 0 else 0.0
    
    return round(papa_global, 2)

# Función para calcular el PAPA por tipología
def calcular_papa_por_tipologia(df, tipologia):
    """
    Calcula el Promedio Académico Ponderado Acumulado por tipología de asignatura.
    
    Parameters:
        df (DataFrame): DataFrame que contiene las materias, sus calificaciones y créditos.
        tipologia (str): La tipología de la asignatura para la cual se desea calcular el PAPA.
    
    Returns:
        float: El PAPA para la tipología seleccionada.
    """
    df_tipologia = df[df["Tipología"] == tipologia]
    
    return calcular_papa_global(df_tipologia)

# Formulario para ingresar materias, calificaciones, créditos y tipología
st.sidebar.header("Ingrese los datos de sus materias")
materias = []
calificaciones = []
creditos = []
tipologias_seleccionadas = []

# Crear formulario
num_materias = st.sidebar.number_input("Número de materias:", min_value=1, max_value=20, value=1)

for i in range(num_materias):
    st.sidebar.subheader(f"Materia {i+1}")
    materia = st.sidebar.text_input(f"Nombre de la materia {i+1}")
    calificacion = st.sidebar.number_input(f"Calificación de {materia}", min_value=0.0, max_value=5.0, step=0.1)
    credito = st.sidebar.number_input(f"Créditos de {materia}", min_value=1, max_value=6, value=3)
    tipologia = st.sidebar.selectbox(f"Tipología de {materia}", tipologias)
    
    materias.append(materia)
    calificaciones.append(calificacion)
    creditos.append(credito)
    tipologias_seleccionadas.append(tipologia)

# Crear DataFrame con los datos ingresados
df = pd.DataFrame({
    "Materia": materias,
    "Calificación": calificaciones,
    "Créditos": creditos,
    "Tipología": tipologias_seleccionadas
})

# Mostrar los datos ingresados
st.subheader("Materias ingresadas")
st.write(df)

# Calcular el PAPA global
papa_global = calcular_papa_global(df)
st.subheader(f"PAPA Global: {papa_global}")

# Calcular el PAPA por tipología
st.subheader("PAPA por tipología")
for tipologia in tipologias:
    papa_tipologia = calcular_papa_por_tipologia(df, tipologia)
    st.write(f"{tipologia}: {papa_tipologia}")

