import streamlit as st

# Título de la app
st.title("Conversor Universal")

# Autor
st.write("Esta app fue elaborada por Kevin Guio.")

# Categorías disponibles
categorias = [
    "Temperatura",
    "Longitud",
    "Peso/Masa",
    "Volumen",
    "Tiempo",
    "Velocidad",
    "Área",
    "Energía",
    "Presión",
    "Tamaño de Datos"
]

# Diccionario con las conversiones y funciones
conversiones = {
    "Temperatura": {
        "Celsius a Fahrenheit": lambda x: (x * 9 / 5) + 32,
        "Fahrenheit a Celsius": lambda x: (x - 32) * 5 / 9,
        "Celsius a Kelvin": lambda x: x + 273.15,
        "Kelvin a Celsius": lambda x: x - 273.15
    },
    "Longitud": {
        "Pies a metros": lambda x: x * 0.3048,
        "Metros a pies": lambda x: x / 0.3048,
        "Pulgadas a centímetros": lambda x: x * 2.54,
        "Centímetros a pulgadas": lambda x: x / 2.54
    },
    "Peso/Masa": {
        "Libras a kilogramos": lambda x: x * 0.453592,
        "Kilogramos a libras": lambda x: x / 0.453592,
        "Onzas a gramos": lambda x: x * 28.3495,
        "Gramos a onzas": lambda x: x / 28.3495
    },
    "Volumen": {
        "Galones a litros": lambda x: x * 3.78541,
        "Litros a galones": lambda x: x / 3.78541,
        "Pulgadas cúbicas a centímetros cúbicos": lambda x: x * 16.3871,
        "Centímetros cúbicos a pulgadas cúbicas": lambda x: x / 16.3871
    },
    "Tiempo": {
        "Horas a minutos": lambda x: x * 60,
        "Minutos a segundos": lambda x: x * 60,
        "Días a horas": lambda x: x * 24,
        "Semanas a días": lambda x: x * 7
    },
    "Velocidad": {
        "Millas por hora a kilómetros por hora": lambda x: x * 1.60934,
        "Kilómetros por hora a metros por segundo": lambda x: x / 3.6,
        "Nudos a millas por hora": lambda x: x * 1.15078,
        "Metros por segundo a pies por segundo": lambda x: x * 3.28084
    },
    "Área": {
        "Metros cuadrados a pies cuadrados": lambda x: x * 10.7639,
        "Pies cuadrados a metros cuadrados": lambda x: x / 10.7639,
        "Kilómetros cuadrados a millas cuadradas": lambda x: x * 0.386102,
        "Millas cuadradas a kilómetros cuadrados": lambda x: x / 0.386102
    },
    "Energía": {
        "Julios a calorías": lambda x: x / 4.184,
        "Calorías a kilojulios": lambda x: x * 0.004184,
        "Kilovatios-hora a megajulios": lambda x: x * 3.6,
        "Megajulios a kilovatios-hora": lambda x: x / 3.6
    },
    "Presión": {
        "Pascales a atmósferas": lambda x: x / 101325,
        "Atmósferas a pascales": lambda x: x * 101325,
        "Barras a libras por pulgada cuadrada": lambda x: x * 14.5038,
        "Libras por pulgada cuadrada a bares": lambda x: x / 14.5038
    },
    "Tamaño de Datos": {
        "Megabytes a gigabytes": lambda x: x / 1024,
        "Gigabytes a Terabytes": lambda x: x / 1024,
        "Kilobytes a megabytes": lambda x: x / 1024,
        "Terabytes a petabytes": lambda x: x / 1024
    }
}

# Selección de categoría
categoria_seleccionada = st.selectbox("Selecciona una categoría:", categorias)

# Selección de tipo de conversión
if categoria_seleccionada:
    conversion_seleccionada = st.selectbox(
        "Selecciona el tipo de conversión:",
        list(conversiones[categoria_seleccionada].keys())
    )

    # Entrada del usuario
    valor = st.number_input("Ingresa el valor a convertir:", format="%.6f")

    # Realizar conversión
    if st.button("Convertir"):
        if conversion_seleccionada and valor is not None:
            resultado = conversiones[categoria_seleccionada][conversion_seleccionada](valor)
            st.write(f"Resultado: {resultado:.6f}")
