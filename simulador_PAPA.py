import streamlit as st
import random

# Definir los emojis que usaremos en el juego
EMOJIS = ["😊", "😎", "😍", "🤣", "🤔", "😜", "😇", "😎", "😈", "🎉", "🍀", "🍕"]

# Mezclar los emojis y duplicarlos para hacer las parejas
def generar_tablero():
    tablero = EMOJIS + EMOJIS
    random.shuffle(tablero)
    return tablero

# Función para mostrar el tablero de juego en formato matriz
def mostrar_tablero(tablero, cartas_destapadas):
    """ Muestra el tablero en forma de matriz 4x4 con cartas destapadas o cubiertas """
    st.write("### Tablero de juego:")
    botones = []
    for i in range(4):  # Filas
        fila = []
        for j in range(4):  # Columnas
            carta = i * 4 + j
            if carta in cartas_destapadas:
                fila.append(tablero[carta])  # Si está destapada, mostramos el emoji
            else:
                fila.append("❓")  # Si está tapada, mostramos el signo de interrogación
        botones.append(fila)
    return botones

# Función principal del juego
def juego():
    st.title("Paremoji")
    st.write("""
    **Cómo jugar:**
    - Encuentra todas las parejas de emojis.
    - Haz clic en dos cartas para destaparlas.
    - Si las cartas son una pareja, se quedan destapadas.
    - Si no, se vuelven a tapar.
    - ¡Gana al encontrar todas las parejas!
    """)
    st.write("Juego creado por **Kevin Guio**")
    
    # Inicializar las variables del juego
    if 'tablero' not in st.session_state:
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0
        st.session_state.cartas_seleccionadas = []

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    intentos = st.session_state.intentos
    parejas_encontradas = st.session_state.parejas_encontradas
    cartas_seleccionadas = st.session_state.cartas_seleccionadas

    # Mostrar el tablero de juego en formato matriz
    botones = mostrar_tablero(tablero, cartas_destapadas)

    # Interacción con el usuario: seleccionar cartas
    selected_card = None
    for i in range(4):
        for j in range(4):
            carta = i * 4 + j
            if st.button(f"Card {carta}", key=carta):
                selected_card = carta
                if selected_card not in cart
