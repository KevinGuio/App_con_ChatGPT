import streamlit as st
import random
import time

# Definir los emojis que usaremos en el juego
EMOJIS = ["😊", "😎", "😍", "🤣", "🤔", "😜", "😇", "😎", "😈", "🎉", "🍀", "🍕"]

# Mezclar los emojis y duplicarlos para hacer las parejas
def generar_tablero():
    tablero = EMOJIS + EMOJIS
    random.shuffle(tablero)
    return tablero

# Mostrar el juego
def mostrar_tablero(tablero, cartas_destapadas):
    """ Muestra el tablero con las cartas destapadas o cubiertas """
    for i in range(len(tablero)):
        if i in cartas_destapadas:
            st.write(f"**{tablero[i]}**", end="   ")
        else:
            st.write("❓", end="   ")
        if (i + 1) % 4 == 0:
            st.write()  # Saltar una línea después de cada fila de 4 cartas

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

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    intentos = st.session_state.intentos
    parejas_encontradas = st.session_state.parejas_encontradas

    mostrar_tablero(tablero, cartas_destapadas)

    # Seleccionar dos cartas para destapar
    carta1 = st.selectbox("Selecciona la primera carta", list(range(16)), key="carta1")
    carta2 = st.selectbox("Selecciona la segunda carta", list(range(16)), key="carta2")

    if st.button("Comprobar"):
        st.session_state.intentos += 1

        # Verificar si las cartas seleccionadas son una pareja
        if tablero[carta1] == tablero[carta2] and carta1 != carta2:
            st.session_state.parejas_encontradas += 1
            st.session_state.cartas_destapadas.append(carta1)
            st.session_state.cartas_destapadas.append(carta2)
            st.write(f"¡Es una pareja! 💖")
        else:
            st.write("¡No es una pareja! 😥")
        
        # Actualizar el estado de las cartas destapadas
        if len(st.session_state.cartas_destapadas) == len(tablero):
            st.write(f"¡Felicidades! Has encontrado todas las parejas en {intentos} intentos.")

    # Ver el progreso
    st.write(f"Parejas encontradas: {parejas_encontradas}")
    st.write(f"Intentos realizados: {intentos}")

    # Reiniciar el juego
    if st.button("Reiniciar Juego"):
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0

if __name__ == "__main__":
    juego()
