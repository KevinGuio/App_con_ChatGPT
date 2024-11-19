import streamlit as st
import random

# Definir los emojis que usaremos en el juego
EMOJIS = ["🍎", "🍌", "🍉", "🍓", "🍍", "🍒", "🥭", "🍑"]

# Mezclar los emojis y duplicarlos para hacer las parejas
def generar_tablero():
    """Genera un tablero con las cartas mezcladas"""
    tablero = EMOJIS + EMOJIS
    random.shuffle(tablero)
    return tablero

# Mostrar el tablero en formato matriz 4x4
def mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas, cartas_temporalmente_destapadas):
    """Muestra el tablero con las cartas destapadas o cubiertas"""
    st.write("### Tablero de juego:")
    for i in range(4):  # Filas
        cols = st.columns(4)  # Crear columnas para cada fila
        for j in range(4):  # Columnas
            carta = i * 4 + j
            with cols[j]:
                if carta in cartas_destapadas:
                    st.write(tablero[carta])  # Mostrar emoji permanente
                elif carta in cartas_temporalmente_destapadas:
                    st.write(tablero[carta])  # Mostrar emoji temporalmente
                else:
                    if st.button(f"Card {carta}", key=carta):
                        st.session_state.cartas_seleccionadas.append(carta)

# Función principal del juego
def juego():
    st.title("Paremoji")
    st.write("""
    **Cómo jugar:**
    - Encuentra todas las parejas de emojis.
    - Haz clic en dos cartas para destaparlas.
    - Si las cartas son una pareja, se quedan destapadas.
    - Si no, aparecerá un botón para volver a taparlas.
    - ¡Gana al encontrar todas las parejas!
    """)
    st.write("Juego creado por **Kevin Guio**")

    # Inicializar las variables del juego
    if 'tablero' not in st.session_state:
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.cartas_seleccionadas = []
        st.session_state.cartas_temporalmente_destapadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    cartas_seleccionadas = st.session_state.cartas_seleccionadas
    cartas_temporalmente_destapadas = st.session_state.cartas_temporalmente_destapadas
    intentos = st.session_state.intentos
    parejas_encontradas = st.session_state.parejas_encontradas

    # Mostrar el tablero de juego
    mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas, cartas_temporalmente_destapadas)

    # Comprobar las cartas seleccionadas
    if len(cartas_seleccionadas) == 2:
        carta1, carta2 = cartas_seleccionadas
        cartas_temporalmente_destapadas.extend(cartas_seleccionadas)

        if tablero[carta1] == tablero[carta2]:
            st.write(f"¡Pareja encontrada! {tablero[carta1]} y {tablero[carta2]}")
            cartas_destapadas.extend(cartas_seleccionadas)
            st.session_state.parejas_encontradas += 1
        else:
            st.write(f"No es pareja: {tablero[carta1]} y {tablero[carta2]}")
            if st.button("Voltear cartas"):
                st.session_state.cartas_temporalmente_destapadas = []
                st.session_state.cartas_seleccionadas = []

        st.session_state.intentos += 1
        st.session_state.cartas_seleccionadas = []

    # Verificar progreso
    st.write(f"Parejas encontradas: {parejas_encontradas}")
    st.write(f"Intentos realizados: {intentos}")

    # Verificar si el juego ha terminado
    if len(cartas_destapadas) == len(tablero):
        st.write("🎉 ¡Felicidades! Has encontrado todas las parejas.")

    # Reiniciar juego
    if st.button("Reiniciar Juego"):
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.cartas_seleccionadas = []
        st.session_state.cartas_temporalmente_destapadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0

if __name__ == "__main__":
    juego()
