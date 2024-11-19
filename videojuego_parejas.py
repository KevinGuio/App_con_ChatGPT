import streamlit as st
import random

# Lista de emojis para el juego
EMOJIS = ["🍎", "🍌", "🍉", "🍓", "🍍", "🍒", "🥭", "🍑"]

# Generar el tablero con emojis mezclados
def generar_tablero():
    """Genera un tablero con parejas de emojis mezcladas"""
    tablero = EMOJIS + EMOJIS  # Duplicar emojis para formar parejas
    random.shuffle(tablero)  # Mezclar las cartas
    return tablero

# Mostrar el tablero de juego
def mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas, bloqueadas):
    """Muestra el tablero en formato de matriz 4x4"""
    st.write("### Tablero de juego:")
    for i in range(4):  # Filas
        cols = st.columns(4)  # Crear columnas
        for j in range(4):  # Columnas
            carta = i * 4 + j
            with cols[j]:
                if carta in cartas_destapadas or carta in cartas_seleccionadas:
                    st.write(tablero[carta])  # Mostrar emoji
                elif carta in bloqueadas:
                    st.write(tablero[carta])  # Mostrar emoji de cartas bloqueadas
                else:
                    if st.button(f"Card {carta}", key=f"card_{carta}"):
                        st.session_state.cartas_seleccionadas.append(carta)

# Función principal del juego
def juego():
    st.title("Paremoji")
    st.write("""
    **Cómo jugar:**
    - Encuentra todas las parejas de emojis.
    - Haz clic en dos cartas para destaparlas.
    - Si son pareja, se quedan visibles.
    - Si no son pareja, usa el botón **Voltear cartas** para taparlas nuevamente.
    - ¡Encuentra todas las parejas para ganar!
    """)
    st.write("Juego creado por **Kevin Guio**")

    # Inicializar el estado del juego
    if 'tablero' not in st.session_state:
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.cartas_seleccionadas = []
        st.session_state.bloqueadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    cartas_seleccionadas = st.session_state.cartas_seleccionadas
    bloqueadas = st.session_state.bloqueadas

    # Mostrar el tablero
    mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas, bloqueadas)

    # Lógica del juego al seleccionar dos cartas
    if len(cartas_seleccionadas) == 2:
        carta1, carta2 = cartas_seleccionadas
        st.session_state.intentos += 1

        if tablero[carta1] == tablero[carta2]:
            st.write(f"¡Pareja encontrada! {tablero[carta1]} y {tablero[carta2]}")
            st.session_state.cartas_destapadas.extend(cartas_seleccionadas)
            st.session_state.parejas_encontradas += 1
            st.session_state.cartas_seleccionadas = []
        else:
            st.write(f"No es pareja: {tablero[carta1]} y {tablero[carta2]}")
            if st.button("Voltear cartas"):
                st.session_state.cartas_seleccionadas = []

    # Verificar progreso del juego
    st.write(f"Parejas encontradas: {st.session_state.parejas_encontradas}")
    st.write(f"Intentos realizados: {st.session_state.intentos}")

    if len(cartas_destapadas) == len(tablero):
        st.write("🎉 ¡Felicidades! Has encontrado todas las parejas.")

    # Botón para reiniciar el juego
    if st.button("Reiniciar Juego"):
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.cartas_seleccionadas = []
        st.session_state.bloqueadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0

if __name__ == "__main__":
    juego()
