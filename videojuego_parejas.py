import streamlit as st
import random

# Lista de emojis para el juego
EMOJIS = ["🍎", "🍌", "🍉", "🍓", "🍍", "🍒", "🥭", "🍑"]

# Generar el tablero con parejas de emojis mezcladas
def generar_tablero():
    """Genera un tablero con parejas de emojis mezcladas."""
    tablero = EMOJIS + EMOJIS  # Duplicar emojis para formar parejas
    random.shuffle(tablero)  # Mezclar las cartas
    return tablero

# Mostrar el tablero de juego
def mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas):
    """Muestra el tablero en formato de matriz 4x4."""
    st.write("### Tablero de juego:")
    for i in range(4):  # Filas
        cols = st.columns(4)  # Crear columnas
        for j in range(4):  # Columnas
            carta = i * 4 + j
            with cols[j]:
                # Mostrar el emoji si la carta está destapada o seleccionada
                if carta in cartas_destapadas or carta in cartas_seleccionadas:
                    st.write(tablero[carta])
                else:
                    # Botón de carta oculta
                    st.button("🂠", key=f"card_{carta}", on_click=seleccionar_carta, args=(carta,))

# Función para manejar la selección de cartas
def seleccionar_carta(indice):
    """Selecciona una carta del tablero."""
    if len(st.session_state.cartas_seleccionadas) < 2 and indice not in st.session_state.cartas_seleccionadas:
        st.session_state.cartas_seleccionadas.append(indice)

# Función principal del juego
def juego():
    st.title("Paremoji")
    st.write("""
    **Cómo jugar:**
    - Encuentra todas las parejas de emojis.
    - Haz clic en dos cartas para seleccionarlas.
    - Presiona el botón **Voltear cartas** para ver si son pareja.
    - ¡Encuentra todas las parejas para ganar!
    """)
    st.write("Juego creado por **Kevin Guio**")

    # Inicializar el estado del juego
    if "tablero" not in st.session_state:
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.cartas_seleccionadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0
        st.session_state.mensaje = ""

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    cartas_seleccionadas = st.session_state.cartas_seleccionadas

    # Mostrar el tablero
    mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas)

    # Botón para voltear cartas
    if st.button("Verificar"):
        if len(cartas_seleccionadas) == 2:
            carta1, carta2 = cartas_seleccionadas
            st.session_state.intentos += 1

            # Comprobar si las cartas son pareja
            if tablero[carta1] == tablero[carta2]:
                st.session_state.mensaje = f"¡Pareja encontrada! {tablero[carta1]} y {tablero[carta2]}"
                st.session_state.cartas_destapadas.extend(cartas_seleccionadas)
                st.session_state.parejas_encontradas += 1
            else:
                st.session_state.mensaje = f"No es pareja: {tablero[carta1]} y {tablero[carta2]}"
            # Limpiar las cartas seleccionadas
            st.session_state.cartas_seleccionadas = []

    # Mostrar mensaje de resultado
    if st.session_state.mensaje:
        st.write(st.session_state.mensaje)

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
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0
        st.session_state.mensaje = ""

if __name__ == "__main__":
    juego()
