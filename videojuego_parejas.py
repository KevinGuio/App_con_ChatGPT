import streamlit as st
import random

# Definir los emojis que usaremos en el juego
EMOJIS = ["🍎", "🍌", "🍉", "🍓", "🍍", "🍒", "🥭", "🍑", "🍊", "🍇", "🍋", "🍒"]

# Mezclar los emojis y duplicarlos para hacer las parejas
def generar_tablero():
    """Genera un tablero con las cartas mezcladas"""
    tablero = EMOJIS + EMOJIS
    random.shuffle(tablero)
    return tablero

# Función para mostrar el tablero de juego en formato matriz 4x4
def mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas):
    """ Muestra el tablero en forma de matriz 4x4 con cartas destapadas o cubiertas """
    st.write("### Tablero de juego:")
    botones = []
    for i in range(4):  # Filas
        fila = []
        for j in range(4):  # Columnas
            carta = i * 4 + j
            if carta in cartas_destapadas or carta in cartas_seleccionadas:
                fila.append(tablero[carta])  # Si está destapada, mostramos el emoji
            else:
                fila.append("❓")  # Si está tapada, mostramos el signo de interrogación
        botones.append(fila)

    # Mostrar el tablero en forma de botones interactivos
    for i in range(4):
        cols = st.columns(4)  # Crear 4 columnas para mostrar las cartas
        for j in range(4):
            carta = i * 4 + j
            with cols[j]:
                if carta in cartas_destapadas or carta in cartas_seleccionadas:
                    st.write(tablero[carta])
                else:
                    if st.button(f"Card {carta}", key=carta):
                        st.session_state.cartas_seleccionadas.append(carta)
                        st.session_state.ultimo_emoji = tablero[carta]  # Guardamos el emoji seleccionado

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
        st.session_state.ultimo_emoji = None  # Para mostrar el último emoji seleccionado

    tablero = st.session_state.tablero
    cartas_destapadas = st.session_state.cartas_destapadas
    intentos = st.session_state.intentos
    parejas_encontradas = st.session_state.parejas_encontradas
    cartas_seleccionadas = st.session_state.cartas_seleccionadas
    ultimo_emoji = st.session_state.ultimo_emoji

    # Mostrar el tablero de juego en formato matriz
    mostrar_tablero(tablero, cartas_destapadas, cartas_seleccionadas)

    # Mostrar el emoji seleccionado inmediatamente al presionar un botón
    if ultimo_emoji:
        st.write(f"Has seleccionado el emoji: {ultimo_emoji}")

    # Comprobar las cartas seleccionadas
    if len(cartas_seleccionadas) == 2:
        carta1, carta2 = cartas_seleccionadas
        st.write(f"Has seleccionado: {tablero[carta1]} y {tablero[carta2]}")  # Mostrar las dos cartas seleccionadas
        if tablero[carta1] == tablero[carta2]:
            st.session_state.cartas_destapadas.append(carta1)
            st.session_state.cartas_destapadas.append(carta2)
            st.session_state.parejas_encontradas += 1
            st.write(f"¡Encontraste una pareja! {tablero[carta1]} y {tablero[carta2]}")
        else:
            st.write("¡No es una pareja!")
        
        # Incrementar intentos
        st.session_state.intentos += 1

        # Resetear las cartas seleccionadas
        st.session_state.cartas_seleccionadas = []
        st.session_state.ultimo_emoji = None  # Reseteamos el emoji mostrado

    # Ver el progreso
    st.write(f"Parejas encontradas: {parejas_encontradas}")
    st.write(f"Intentos realizados: {intentos}")

    # Comprobar si el juego ha terminado
    if len(st.session_state.cartas_destapadas) == len(tablero):
        st.write("¡Felicidades! Has encontrado todas las parejas.")

    # Reiniciar el juego
    if st.button("Reiniciar Juego"):
        st.session_state.tablero = generar_tablero()
        st.session_state.cartas_destapadas = []
        st.session_state.intentos = 0
        st.session_state.parejas_encontradas = 0
        st.session_state.cartas_seleccionadas = []
        st.session_state.ultimo_emoji = None

if __name__ == "__main__":
    juego()
