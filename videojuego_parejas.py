import streamlit as st
import random

# Lista de emojis para el juego
EMOJIS = ["🍎", "🍌", "🍉", "🍓", "🍍", "🍒", "🥭", "🍑"]

# Generar el tablero con parejas de emojis mezclados
def generar_tablero():
    """Genera un tablero con parejas de emojis mezcladas."""
    tablero = EMOJIS + EMOJIS  # Duplicar emojis para formar parejas
    random.shuffle(tablero)  # Mezclar las cartas
    return tablero

# Inicialización de estado
if "tablero" not in st.session_state:
    st.session_state.tablero = generar_tablero()
if "seleccionadas" not in st.session_state:
    st.session_state.seleccionadas = []
if "cartas_volteadas" not in st.session_state:
    st.session_state.cartas_volteadas = [False] * len(st.session_state.tablero)

# Función para manejar la selección de cartas
def seleccionar_carta(indice):
    """Selecciona una carta del tablero."""
    if len(st.session_state.seleccionadas) < 2 and indice not in st.session_state.seleccionadas:
        st.session_state.seleccionadas.append(indice)

# Función para voltear cartas
def voltear_cartas():
    """Voltea las cartas seleccionadas."""
    seleccionadas = st.session_state.seleccionadas
    if len(seleccionadas) == 2:
        # Mostrar las cartas seleccionadas
        for indice in seleccionadas:
            st.session_state.cartas_volteadas[indice] = True
        
        # Revisar si son pareja
        carta1, carta2 = seleccionadas
        if st.session_state.tablero[carta1] != st.session_state.tablero[carta2]:
            # No son pareja: tapar nuevamente
            st.session_state.cartas_volteadas[carta1] = False
            st.session_state.cartas_volteadas[carta2] = False
        
        # Limpiar selección
        st.session_state.seleccionadas = []

# Título e instrucciones
st.title("Paremoji")
st.write("**Juego creado por Kevin Guio**")
st.write("Haz clic en dos cartas para seleccionarlas. Luego presiona 'Voltear cartas' para ver si son pareja.")

# Tablero de juego
cols = st.columns(4)  # Mostrar el tablero en 4 columnas
for i, emoji in enumerate(st.session_state.tablero):
    if st.session_state.cartas_volteadas[i]:
        # Mostrar emoji si está volteada
        cols[i % 4].button(emoji, key=f"emoji_{i}", disabled=True)
    else:
        # Mostrar botón tapado si no está volteada
        if cols[i % 4].button("🂠", key=f"card_{i}"):
            seleccionar_carta(i)

# Botón para voltear cartas
if st.button("Voltear cartas"):
    voltear_cartas()

# Mostrar las cartas seleccionadas solo después de presionar "Voltear cartas"
if len(st.session_state.seleccionadas) == 2:
    seleccionadas = st.session_state.seleccionadas
    carta1, carta2 = seleccionadas
    st.write(f"Has seleccionado: {st.session_state.tablero[carta1]} y {st.session_state.tablero[carta2]}")
