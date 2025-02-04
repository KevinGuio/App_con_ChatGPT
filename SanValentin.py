import streamlit as st
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="San Valentín",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Estilos CSS para la página
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 20px;
        padding: 10px 24px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff1a1a;
        transform: scale(1.1);
    }
    .stApp {
        background-image: url("https://www.transparenttextures.com/patterns/flowers.png");
        background-size: cover;
    }
    .title {
        font-size: 50px;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 20px;
    }
    .heart {
        color: #ff4b4b;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .big-button {
        width: 100%;
        font-size: 30px !important;
        padding: 20px 40px !important;
        transition: all 0.5s ease;
    }
    .invitation {
        font-size: 25px;
        text-align: center;
        margin-top: 20px;
        color: #ff4b4b;
    }
    .hidden {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la aplicación
st.markdown('<h1 class="title">¿Puedo ser tu Valentín?</h1>', unsafe_allow_html=True)

# Imágenes de corazones y flores
heart_image = Image.open("heart.png")  # Asegúrate de tener una imagen de corazón en el mismo directorio
flower_image = Image.open("flower.png")  # Asegúrate de tener una imagen de flor en el mismo directorio
heart_image2 = Image.open("heart2.png")  # Otra imagen de corazón
flower_image2 = Image.open("flower2.png")  # Otra imagen de flor

# Mostrar imágenes decorativas
col1, col2, col3 = st.columns(3)
with col1:
    st.image(flower_image, width=100)
with col2:
    st.image(heart_image, width=100)
with col3:
    st.image(flower_image2, width=100)

# Estado de la sesión para controlar el crecimiento del botón "Sí" y la visibilidad de los botones
if 'button_grow' not in st.session_state:
    st.session_state.button_grow = False
if 'buttons_visible' not in st.session_state:
    st.session_state.buttons_visible = True

# Botones de respuesta
if st.session_state.buttons_visible:
    st.markdown('<div class="center">', unsafe_allow_html=True)

    # Botón "Sí"
    if st.button("¡Sí! ❤️"):
        st.session_state.button_grow = False
        st.session_state.buttons_visible = False  # Oculta los botones
        st.balloons()
        st.success("Sabía que lo harías, eres la mejor ❤️")
        st.markdown(
            """
            <div class="invitation">
                <h2>¡Te invito a una cita!</h2>
                <p>¿Qué tal si celebramos este día especial juntos?</p>
                <p>📅 Fecha: 14 de febrero</p>
                <p>⏰ Hora: 6:00 PM</p>
                <p>📍 Lugar: [Nombre del lugar]</p>
                <p>¡Espero que puedas acompañarme!</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Botón "No"
    if st.button("No 😢"):
        st.session_state.button_grow = True  # Activa el crecimiento del botón "Sí"

    # Aplicar el efecto de crecimiento al botón "Sí"
    if st.session_state.button_grow:
        st.markdown(
            """
            <style>
            .stButton>button {
                width: 100%;
                font-size: 30px !important;
                padding: 20px 40px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.write("¿Segura? ¡Dale otra oportunidad al botón de 'Sí'! 😉")

    st.markdown('</div>', unsafe_allow_html=True)

# Mostrar más imágenes después de presionar "Sí"
if not st.session_state.buttons_visible:
    st.markdown('<h2 style="text-align: center; color: #ff4b4b;">¡Gracias por aceptar! ❤️</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(heart_image2, width=150)
    with col2:
        st.image(flower_image, width=150)
    with col3:
        st.image(heart_image, width=150)

# Efecto de confeti al cargar la página
if 'confetti' not in st.session_state:
    st.session_state.confetti = True
    st.balloons()

# Pie de página
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px;">
        <p>Hecho con ❤️ por tu Valentín</p>
    </div>
    """,
    unsafe_allow_html=True,
)
