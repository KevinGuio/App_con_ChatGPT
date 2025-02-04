import streamlit as st
from PIL import Image
import random

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Título de la aplicación
st.markdown('<h1 class="title">¿Puedo ser tu Valentín?</h1>', unsafe_allow_html=True)

# Imágenes de corazones y flores
heart_image = Image.open("heart.png")  # Asegúrate de tener una imagen de corazón en el mismo directorio
flower_image = Image.open("flower.png")  # Asegúrate de tener una imagen de flor en el mismo directorio

# Mostrar imágenes decorativas
col1, col2, col3 = st.columns(3)
with col1:
    st.image(flower_image, width=100)
with col2:
    st.image(heart_image, width=100)
with col3:
    st.image(flower_image, width=100)

# Botones de respuesta
col1, col2 = st.columns(2)
with col1:
    if st.button("¡Sí! ❤️"):
        st.balloons()
        st.success("¡Eres la mejor! ¡Feliz San Valentín! ❤️")
        st.image(heart_image, width=200)
with col2:
    if st.button("No 😢"):
        st.image(flower_image, width=200)
        st.write("¿Segura? ¡Dale otra oportunidad al botón de 'Sí'! 😉")

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
