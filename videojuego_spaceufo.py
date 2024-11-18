import pygame
import random
import streamlit as st

# Inicialización de Pygame
pygame.init()

# Dimensiones de la ventana
ANCHO = 600
ALTO = 400

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE = (0, 255, 0)
ROJO = (255, 0, 0)

# Configuración de la ventana
pantalla = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("Space Alien - UFO Game")

# Fuentes
fuente_titulo = pygame.font.SysFont("arial", 40)
fuente_puntaje = pygame.font.SysFont("arial", 30)

# Variables del juego
ufo_ancho = 50
ufo_alto = 40
ufo_x = 100
ufo_y = ALTO // 2
velocidad_y = 0
gravedad = 0.5
saltando = False
velocidad = 5
meteoritos = []
meteorito_ancho = 50
meteorito_alto = 50
puntaje = 0

# Función para dibujar el UFO
def dibujar_ufo(x, y):
    pygame.draw.rect(pantalla, VERDE, (x, y, ufo_ancho, ufo_alto))

# Función para generar meteoritos
def generar_meteoritos():
    if random.randint(1, 100) <= 2:
        meteoritos.append([ANCHO, random.randint(100, ALTO-100)])

# Función para mover meteoritos
def mover_meteoritos():
    global puntaje
    for meteorito in meteoritos:
        meteorito[0] -= velocidad
        if meteorito[0] < 0:
            meteoritos.remove(meteorito)
            puntaje += 1

# Función para verificar colisiones
def verificar_colision(ufo_x, ufo_y):
    global meteoritos
    for meteorito in meteoritos:
        if (meteorito[0] < ufo_x + ufo_ancho and meteorito[0] + meteorito_ancho > ufo_x and
            meteorito[1] < ufo_y + ufo_alto and meteorito[1] + meteorito_alto > ufo_y):
            return True
    return False

# Función principal del juego
def juego():
    global ufo_y, velocidad_y, saltando, meteoritos, puntaje
    ufo_y = ALTO // 2
    velocidad_y = 0
    meteoritos = []
    puntaje = 0
    
    reloj = pygame.time.Clock()
    corriendo = True
    
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                corriendo = False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_SPACE and not saltando:
                    velocidad_y = -10
                    saltando = True
        
        # Movimiento del UFO
        ufo_y += velocidad_y
        velocidad_y += gravedad
        
        # Limitar el UFO a la pantalla
        if ufo_y < 0:
            ufo_y = 0
        if ufo_y > ALTO - ufo_alto:
            ufo_y = ALTO - ufo_alto
        
        # Generar y mover meteoritos
        generar_meteoritos()
        mover_meteoritos()
        
        # Verificar colisiones
        if verificar_colision(ufo_x, ufo_y):
            corriendo = False
        
        # Dibujar fondo y objetos
        pantalla.fill(NEGRO)
        dibujar_ufo(ufo_x, ufo_y)
        for meteorito in meteoritos:
            pygame.draw.rect(pantalla, ROJO, (meteorito[0], meteorito[1], meteorito_ancho, meteorito_alto))
        
        # Mostrar puntaje
        texto_puntaje = fuente_puntaje.render(f"Puntaje: {puntaje}", True, BLANCO)
        pantalla.blit(texto_puntaje, (10, 10))
        
        # Actualizar la pantalla
        pygame.display.update()
        
        # Controlar el FPS
        reloj.tick(30)

    # Pantalla de fin
    pantalla.fill(NEGRO)
    texto_final = fuente_titulo.render(f"Game Over! Puntaje: {puntaje}", True, ROJO)
    pantalla.blit(texto_final, (ANCHO // 4, ALTO // 2))
    pygame.display.update()
    pygame.time.wait(2000)

# Configuración de Streamlit
st.title("Space Alien")
st.write("""
Este es el juego **Space Alien** donde controlas un UFO para esquivar meteoritos.
El objetivo es sobrevivir el mayor tiempo posible y obtener la mayor cantidad de puntos.

**Controles**:
- Pulsa **espacio** para hacer que el UFO suba.
- Evita los meteoritos que caen del cielo.

Este juego fue creado por **Kevin Guio**.
""")
st.write("¡Haz clic en el botón de abajo para comenzar a jugar!")

if st.button("Iniciar Juego"):
    juego()

pygame.quit()
