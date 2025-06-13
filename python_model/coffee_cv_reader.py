import cv2
import numpy as np

# -------------- Configuración de umbrales y parámetros --------------
# Ajusta según la iluminación de tus fotos
CANNY_THRESH_1 = 50
CANNY_THRESH_2 = 150
MIN_CONTOUR_AREA = 100  # Ignora manchas pequeñísimas

# Diccionario de interpretaciones según patrones
interpretaciones = {
    'many_contours':    "Tu vida está llena de posibilidades: ¡aprovéchalas todas!",
    'few_contours':     "Momento de calma y reflexión. Disfruta la serenidad.",
    'circle_detected':  "Un ciclo se completa: prepárate para un nuevo comienzo.",
    'straight_line':    "Se avecina un viaje o un cambio de dirección importante.",
    'irregular_shape':  "La creatividad te guiará, confía en tu intuición."
}

# -------------- Funciones de análisis --------------

def detectar_contornos(img_gray):
    """Detecta contornos principales tras filtro y Canny."""
    blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    edges = cv2.Canny(blur, CANNY_THRESH_1, CANNY_THRESH_2)
    contornos, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Filtrar por área mínima
    contornos = [c for c in contornos if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    return contornos

def es_circulo(contorno):
    """Comprueba si un contorno es casi circular."""
    perimetro = cv2.arcLength(contorno, True)
    area = cv2.contourArea(contorno)
    if perimetro == 0: return False
    circularidad = 4 * np.pi * (area / (perimetro**2))
    return 0.7 < circularidad <= 1.2  # 1.0 es círculo perfecto

def es_linea(contorno):
    """Comprueba si un contorno es muy alargado (línea)."""
    x,y,w,h = cv2.boundingRect(contorno)
    ratio = max(w, h) / (min(w, h) + 1e-6)
    return ratio > 5  # muy alargado

def interpretar(contornos):
    """Aplica reglas simples para devolver un mensaje horóscopo."""
    n = len(contornos)
    # 1) Muchos vs pocos contornos
    if n > 12:
        return interpretaciones['many_contours']
    if n < 5:
        return interpretaciones['few_contours']
    # 2) Buscar si hay círculos o líneas
    for c in contornos:
        if es_circulo(c):
            return interpretaciones['circle_detected']
        if es_linea(c):
            return interpretaciones['straight_line']
    # 3) Forma irregular
    return interpretaciones['irregular_shape']

# -------------- Función principal --------------

def leer_taza(path_imagen):
    # 1. Cargar y pasar a gris
    img = cv2.imread(path_imagen)
    if img is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path_imagen}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 2. Detectar contornos
    contornos = detectar_contornos(gray)
    # 3. Generar lectura
    mensaje = interpretar(contornos)
    return {
        'num_contornos': len(contornos),
        'mensaje': mensaje
    }

# -------------- Uso desde línea de comandos --------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python coffee_cv_reader.py ruta/a/tu_foto.jpg")
        sys.exit(1)

    resultado = leer_taza(sys.argv[1])
    print(f"Contornos detectados: {resultado['num_contornos']}")
    print(f"⛤ Lectura: {resultado['mensaje']}")
