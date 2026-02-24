import cv2
import os

# --- CONFIGURACIÓN DE PRUEBA ---
VIDEO_PATH = 'video_chino.mp4'  # ¡Cambia esto al nombre exacto de tu archivo!

# ** COORDENADAS A AJUSTAR **
# (Y_INICIO, Y_FIN, X_INICIO, X_FIN)
Y_INICIO = 840
Y_FIN = 960
X_INICIO = 10
X_FIN = 1720

# >>> NUEVA CONFIGURACIÓN DE TIEMPO <<<
# Define el punto de tiempo al que quieres saltar:
MINUTOS_A_SALTAR = 6 # Por ejemplo, salta a los 5 minutos
SEGUNDOS_ADICIONALES = 33 # Y 30 segundos más (5:30)

# --- FUNCIÓN DE PRUEBA ---

def probar_roi():
    # 1. Abrir el video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el video en {VIDEO_PATH}. Revisa el nombre.")
        return

    # 2. CÁLCULO Y SALTO DE TIEMPO
    tiempo_total_ms = (MINUTOS_A_SALTAR * 60 + SEGUNDOS_ADICIONALES) * 1000
    
    # Mover el cursor del video a la posición deseada (en milisegundos)
    cap.set(cv2.CAP_PROP_POS_MSEC, tiempo_total_ms)
    print(f"Saltando a la posición: {MINUTOS_A_SALTAR} min {SEGUNDOS_ADICIONALES} seg ({tiempo_total_ms} ms)...")

    # 3. Leer el fotograma en esa posición
    ret, frame = cap.read()
    if not ret:
        print("ERROR: No se pudo leer el fotograma después de saltar. Puede que el tiempo sea posterior al final del video.")
        cap.release()
        return

    # 4. Aplicar el recorte de la ROI
    try:
        # Formato: [Y_INICIO:Y_FIN, X_INICIO:X_FIN]
        roi_recortada = frame[Y_INICIO:Y_FIN, X_INICIO:X_FIN]
    except IndexError:
        print("\nERROR: ¡Las coordenadas están fuera de los límites del video!")
        print(f"Revisa tu video. Dimensión del fotograma: {frame.shape}")
        return

    # 5. Guardar el resultado en un archivo para verlo
    filename = "prueba_roi_resultado.jpg"
    cv2.imwrite(filename, roi_recortada)
    print(f"ROI guardada en {filename}. Revisa tus archivos.")
    
    cap.release()

if __name__ == '__main__':
    probar_roi()