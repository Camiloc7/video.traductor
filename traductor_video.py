import cv2
import easyocr
import time
import json
import hashlib
import os
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURACIÓN DEL PROYECTO ---
# 1. Archivo de video y salida
VIDEO_PATH = 'video_chino.mp4'  
SRT_OUTPUT = 'traduccion_final.srt'
# 2. VALORES DE REGIÓN DE INTERÉS (ROI)
Y_INICIO = 840
Y_FIN = 960
X_INICIO = 10
X_FIN = 1720
# 3. Idiomas a usar
LANG_ORIGEN = 'ch_sim'
LANG_DESTINO = 'es'
# 4. CONFIGURACIÓN DE SEGURIDAD Y PROGRESO
PROGRESS_FILE = 'progreso_traduccion.json'
OCR_CACHE_FILE = "ocr_cache.json"
TRANS_CACHE_FILE = "translation_cache.json"
# --- CACHÉS EN MEMORIA (Y DISCO) ---
translation_cache = {}
ocr_cache = {}

# 🔒 Candados (Locks) para Multithreading Seguro
ocr_lock = threading.Lock()
trans_lock = threading.Lock()

def cargar_caches_persistentes():
    global ocr_cache, translation_cache
    if os.path.exists(OCR_CACHE_FILE):
        try:
            with open(OCR_CACHE_FILE, 'r', encoding='utf-8') as f:
                ocr_cache = json.load(f)
        except: pass
    if os.path.exists(TRANS_CACHE_FILE):
        try:
            with open(TRANS_CACHE_FILE, 'r', encoding='utf-8') as f:
                translation_cache = json.load(f)
        except: pass

def guardar_caches_persistentes():
    with ocr_lock:
        with open(OCR_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(ocr_cache, f, ensure_ascii=False, indent=4)
    with trans_lock:
        with open(TRANS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=4)

cargar_caches_persistentes()

print("Inicializando EasyOCR (CPU)...")
# 🥇 MEJORA 4: EasyOCR optimizado CPU y batching (gpu=False y batch_size)
reader = easyocr.Reader([LANG_ORIGEN, 'en'], gpu=False)

# --- 🥇 NUEVO: TRADUCTOR MULTILENGUAJE NEURONAL (MarianMT Offline) ---
print("Cargando Modelos Neuronales MarianMT (Puede tardar la primera vez si descarga modelos)...")
# Deshabilitar warnings molestos de HuggingFace/Torch
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import MarianMTModel, MarianTokenizer

# Cargar modelos UNA sola vez (Quedan en RAM)
# 🥇 MEJORA 1: Helsinki-NLP ZH -> EN -> ES
tok_zh_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
mod_zh_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

tok_en_es = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
mod_en_es = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")

def traducir_texto_localmente(texto_chino):
    """Traducción Neuronal Local en 2 pasos (ZH -> EN -> ES) para mayor calidad"""
    try:
        # Paso 1: Chino a Inglés
        inputs = tok_zh_en([texto_chino], return_tensors="pt", padding=True)
        translated = mod_zh_en.generate(**inputs)
        texto_ing = tok_zh_en.decode(translated[0], skip_special_tokens=True)

        # Paso 2: Inglés a Español
        inputs = tok_en_es([texto_ing], return_tensors="pt", padding=True)
        translated = mod_en_es.generate(**inputs)
        texto_esp = tok_en_es.decode(translated[0], skip_special_tokens=True)

        return texto_esp
    except Exception as e:
        print(f"Error MarianMT: {e}")
        return texto_chino

def ms_to_srt_time(ms):
    ms = int(ms) 
    horas = int(ms / 3600000)
    ms %= 3600000
    minutos = int(ms / 60000)
    ms %= 60000
    segundos = int(ms / 1000)
    ms = int(ms % 1000)
    return f"{horas:02d}:{minutos:02d}:{segundos:02d},{ms:03d}"

def cargar_progreso():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("last_frame", 0), data.get("subtitles", [])
        except:
            pass
    return 0, []

def guardar_progreso(last_frame, subtitulos):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"last_frame": last_frame, "subtitles": subtitulos}, f, ensure_ascii=False, indent=4)

def procesar_subtitulo(roi_res, inicio_ms, fin_ms, hash_actual):
    """ Función Worker ejecutada en SEGUNDO PLANO """
    texto_ori = None
    
    # 1. OCR (con caché seguro)
    with ocr_lock:
        if hash_actual in ocr_cache:
            texto_ori = ocr_cache[hash_actual]
            
    if texto_ori is None:
        # Usando batch_size=4 para CPU
        resultados = reader.readtext(roi_res, detail=0, decoder='greedy', paragraph=True, canvas_size=640, batch_size=4)
        texto_ori = " ".join(resultados).strip()
        if texto_ori:
            with ocr_lock:
                ocr_cache[hash_actual] = texto_ori
    if not texto_ori or len(texto_ori) < 2:
        return None
    # 2. Traducción (con caché seguro y OFFLINE)
    texto_traducido = None
    with trans_lock:
        if texto_ori in translation_cache:
            texto_traducido = translation_cache[texto_ori]
            
    if texto_traducido is None:
        try:
            # MAGIA OFFLINE: Ya no se cuelga con internet ni bloqueos API
            texto_traducido = traducir_texto_localmente(texto_ori)
            with trans_lock:
                translation_cache[texto_ori] = texto_traducido
        except Exception as e:
            print(f"Error AI: {e}")
            texto_traducido = texto_ori 
    return {
        'inicio': inicio_ms,
        'fin': fin_ms,
        'texto_ori': texto_ori,
        'texto_final': texto_traducido
    }

def procesar_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    FRAME_SKIP = int(fps * 0.4) 
    if FRAME_SKIP < 1: FRAME_SKIP = 1    
    
    last_frame, subtitulos_completados = cargar_progreso()
    
    if last_frame > 0:
        print(f"Resumiendo desde el fotograma {last_frame} de {total_frames}...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)
    estado_texto = False
    inicio_sub = 0
    mejor_roi = None
    max_densidad = 0
    futures = []
    max_workers_disponibles = os.cpu_count() or 4
    WORKERS = max(2, max_workers_disponibles - 1)
    print(f"\n[⚡ MODO INDUSTRIAL: {WORKERS} Hilos + State Machine + Caché Persistente ⚡]")
    print(f"Analizando video ({fps}fps) saltando cada {FRAME_SKIP} frames en búsqueda rápida...")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        pbar = tqdm(total=total_frames, initial=last_frame, desc="Procesando Video")
        try:
            while True:
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                pbar.update(pos_frame - pbar.n)
                ret = cap.grab() 
                if not ret: break
                if pos_frame % FRAME_SKIP != 0:
                    continue
                ret, frame = cap.retrieve() 
                if not ret: continue
                
                tiempo_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                try:
                    roi = frame[Y_INICIO:Y_FIN, X_INICIO:X_FIN]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mask_blanco = cv2.inRange(roi_gray, 180, 255) 
                    edges = cv2.Canny(roi_gray, 30, 100) 
                    posible_texto = cv2.bitwise_and(edges, mask_blanco)
                    densidad = cv2.countNonZero(posible_texto) / posible_texto.size
                    if densidad >= 0.002:
                        roi_blur = cv2.GaussianBlur(roi_gray, (3,3), 0)
                        roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        alto, ancho = roi_thresh.shape
                        roi_res = cv2.resize(roi_thresh, (int(ancho * 1.5), int(alto * 1.5)))
                        if not estado_texto:
                            estado_texto = True
                            inicio_sub = tiempo_ms
                            mejor_roi = roi_res.copy()
                            max_densidad = densidad
                        else:
                            if densidad > max_densidad:
                                max_densidad = densidad
                                mejor_roi = roi_res.copy()
                    else:
                        if estado_texto:
                            estado_texto = False
                            fin_sub = tiempo_ms
                            
                            if mejor_roi is not None:
                                hash_limpio = hashlib.md5(mejor_roi.tobytes()).hexdigest()
                                ya_existe = False
                                with ocr_lock:
                                    if hash_limpio in ocr_cache:
                                        ya_existe = True
                                        texto_conocido = ocr_cache[hash_limpio]
                                        
                                if ya_existe and texto_conocido:
                                    texto_trad_conocido = texto_conocido
                                    with trans_lock:
                                        if texto_conocido in translation_cache:
                                            texto_trad_conocido = translation_cache[texto_conocido]
                                    
                                    if len(texto_conocido) >= 2:
                                        subtitulos_completados.append({
                                            'inicio': inicio_sub, 'fin': fin_sub,
                                            'texto_ori': texto_conocido, 'texto_final': texto_trad_conocido
                                        })
                                else:
                                    future = executor.submit(procesar_subtitulo, mejor_roi, inicio_sub, fin_sub, hash_limpio)
                                    futures.append(future)
                                
                            mejor_roi = None
                            max_densidad = 0
                            
                except Exception as e:
                    continue
                
                if len(futures) > 10:
                    completados = [f for f in futures if f.done()]
                    if completados:
                        for f in completados:
                            res = f.result()
                            if res: subtitulos_completados.append(res)
                            futures.remove(f)
                        if subtitulos_completados:
                            tqdm.write(f" -> {subtitulos_completados[-1]['texto_final']}")
                            
                        guardar_progreso(pos_frame, subtitulos_completados)

        except KeyboardInterrupt:
            print("\nProceso pausado por usuario. Recolectando hilos vivos...")
        
        finally:
            pbar.close()
            
            if estado_texto and mejor_roi is not None:
                hash_limpio = hashlib.md5(mejor_roi.tobytes()).hexdigest()
                future = executor.submit(procesar_subtitulo, mejor_roi, inicio_sub, cap.get(cv2.CAP_PROP_POS_MSEC), hash_limpio)
                futures.append(future)
            
            if futures:
                # 🥇 MEJORA: Barra de progreso para la fase final de sincronización
                pbar_sync = tqdm(total=len(futures), desc="Finalizando OCR/Traducción")
                for f in as_completed(futures):
                    res = f.result()
                    if res: subtitulos_completados.append(res)
                    pbar_sync.update(1)
                pbar_sync.close()
                    
            pos_final = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            subtitulos_completados.sort(key=lambda x: x['inicio'])
            
            # 🥈 MEJORA 2 & 3: FUSIÓN DE SUBTÍTULOS IDÉNTICOS
            subtitulos_fusionados = []
            for sub in subtitulos_completados:
                if not subtitulos_fusionados:
                    subtitulos_fusionados.append(sub)
                    continue
                
                ultimo = subtitulos_fusionados[-1]
                # Si el texto es igual y el subtitulo empieza muy poco después de que terminó el anterior (max 1 seg):
                if sub['texto_ori'] == ultimo['texto_ori'] and (sub['inicio'] - ultimo['fin']) < 1000:
                    # Extendemos el tiempo del subtítulo anterior en lugar de parpadear uno nuevo
                    ultimo['fin'] = sub['fin']
                else:
                    subtitulos_fusionados.append(sub)

            # Reasignar IDs perfectamente secuenciales a los fusionados
            for idx, sub in enumerate(subtitulos_fusionados):
                sub['id'] = idx + 1
                
            guardar_progreso(pos_final, subtitulos_fusionados)
            guardar_caches_persistentes()
            cap.release()
            cv2.destroyAllWindows()

    with open(SRT_OUTPUT, 'w', encoding='utf-8') as f:
        for sub in subtitulos_fusionados:
            f.write(f"{sub['id']}\n")
            f.write(f"{ms_to_srt_time(sub['inicio'])} --> {ms_to_srt_time(sub['fin'])}\n")
            f.write(f"{sub.get('texto_final', sub.get('texto_ori', ''))}\n\n")

    print(f"\n✅ ¡Proceso finalizado! Subtítulos totales luego de fusionar: {len(subtitulos_fusionados)}")
    if os.path.exists(PROGRESS_FILE) and pos_final >= total_frames - 10:
        os.remove(PROGRESS_FILE)
        print("Archivo de seguridad temporal (progreso_traduccion.json) eliminado limpiamente.")
        
if __name__ == '__main__':
    procesar_video()