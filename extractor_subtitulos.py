"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         EXTRACTOR DE SUBTÍTULOS CHINO → ESPAÑOL  (v2.0 - CPU Optimizado)   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Instalación de dependencias:                                                ║
║                                                                              ║
║  pip install paddlepaddle paddleocr                                          ║
║  pip install transformers sentencepiece sacremoses                           ║
║  pip install opencv-python tqdm torch                                        ║
║                                                                              ║
║  Nota: En la PRIMERA ejecución se descargarán automáticamente:               ║
║    - Modelos PaddleOCR para chino (~150MB)                                   ║
║    - Modelo NLLB-200 distilled 600M (~2.4GB)                                 ║
║  Las siguientes ejecuciones usan caché local, sin re-descarga.               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import re
# IMPORTANTE: Torch debe importarse ANTES que PaddleOCR en Windows para evitar conflictos de DLLs (WinError 127)
import torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import json
import hashlib
import threading
import logging
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN DEL PROYECTO  (edita aquí tus parámetros)
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH      = 'video_chino.mp4'
SRT_OUTPUT      = 'traduccion_final.srt'

# Región de interés (ROI) — zona donde aparecen los subtítulos en el video
Y_INICIO = 840
Y_FIN    = 960
X_INICIO = 10
X_FIN    = 710   # Video es 720 wide

# Salto de frames: analiza 1 frame cada N milisegundos (50ms = captura de subtítulos muy rápidos)
FRAME_SKIP_MS   = 47

# Densidad mínima de píxeles de texto para considerar que hay subtítulo
DENSIDAD_MIN    = 0.001

# Máxima pausa entre subtítulos iguales para fusionarlos (en ms)
FUSION_GAP_MS   = 1500

# Archivos de estado y caché
PROGRESS_FILE   = 'progreso_traduccion.json'
OCR_CACHE_FILE  = 'ocr_cache.json'
TRANS_CACHE_FILE= 'translation_cache.json'

# Cuántas futures acumular antes de recolectarlas (evita consumo excesivo de RAM)
FUTURES_BUFFER  = 100

# ─────────────────────────────────────────────────────────────────────────────
#  SILENCIAR LOGS INNECESARIOS
# ─────────────────────────────────────────────────────────────────────────────
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")
os.environ["GLOG_minloglevel"] = "3"   # Silencia logs de PaddlePaddle en C++
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True" 
os.environ["FLAGS_enable_pir_api"] = "0"    # Desactiva el nuevo motor PIR (más estable en CPU AMD)
os.environ["FLAGS_use_mkldnn"] = "0"        # Desactiva MKLDNN a nivel de sistema

# ─────────────────────────────────────────────────────────────────────────────
#  CACHÉS EN MEMORIA + DISCO
# ─────────────────────────────────────────────────────────────────────────────
translation_cache: dict = {}
ocr_cache: dict         = {}
ocr_lock   = threading.Lock()
trans_lock = threading.Lock()

def cargar_caches():
    global ocr_cache, translation_cache
    for path, ref in [(OCR_CACHE_FILE, 'ocr'), (TRANS_CACHE_FILE, 'trans')]:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if ref == 'ocr':
                    ocr_cache = data
                else:
                    translation_cache = data
                print(f"  ✔ Caché cargada: {path} ({len(data)} entradas)")
            except Exception as e:
                print(f"  ⚠ No se pudo leer {path}: {e}")

def normalizar_texto_cache(texto: str) -> str:
    """Genera una clave única y limpia para el caché de traducción."""
    if not texto: return ""
    texto = texto.strip()
    # quitar puntuación final común (china y latina)
    texto = re.sub(r'[。.!！?？,，]+$', '', texto)
    # quitar ABSOLUTAMENTE todo espacio en blanco para que "A B" y "AB" sean la misma clave
    texto = re.sub(r'\s+', '', texto)
    return texto.lower()

def guardar_caches():
    with ocr_lock:
        with open(OCR_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(ocr_cache, f, ensure_ascii=False, indent=2)
    with trans_lock:
        with open(TRANS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(translation_cache, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  INICIALIZACIÓN DE MODELOS
# ─────────────────────────────────────────────────────────────────────────────
cargar_caches()

# ── 1. PaddleOCR (reemplaza EasyOCR — mucho más rápido en CPU) ───────────────
print("\n[1/2] Inicializando PaddleOCR (CPU, chino simplificado)...")
from paddleocr import PaddleOCR

ocr_engine = PaddleOCR(
    use_angle_cls=False,   # Los subtítulos siempre están horizontales
    lang='ch',
    use_gpu=False,
    show_log=False,
)
print("  ✔ PaddleOCR listo.")

# ── 2. NLLB-200 Distilled (traducción con idioma pivote: ZH → EN → ES) ─────
print("\n[2/2] Cargando NLLB-200 distilled-600M (primera vez descarga ~2.4GB)...")
# Los modelos ya fueron importados arriba

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
nllb_model     = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)

# ── Quantización dinámica: reduce RAM ~40% y acelera inferencia en CPU ────────
nllb_model = torch.quantization.quantize_dynamic(
    nllb_model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
nllb_model.eval()
nllb_model.config.use_cache = True
print("  ✔ NLLB-200 listo (con quantización int8 y cache activado).")

# Candado para traducción (el modelo no es thread-safe sin esto)
nllb_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def limpiar_texto(t: str) -> str:
    """Elimina ruidos comunes del OCR y caracteres que ensucian la traducción."""
    if not t: return ""
    # Eliminar símbolos raros, guiones bajos extraños, etc.
    t = re.sub(r'[_|\\[\\]{}#]', '', t)
    # Eliminar espacios múltiples
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def traducir_batch(textos: list, batch_size: int = 8) -> list:
    """
    Traduce una lista de textos usando batching de NLLB para mayor velocidad.
    """
    if not textos:
        return []

    resultados = [""] * len(textos)
    pendientes_idx = []
    pendientes_txt = []

    # 1. Limpieza y revisión de caché
    for i, t in enumerate(textos):
        t_limpio = limpiar_texto(t)
        if not t_limpio:
            continue
        
        t_key = normalizar_texto_cache(t_limpio)
        with trans_lock:
            cached = translation_cache.get(t_key)
        
        if cached:
            resultados[i] = cached
        else:
            pendientes_idx.append(i)
            # Guardamos el texto "limpio" (con espacios) para traducir, 
            # pero usaremos la t_key para guardarlo después
            pendientes_txt.append(t_limpio)

    if not pendientes_txt:
        return resultados

    # --- TRUCO 1: Ordenar por longitud para reducir padding ---
    indices_ordenados = sorted(
        range(len(pendientes_txt)),
        key=lambda i: len(pendientes_txt[i])
    )
    pendientes_txt = [pendientes_txt[i] for i in indices_ordenados]
    pendientes_idx = [pendientes_idx[i] for i in indices_ordenados]

    # 2. Traducción en lotes
    pbar = tqdm(total=len(pendientes_txt), desc="Traduciendo (NLLB Batch)", unit="txt", leave=False)
    for i in range(0, len(pendientes_txt), batch_size):
        lote = pendientes_txt[i : i + batch_size]
        lote_idx = pendientes_idx[i : i + batch_size]

        try:
            with nllb_lock:
                with torch.inference_mode():
                    inputs = nllb_tokenizer(
                        lote,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256,
                        src_lang="zho_Hans"
                    )

                    # --- TRUCO 4: max_length inteligente ---
                    out_len = min(256, int(inputs.input_ids.shape[1] * 2.5))

                    # --- TRUCO 2: Greedy search (num_beams=1) ---
                    tokens = nllb_model.generate(
                        **inputs,
                        forced_bos_token_id=nllb_tokenizer.lang_code_to_id["spa_Latn"],
                        max_length=out_len,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        repetition_penalty=1.1
                    )

            traducciones = nllb_tokenizer.batch_decode(tokens, skip_special_tokens=True)

                for j, traducido in enumerate(traducciones):
                traducido = traducido.strip()
                real_idx = lote_idx[j]
                resultados[real_idx] = traducido
                
                # Guardar en caché usando la clave normalizada
                t_key = normalizar_texto_cache(pendientes_txt[i + j])
                with trans_lock:
                    translation_cache[t_key] = traducido

            pbar.update(len(lote))

        except Exception as e:
            print(f"  ⚠ Error en batch translation: {e}")
    
    pbar.close()

    return resultados

def traducir_zh_es(texto: str) -> str:
    """Wrapper para mantener compatibilidad con llamadas individuales."""
    res = traducir_batch([texto], batch_size=1)
    return res[0] if res else ""

def preprocesar_roi(roi_bgr: "np.ndarray") -> "np.ndarray":
    """
    Pipeline de preprocesamiento mejorado para PaddleOCR.
    """
    import cv2
    import numpy as np

    # 1. Escalar primero (mejor para preservar bordes en OCR)
    h, w = roi_bgr.shape[:2]
    roi_large = cv2.resize(roi_bgr, (int(w * 2), int(h * 2)), interpolation=cv2.INTER_CUBIC)

    # 2. Reducción de ruido preservando bordes
    denoised = cv2.bilateralFilter(roi_large, 9, 75, 75)

    # 3. Grayscale
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)

    # 4. Sharpening (Máscara de enfoque)
    # Aumenta el contraste de los bordes del texto
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

    return sharpened


def densidad_texto(roi_bgr: "np.ndarray") -> float:
    """Calcula la densidad de píxeles de texto en la ROI (0.0 - 1.0)."""
    import cv2
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    mask  = cv2.inRange(gray, 180, 255)
    edges = cv2.Canny(gray, 30, 100)
    posible_texto = cv2.bitwise_and(edges, mask)
    return cv2.countNonZero(posible_texto) / posible_texto.size


def ms_to_srt_time(ms: float) -> str:
    ms = int(ms)
    h  = ms // 3_600_000; ms %= 3_600_000
    m  = ms //    60_000; ms %=    60_000
    s  = ms //     1_000; ms %=     1_000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def cargar_progreso():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                d = json.load(f)
            return d.get("last_frame", 0), d.get("subtitles", [])
        except Exception:
            pass
    return 0, []


def guardar_progreso(last_frame: int, subs: list):
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:
        json.dump({"last_frame": last_frame, "subtitles": subs}, f,
                  ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  WORKER: OCR + TRADUCCIÓN (ejecutado en hilo secundario)
# ─────────────────────────────────────────────────────────────────────────────

def procesar_subtitulo(roi_preprocesada, inicio_ms: float, fin_ms: float, img_hash: str):
    """
    Worker que corre en el ThreadPoolExecutor.
    1. OCR con PaddleOCR (con caché por hash de imagen)
    2. Traducción con NLLB-200 (con caché por texto)
    """
    # ── OCR ──────────────────────────────────────────────────────────────────
    texto_ori = None
    with ocr_lock:
        texto_ori = ocr_cache.get(img_hash)

    if texto_ori is None:
        try:
            resultado = ocr_engine.ocr(roi_preprocesada)
            lineas = []
            if resultado and resultado[0]:
                for linea in resultado[0]:
                    # linea[1] = (texto, confianza)
                    if linea[1][1] >= 0.5:          # Descartar detecciones de baja confianza
                        lineas.append(linea[1][0])
            texto_ori = " ".join(lineas).strip()
        except Exception as e:
            print(f"\n  ⚠ Error OCR: {e}")
            texto_ori = ""

        if texto_ori:
            with ocr_lock:
                ocr_cache[img_hash] = texto_ori

    if not texto_ori or len(texto_ori) < 2:
        return None

    # ── Traducción ────────────────────────────────────────────────────────────
    # La traducción individual se eliminó para usar la traducción por bloques al final
    return {
        'inicio':     inicio_ms,
        'fin':        fin_ms,
        'texto_ori':  texto_ori,
        'texto_final': "",
    }

# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def traducir_por_bloques(
    subs,
    max_gap_ms=4000,
    max_chars=300,
    max_items=16
):
    """
    Traduce subtítulos agrupándolos en bloques grandes con contexto.

    max_gap_ms  = máximo espacio temporal entre subtítulos del mismo bloque
    max_chars   = máximo caracteres concatenados por bloque
    max_items   = máximo subtítulos por bloque
    """

    if not subs:
        return []

    bloques = []
    bloque_actual = []
    chars_actual = 0

    for sub in subs:

        texto = sub.get("texto_ori", "").strip()

        if not texto:
            continue

        if not bloque_actual:
            bloque_actual.append(sub)
            chars_actual = len(texto)
            continue

        ultimo = bloque_actual[-1]

        gap = sub["inicio"] - ultimo["fin"]

        # decidir si agregar al bloque actual
        if (
            gap <= max_gap_ms
            and chars_actual + len(texto) <= max_chars
            and len(bloque_actual) < max_items
        ):
            bloque_actual.append(sub)
            chars_actual += len(texto)

        else:
            bloques.append(bloque_actual)
            bloque_actual = [sub]
            chars_actual = len(texto)

    if bloque_actual:
        bloques.append(bloque_actual)

    resultado_final = []

    # --- NUEVA LÓGICA DE BATCHING ---
    textos_concat = []
    for bloque in bloques:
        txt = " ".join(limpiar_texto(b["texto_ori"]) for b in bloque).strip()
        textos_concat.append(txt)

    print(f"\n📦 Traduciendo {len(bloques)} bloques en modo Batch (Turbo)...")
    # Aumentamos a 8 para aprovechar los trucos de optimización
    traducciones_lote = traducir_batch(textos_concat, batch_size=8)

    for bloque, traduccion in zip(bloques, traducciones_lote):
        if not traduccion:
            resultado_final.extend(bloque)
            continue

        # Dividir la traducción resultante entre los subtítulos del bloque
        palabras = traduccion.split()
        total_palabras = len(palabras)
        total_subs = len(bloque)

        if total_subs == 1:
            bloque[0]["texto_final"] = traduccion
            resultado_final.append(bloque[0])
            continue

        # Distribución proporcional de palabras
        ratio = total_palabras / total_subs
        idx = 0
        for i, sub in enumerate(bloque):
            if i == total_subs - 1:
                segmento = palabras[idx:]
            else:
                n = max(1, round(ratio))
                segmento = palabras[idx:idx+n]
                idx += n
            sub["texto_final"] = " ".join(segmento)
            resultado_final.append(sub)

    return resultado_final




def procesar_video():
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"\n❌ Error: No se pudo abrir el video en '{VIDEO_PATH}'")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duracion_seg = total_frames / fps

    # Convertir salto de ms a número de frames
    FRAME_SKIP = max(1, int(fps * FRAME_SKIP_MS / 1000))

    print(f"\n📹 Video: {VIDEO_PATH}")
    print(f"   FPS: {fps:.2f} | Frames totales: {total_frames} | Duración: {duracion_seg/60:.1f} min")
    print(f"   Analizando 1 frame cada {FRAME_SKIP_MS}ms ({FRAME_SKIP} frames)")
    print(f"   ROI: Y[{Y_INICIO}:{Y_FIN}] X[{X_INICIO}:{X_FIN}]")

    # ── Cargar checkpoint si existe ───────────────────────────────────────────
    last_frame, subtitulos_completados = cargar_progreso()
    if last_frame > 0:
        print(f"\n🔄 Reanudando desde el frame {last_frame} ({last_frame/fps/60:.1f} min)...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_frame)

    # ── Estado de la máquina de estados ──────────────────────────────────────
    en_subtitulo   = False
    inicio_sub     = 0.0
    mejor_roi      = None
    max_densidad   = 0.0

    futures        = []
    WORKERS        = max(2, (os.cpu_count() or 4) - 1)
    print(f"\n⚡ Usando {WORKERS} hilos de trabajo\n")

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        pbar = tqdm(total=total_frames, initial=last_frame,
                    desc="Procesando", unit="fr",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        try:
            while True:
                pos_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                pbar.update(pos_frame - pbar.n)

                # grab() es rápido (no decodifica) — solo decodifica frames que necesitamos
                ret = cap.grab()
                if not ret:
                    break

                if pos_frame % FRAME_SKIP != 0:
                    continue

                ret, frame = cap.retrieve()
                if not ret:
                    continue

                tiempo_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

                try:
                    roi = frame[Y_INICIO:Y_FIN, X_INICIO:X_FIN]
                    densidad = densidad_texto(roi)

                    if densidad >= DENSIDAD_MIN:
                        # ── Hay texto en pantalla ─────────────────────────────
                        roi_proc = preprocesar_roi(roi)

                        if not en_subtitulo:
                            en_subtitulo = True
                            inicio_sub   = tiempo_ms
                            mejor_roi    = roi_proc.copy()
                            max_densidad = densidad
                        else:
                            # Guardar el frame con más densidad de texto
                            if densidad > max_densidad:
                                max_densidad = densidad
                                mejor_roi    = roi_proc.copy()

                    else:
                        # ── Fin de subtítulo ──────────────────────────────────
                        if en_subtitulo:
                            en_subtitulo = False
                            fin_sub      = tiempo_ms

                            if mejor_roi is not None:
                                img_hash = hashlib.md5(mejor_roi.tobytes()).hexdigest()

                                # Verificar caché antes de lanzar hilo
                                texto_conocido = None
                                with ocr_lock:
                                    texto_conocido = ocr_cache.get(img_hash)

                                if texto_conocido and len(texto_conocido) >= 2:
                                    subtitulos_completados.append({
                                        'inicio':     inicio_sub,
                                        'fin':        fin_sub,
                                        'texto_ori':  texto_conocido,
                                        'texto_final': "",
                                    })
                                else:
                                    future = executor.submit(
                                        procesar_subtitulo,
                                        mejor_roi.copy(), inicio_sub, fin_sub, img_hash
                                    )
                                    futures.append(future)

                            mejor_roi    = None
                            max_densidad = 0.0

                except Exception:
                    continue

                # ── Recolección periódica de futures completadas ───────────────
                if len(futures) >= FUTURES_BUFFER:
                    completadas = [f for f in futures if f.done()]
                    for f in completadas:
                        try:
                            res = f.result()
                            if res:
                                subtitulos_completados.append(res)
                                tqdm.write(f"  💬 {res['texto_ori'][:40]} → {res['texto_final'][:60]}")
                        except Exception as e:
                            tqdm.write(f"  ⚠ Error en worker: {e}")
                        futures.remove(f)

                    if completadas:
                        guardar_progreso(pos_frame, subtitulos_completados)
                        guardar_caches()

        except KeyboardInterrupt:
            tqdm.write("\n⏸ Proceso pausado por el usuario. Guardando progreso...")

        finally:
            pbar.close()

            # Si el video terminó con un subtítulo activo, procesarlo
            if en_subtitulo and mejor_roi is not None:
                img_hash = hashlib.md5(mejor_roi.tobytes()).hexdigest()
                futures.append(executor.submit(
                    procesar_subtitulo,
                    mejor_roi, inicio_sub, cap.get(cv2.CAP_PROP_POS_MSEC), img_hash
                ))

            # Esperar y recolectar TODAS las futures pendientes
            if futures:
                print(f"\n⏳ Finalizando {len(futures)} tareas pendientes...")
                pbar_sync = tqdm(total=len(futures), desc="Finalizando OCR/Traducción", unit="sub")
                for f in as_completed(futures):
                    try:
                        res = f.result()
                        if res:
                            subtitulos_completados.append(res)
                    except Exception as e:
                        print(f"  ⚠ Error en worker final: {e}")
                    pbar_sync.update(1)
                pbar_sync.close()

            pos_final = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.release()
            cv2.destroyAllWindows()

    # ── Post-proceso: ordenar + fusionar subtítulos duplicados ────────────────
    subtitulos_completados.sort(key=lambda x: x['inicio'])

    subtitulos_fusionados = []
    for sub in subtitulos_completados:
        if not subtitulos_fusionados:
            subtitulos_fusionados.append(dict(sub))
            continue
        ultimo = subtitulos_fusionados[-1]
        # Fusionar si mismo texto y brecha pequeña
        mismo_texto = sub['texto_ori'] == ultimo['texto_ori']
        brecha_ok   = (sub['inicio'] - ultimo['fin']) < FUSION_GAP_MS
        if mismo_texto and brecha_ok:
            ultimo['fin'] = sub['fin']   # Extender duración
        else:
            subtitulos_fusionados.append(dict(sub))

    print("\n[2.5/3] Traduciendo por bloques con contexto...")
    subtitulos_fusionados = traducir_por_bloques(subtitulos_fusionados)

    # ── Post-proceso 2: Reparar traducciones vacías por contexto ──────────────
    print("\n[3/3] Reparando traducciones en blanco mediante contexto...")
    reparados = []
    i = 0
    pbar_rep = tqdm(total=len(subtitulos_fusionados), desc="Reparando", unit="sub")
    
    while i < len(subtitulos_fusionados):
        sub = subtitulos_fusionados[i]
        txt_trad = sub.get('texto_final') or ""
        txt_trad = txt_trad.strip()
        
        if not txt_trad:
            fusionado = False
            # 1. Intentar con el ANTERIOR
            if reparados:
                ultimo = reparados[-1]
                if (sub['inicio'] - ultimo['fin']) < (FUSION_GAP_MS * 1.5):
                    nuevo_ori = (ultimo.get('texto_ori', '') + " " + sub.get('texto_ori', '')).strip()
                    n_key = normalizar_texto_cache(nuevo_ori)
                    
                    with trans_lock:
                        nueva_trad = translation_cache.get(n_key)
                    
                    if not nueva_trad:
                        nueva_trad = traducir_zh_es(nuevo_ori)
                        # traducir_zh_es ya guarda en caché internamente
                    
                    if nueva_trad:
                        ultimo['texto_ori'] = nuevo_ori
                        ultimo['texto_final'] = nueva_trad
                        ultimo['fin'] = max(ultimo['fin'], sub['fin'])
                        fusionado = True
                        tqdm.write(f"  🔧 Reparado (<- anterior): {nueva_trad}")
            
            # 2. Intentar con el SIGUIENTE
            if not fusionado and i + 1 < len(subtitulos_fusionados):
                siguiente = subtitulos_fusionados[i+1]
                if (siguiente['inicio'] - sub['fin']) < (FUSION_GAP_MS * 1.5):
                    nuevo_ori = (sub.get('texto_ori', '') + " " + siguiente.get('texto_ori', '')).strip()
                    n_key = normalizar_texto_cache(nuevo_ori)
                    
                    with trans_lock:
                        nueva_trad = translation_cache.get(n_key)
                        
                    if not nueva_trad:
                        nueva_trad = traducir_zh_es(nuevo_ori)
                            
                    if nueva_trad:
                        siguiente['texto_ori'] = nuevo_ori
                        siguiente['texto_final'] = nueva_trad
                        siguiente['inicio'] = min(siguiente['inicio'], sub['inicio'])
                        fusionado = True
                        tqdm.write(f"  🔧 Reparado (-> siguiente): {nueva_trad}")
            
            if not fusionado:
                reparados.append(sub)
        else:
            reparados.append(sub)
            
        i += 1
        pbar_rep.update(1)
        
    pbar_rep.close()
    subtitulos_fusionados = reparados

    # Asignar IDs secuenciales finales
    for idx, sub in enumerate(subtitulos_fusionados, start=1):
        sub['id'] = idx

    # ── Guardar estado y cachés finales ───────────────────────────────────────
    guardar_progreso(pos_final, subtitulos_fusionados)
    guardar_caches()

    # ── Escribir archivo SRT (solo subtítulos con texto válido) ───────────────
    with open(SRT_OUTPUT, 'w', encoding='utf-8') as f:
        real_idx = 1
        for sub in subtitulos_fusionados:
            txt = sub.get('texto_final', '').strip()
            # Si no hay traducción, intentar usar el original si no es basura
            if not txt:
                cont_zh = sub.get('texto_ori', '')
                if len(cont_zh) > 1 and not re.search(r'^[^\w\u4e00-\u9fff]+$', cont_zh):
                    txt = f"[{cont_zh}]" # Entre corchetes si no se pudo traducir
            
            if txt:
                f.write(f"{real_idx}\n")
                f.write(f"{ms_to_srt_time(sub['inicio'])} --> {ms_to_srt_time(sub['fin'])}\n")
                f.write(f"{txt}\n\n")
                real_idx += 1

    print(f"\n{'═'*60}")
    print(f"✅ ¡Proceso completado!")
    print(f"   Subtítulos encontrados: {len(subtitulos_fusionados)}")
    print(f"   Archivo generado: {SRT_OUTPUT}")
    print(f"{'═'*60}\n")

    # Limpiar checkpoint si se procesó todo el video
    if pos_final >= total_frames - FRAME_SKIP * 2:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("🧹 Checkpoint temporal eliminado (proceso completo).")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    procesar_video()