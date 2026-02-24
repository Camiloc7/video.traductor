import json
import os
import re

CACHE_FILE = 'translation_cache.json'

def limpiar_cache():
    if not os.path.exists(CACHE_FILE):
        print(f"No se encontró {CACHE_FILE}")
        return

    with open(CACHE_FILE, 'r', encoding='utf-8') as f:
        cache = json.load(f)

    original_count = len(cache)
    nuev_cache = {}
    eliminados = 0

    for k, v in cache.items():
        # 1. Detectar repeticiones exageradas de palabras (ej: "ella ella ella")
        # Buscamos si una palabra de más de 2 letras se repite más de 3 veces seguidas o dispersas
        palabras = v.lower().split()
        repeticion_critica = False
        
        counts = {}
        for p in palabras:
            if len(p) > 1:
                counts[p] = counts.get(p, 0) + 1
                if counts[p] > 4: # Si una palabra se repite más de 4 veces, es basura
                    repeticion_critica = True
                    break
        
        # 2. Detectar patrones de tartamudeo (ej: "yo-yo-yo" o "a-a-a")
        if re.search(r'(\b\w+\b)(?:\W+\1){3,}', v, re.IGNORECASE):
            repeticion_critica = True

        # 3. Detectar cadenas de caracteres raros o solo símbolos (asteriscos, etc)
        if len(v) > 20 and (v.count('*') > 10 or v.count('.') > 15 or v.count('-') > 15):
             repeticion_critica = True
             
        # 5. Detectar caracteres chinos en la TRADUCCIÓN (no debería haber!)
        if re.search(r'[\u4e00-\u9fff]', v):
             repeticion_critica = True
        
        # 6. Detectar si la traducción es igual al original (y tiene chino)
        if v == k and re.search(r'[\u4e00-\u9fff]', v):
            repeticion_critica = True

        # 7. Detectar tartamudeo excesivo (ej: "palabra palabra palabra")
        if len(v.split()) > 3:
            words = v.lower().split()
            if len(set(words)) == 1: # Todas las palabras son iguales
                repeticion_critica = True

        if not repeticion_critica:
            nuev_cache[k] = v
        else:
            eliminados += 1
            # print(f"Eliminando: {v[:50]}...")

    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(nuev_cache, f, ensure_ascii=False, indent=4)

    print(f"--- LIMPIEZA COMPLETADA ---")
    print(f"Entradas originales: {original_count}")
    print(f"Entradas eliminadas por basura: {eliminados}")
    print(f"Entradas limpias restantes: {len(nuev_cache)}")

if __name__ == "__main__":
    limpiar_cache()
