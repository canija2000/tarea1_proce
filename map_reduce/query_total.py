import pyarrow.parquet as pq
import os
import re
import math
import itertools
from tqdm import tqdm
import json 

PATH = os.path.join("warehouse_data","fact_news")

json_dir = os.path.join("map_reduce","json_data")
txt_dir = os.path.join("map_reduce","txt_results")

if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir, exist_ok=True)


years  = [2023, 2024, 2025]
months = [f"{i:02d}" for i in range(1, 13)]

# Palabras comunes que no aportan significado relevante para el análisis (stop words)
STOP_WORDS = {
    'a', 'además', 'ahora', 'ahí', 'al', 'ante', 'así', 'año', 'años', 
    'bien', 'cada', 'como', 'con', 'contra', 'cuando', 'cómo', 'de', 
    'del', 'desde', 'después', 'dijo', 'donde', 'dos', 'durante', 'día', 
    'días', 'e', 'el', 'ella', 'en', 'entre', 'era', 'es', 'esa', 'esas', 
    'ese', 'eso', 'esos', 'esta', 'estas', 'este', 'esto', 'estos', 'está', 
    'están', 'fecha', 'fue', 'fueron', 'gran', 'ha', 'había', 'hace', 
    'hacer', 'han', 'hasta', 'hay', 'hecho', 'horas', 'hoy', 'la', 'las', 
    'le', 'les', 'lo', 'lo', 'los', 'luego', 'lugar', 'me', 'mi', 'mientras', 
    'mis', 'mismo', 'momento', 'muy', 'más', 'ni', 'no', 'nos', 'nueva', 
    'nuevo', 'o', 'otros', 'para', 'parte', 'pasado', 'pero', 'por', 
    'por que', 'porque', 'puede', 'que', 'quien', 'quienes', 'qué', 'respecto', 
    'se', 'sea', 'según', 'ser', 'será', 'señaló', 'si', 'sido', 'sin', 'sino', 
    'sobre', 'solo', 'son', 'su', 'su', 'sus', 'también', 'tanto', 'te', 'tener', 
    'the', 'tiene', 'todo', 'todos', 'tras', 'través', 'tres', 'tu', 'tus', 'un', 
    'una', 'uno', 'ver', 'vez', 'y', 'ya'
}

# longitud mínima de palabras a considerar
MIN_WORD_LENGTH = 3

# regex para extraer palabras (min 3 letras, ignorando mayúsculas y acentos)
TOKEN_RE = re.compile(
    rf"[a-záéíóúüñ]{{{MIN_WORD_LENGTH},}}", 
    re.IGNORECASE
)

# umbral mínimo de frecuencia global para considerar una palabra en análisis regional o por fuente
MIN_FREQ = 10


# Mapper que extrae palabras de un texto, normaliza a minúsculas y filtra stop words
def mapper(text: str):
    if not text:
        return
    for word in TOKEN_RE.findall(text):
        w = word.lower()
        if w not in STOP_WORDS:
            yield (w, 1)


# Pasada unica  acumula todo lo necesario para 2.1, 2.2, 2.3 y 2.4

def single_pass():
    monthly_counts = {}  # {(year, month): {word: count}}  → 2.1
    region_counts  = {}  # {region_id:     {word: count}}  → 2.2
    source_counts  = {}  # {source_id:     {word: count}}  → 2.3
    global_counts  = {}  # {word: count}                   → 2.2, 2.3
    daily_counts   = {}  # {"YYYY-MM-DD":  count}          → 2.4

    # cargamos dim_date una sola vez para resolver date_id → fecha
    dim_date = pq.read_table(os.path.join('warehouse_data','dim_date'))
    date_map = dict(zip(dim_date['date_id'].to_pylist(), dim_date['fecha'].to_pylist()))

    for year in tqdm(years, desc="Pasada única"):
        for month in tqdm(months, desc=f"  {year}", leave=False):
            path = os.path.join(PATH, f"year={year}", f"month={month}", "part-0.parquet")
            if not os.path.exists(path):
                continue

            table      = pq.read_table(path, columns=["title", "body", "region_id", "source_id", "date_id"])
            titles     = table.column("title").to_pylist()
            bodies     = table.column("body").to_pylist()
            region_ids = table.column("region_id").to_pylist()
            source_ids = table.column("source_id").to_pylist()
            date_ids   = table.column("date_id").to_pylist()

            month_key = (year, month)
            if month_key not in monthly_counts:
                monthly_counts[month_key] = {}

            # esto itera row by row 
            for title, body, region_id, source_id, date_id in zip(
                titles, bodies, region_ids, source_ids, date_ids
            ):
                # 2.4 conteo diario (solo incrementar el artículo, no las palabras)
                fecha = date_map.get(date_id)
                if fecha:
                    daily_counts[fecha] = daily_counts.get(fecha, 0) + 1

                for word, count in itertools.chain(mapper(title), mapper(body)):
                    # 2.1
                    mc = monthly_counts[month_key]
                    mc[word] = mc.get(word, 0) + count

                    # 2.2
                    if region_id not in region_counts:
                        region_counts[region_id] = {}
                    rc = region_counts[region_id]
                    rc[word] = rc.get(word, 0) + count

                    # 2.3
                    if source_id not in source_counts:
                        source_counts[source_id] = {}
                    sc = source_counts[source_id]
                    sc[word] = sc.get(word, 0) + count

                    # global
                    global_counts[word] = global_counts.get(word, 0) + count

    return monthly_counts, region_counts, source_counts, global_counts, daily_counts



# 2.1 top-K terminos mensuales

def query_top_k_terms_month(monthly_counts: dict, k: int = 20) -> None:
    print("Escribiendo top K términos mensuales...")
    
    with open(os.path.join(txt_dir, "query_top_k_terms_month.txt"), "w") as f:
        for (year, month), counts in sorted(monthly_counts.items()):
            top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]
            for term, count in top:
                f.write(f"{year}-{month}, {term}, {count}\n")



# 2.2 distribución de palabras por región

def query_dist_words_per_region(region_counts: dict, global_counts: dict) -> None:
    regiones = pq.read_table(os.path.join('warehouse_data','dim_region'))
    regiones_dict = dict(zip(
        regiones['region_id'].to_pylist(),
        regiones['region_name'].to_pylist()
    ))

    print("Escribiendo distribución de palabras por región...")
    with open(os.path.join(txt_dir, "query_dist_words_per_region.txt"), "w") as f:
        for region_id, words in region_counts.items():
            region_name = regiones_dict.get(region_id, f"id={region_id}")
            f.write(f"Región: {region_name}\n")
            top_words = sorted(
                ((w, c) for w, c in words.items() if global_counts.get(w, 0) >= MIN_FREQ),
                key=lambda x: x[1], reverse=True
            )[:20]
            for word, reg_count in top_words:
                glob_count = global_counts[word]
                ratio = reg_count / glob_count
                f.write(f"  {word}: regional={reg_count}, global={glob_count}, ratio={ratio:.4f}\n")
            f.write("\n")

    with open(os.path.join(txt_dir, "dist_global.csv"), "w") as f:
        f.write("palabra,f_g\n")
        for word, count in global_counts.items():
            f.write(f"{word},{count}\n")


# 2.3 divergencia de vocabulario por fuente (KL)

def query_kl_divergence_per_source(source_counts: dict, global_counts: dict) -> None:
    fuentes = pq.read_table(os.path.join('warehouse_data','dim_source'))
    fuentes_dict = dict(zip(
        fuentes['source_id'].to_pylist(),
        fuentes['source'].to_pylist()
    ))

    total_global = sum(global_counts.values())

    print("Calculando KL divergencia por fuente...")
    kl_results = []
    for source_id, words in source_counts.items():
        total_source = sum(words.values())
        if total_source == 0:
            continue

        kl = 0.0
        for word, src_count in words.items():
            glob_count = global_counts.get(word, 0)
            if glob_count < MIN_FREQ:
                continue
            p = src_count / total_source   # p_fuente(w)
            q = glob_count / total_global  # q_global(w)
            kl += p * math.log(p / q)

        kl_results.append((source_id, kl))

    kl_results.sort(key=lambda x: x[1], reverse=True)

    with open(os.path.join(txt_dir, "query_kl_per_source.txt"), "w") as f:
        f.write("fuente, kl_divergencia\n")
        for source_id, kl in kl_results:
            name = fuentes_dict.get(source_id, f"id={source_id}")
            f.write(f"{name}, {kl:.6f}\n")



# 2.4 detección de peaks de volumen diario

WINDOW      = 7    # dias de la ventana movil
PEAK_FACTOR = 1.5  # un día es peak si supera 1.5x el promedio móvil

def query_detect_peaks(daily_counts: dict) -> None:

    dates  = sorted(daily_counts.keys()) #fechas ordenadas
    counts = [daily_counts[d] for d in dates] # valores ordenados

    peaks = []
    moving_avgs = []
    for i, (date, count) in enumerate(zip(dates, counts)):
        window = counts[max(0, i - WINDOW):i]
        if len(window) < WINDOW:   # esperar ventana completa para evitar falsos peaks
            moving_avgs.append(None)
            continue
        moving_avg = sum(window) / len(window)
        moving_avgs.append(moving_avg)
        if count >= PEAK_FACTOR * moving_avg:
            peaks.append((date, count, moving_avg))

    peak_dates = {d for d, _, _ in peaks}

    with open(os.path.join(txt_dir, "query_peaks.txt"), "w") as f:
        f.write("fecha, articulos, promedio_movil_7d, factor\n")
        for date, count, avg in peaks:
            f.write(f"{date}, {count}, {avg:.1f}, {count/avg:.2f}x\n")

    # serie completa para visualización
    with open(os.path.join(txt_dir, "query_daily_counts.csv"), "w") as f:
        f.write("fecha,articulos,promedio_movil_7d,es_peak\n")
        for date, count, avg in zip(dates, counts, moving_avgs):
            avg_str = f"{avg:.1f}" if avg is not None else ""
            es_peak = 1 if date in peak_dates else 0
            f.write(f"{date},{count},{avg_str},{es_peak}\n")

    print(f"  {len(peaks)} peaks detectados → query_peaks.txt, query_daily_counts.csv")


def escribir_jsons(m_c, r_c, s_c, g_c, d_c):
    
    # monthly_counts usa tuplas como claves → serializar como "YYYY-MM"
    m_c_str = {f"{y}-{m}": v for (y, m), v in m_c.items()}
    with open(os.path.join(json_dir, "monthly_counts.json"), "w") as f:
        json.dump(m_c_str, f)
    with open(os.path.join(json_dir, "region_counts.json"), "w") as f:
        json.dump({str(k): v for k, v in r_c.items()}, f)
    with open(os.path.join(json_dir, "source_counts.json") , "w") as f:
        json.dump({str(k): v for k, v in s_c.items()}, f)
    with open(os.path.join(json_dir, "global_counts.json"), "w") as f:
        json.dump(g_c, f)
    with open(os.path.join(json_dir, "daily_counts.json"), "w") as f:
        json.dump(d_c, f)
    print("JSONs guardados.")


def leer_jsons():
    with open(os.path.join(json_dir, "monthly_counts.json")) as f:
        raw = json.load(f)
    # reconstruir claves como tuplas (year_int, "MM")
    monthly_counts = {(int(k[:4]), k[5:]): v for k, v in raw.items()}

    with open(os.path.join(json_dir, "region_counts.json")) as f:
        raw = json.load(f)
    region_counts = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in raw.items()}

    with open(os.path.join(json_dir, "source_counts.json")) as f:
        raw = json.load(f)
    source_counts = {int(k) if k.lstrip("-").isdigit() else k: v for k, v in raw.items()}

    with open(os.path.join(json_dir, "global_counts.json")) as f:
        global_counts = json.load(f)

    with open(os.path.join(json_dir, "daily_counts.json")) as f:
        daily_counts = json.load(f)

    return monthly_counts, region_counts, source_counts, global_counts, daily_counts
    

def main():
    jsons_exist = all(
        os.path.exists(p) for p in [
            os.path.join(json_dir, "monthly_counts.json"),
            os.path.join(json_dir, "region_counts.json"),
            os.path.join(json_dir, "source_counts.json"),
            os.path.join(json_dir, "global_counts.json"),
            os.path.join(json_dir, "daily_counts.json")
        ]
    )
    if jsons_exist:
        print(f"{'Cargando datos desde JSONs...':<50}", " | ")
        monthly_counts, region_counts, source_counts, global_counts, daily_counts = leer_jsons()
    else:
        print("Iniciando pasada única sobre los datos...")
        monthly_counts, region_counts, source_counts, global_counts, daily_counts = single_pass()
        escribir_jsons(monthly_counts, region_counts, source_counts, global_counts, daily_counts)

    print(f"{'Ejecutando query 2.1...':<50}", " | ")
    query_top_k_terms_month(monthly_counts)

    print(f"{'Ejecutando query 2.2...':<50}", " | ")
    query_dist_words_per_region(region_counts, global_counts)

    print(f"{'Ejecutando query 2.3...':<50}", " | ")
    query_kl_divergence_per_source(source_counts, global_counts)

    print(f"{'Ejecutando query 2.4...':<50}", " | ")
    query_detect_peaks(daily_counts)

# main

if __name__ == "__main__":
    main()
    
