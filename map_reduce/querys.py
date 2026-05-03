import pyarrow.parquet as pq
import os
import re
from tqdm import tqdm

## 2.1 TOP K terminos mensuales. [20 terminos más frecuentes x año mes]
PATH = os.path.join("..","warehouse_data","fact_news")

years = [2023,2024,2025]
months = [f"{i:02d}" for i in range(1,13)] # 01, 02, etc

STOP_WORDS = {
    'de','la','su','como','a','por','en','que','con','para',
    'se','del','al','lo','es','y','un','una','las','los',
    'el','le','les','no','si','ya','o','e','ni','pero','más',
    'este','esta','estos','estas','ese','esa','esos','esas',
    'ser','ha','han','hay','fue','son','era','está','están',
    'lo','me','te','nos','mi','tu','su','sus','mis','tus',
    'sobre','entre','sin','hasta','desde','ante','tras','donde',
    'cuando','también','por que','todo','muy','años','además','parte',
    'durante','tiene','según','porque','esto','quien','solo','todos',
    'luego','tanto'
}

# regex para descartar numeros y puntuaciones no relevantes al caso. 2023,
# exige al menos 4 letras
TOKEN_RE = re.compile(r"[a-záéíóúüñ]{4,}", re.IGNORECASE)


# mapeamos palabras [este metodo se asemja al del libro Mining od datasets]
def mapper(text: str):
    if not text:
        return
    for word in TOKEN_RE.findall(text):
        w = word.lower()         
        if w not in STOP_WORDS:
            yield (w, 1)


# funcion principal
def top_k_terms_month(year: int, month: str, k: int = 20) -> list[tuple[str, int]]:
    path = os.path.join(PATH, f"year={year}", f"month={month}", "part-0.parquet")
    if not os.path.exists(path):
        return []

    # leemos solo lo que nos interesa. 
    table = pq.read_table(path, columns=["title", "body"])

    # map + reduce
    counts = {}
    for col in ["title", "body"]:
        for text in table.column(col).to_pylist():
            for word, count in mapper(text):
                counts[word] = counts.get(word, 0) + count

    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:k]

## salida quert 2.1 

def query_top_k_terms_month() -> None:
   print("Generando archivo con top K términos mensuales...")
   with open("query_top_k_terms_month.txt", "w") as f:
    for year in tqdm(years, desc="Años top términos"):
        for month in tqdm(months, desc=f"Meses {year}", leave=False):
            top_terms = top_k_terms_month(year, month)
            f.write(f"Top términos para {year}-{month}:")
            for term, count in top_terms:
                f.write(f"  {term}: {count}")
            f.write("\n")


########### 2.2 palabras x region [distribucion]

MIN_FREQ = 10  # filtro mínimo de frecuencia global

def dist_words_per_region():
    regiones = pq.read_table(os.path.join('..','warehouse_data','dim_region'))
    # .to_pylist() para obtener tipos python
    regiones_dict = dict(zip(regiones['region_id'].to_pylist(), regiones['region_name'].to_pylist()))

    regiones_words = {r: {} for r in regiones_dict.keys()}
    global_words = {}

    for year in tqdm(years, desc="Años región"):
        for month in tqdm(months, desc=f"Meses {year}", leave=False):
            path = os.path.join(PATH, f"year={year}", f"month={month}", "part-0.parquet")
            if not os.path.exists(path):
                continue

            table = pq.read_table(path, columns=["title", "body", "region_id"])
            # extraer columnas completas de una vez (acceso columnar, más eficiente)
            titles    = table.column("title").to_pylist()
            bodies    = table.column("body").to_pylist()
            region_ids = table.column("region_id").to_pylist()

            for title, body, region_id in zip(titles, bodies, region_ids):
                # reduce: atribuir a la región de este artículo y al global
                for word, count in (*mapper(title), *mapper(body)):
                    reg_dict = regiones_words.get(region_id)
                    if reg_dict is not None:
                        reg_dict[word] = reg_dict.get(word, 0) + count
                    global_words[word] = global_words.get(word, 0) + count

    # filtrar palabras con frecuencia global < MIN_FREQ
    regiones_words_filtered = {
        r: {word: count for word, count in words.items() if global_words.get(word, 0) >= MIN_FREQ}
        for r, words in regiones_words.items()
    }

    print("Escribiendo resultados de distribución de palabras por región...")
    with open("query_dist_words_per_region.txt", "w") as f:
        for region_id, words in regiones_words_filtered.items():
            region_name = regiones_dict[region_id]
            f.write(f"Región: {region_name} (id={region_id})\n")
            top_words = sorted(words.items(), key=lambda x: x[1], reverse=True)[:20]
            for word, reg_count in top_words:
                glob_count = global_words[word]
                ratio = reg_count / glob_count
                f.write(f"  {word}: regional={reg_count}, global={glob_count}, ratio={ratio:.4f}\n")
            f.write("\n")
    with open('dist_global.csv', 'w') as f: 
        f.write("palabra,f_g\n")
        for word, count in global_words.items():
            f.write(f"{word},{count}\n")
    

if __name__ == "__main__":
    print("Ejecutando query 2.1: Top K términos mensuales...")
    query_top_k_terms_month()
    print("Ejecutando query 2.2: Distribución de palabras por región...")
    dist_words_per_region()           