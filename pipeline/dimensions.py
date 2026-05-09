import os
import sys
from datetime import datetime

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .regions import REGION_ALIASES, REGION_IDS

_DAY_NAMES = {
    0: "Lunes", 1: "Martes", 2: "Miercoles", 3: "Jueves",
    4: "Viernes", 5: "Sabado", 6: "Domingo",
}

_ROMAN = {
    "Arica y Parinacota": "XV",  "Tarapaca": "I",      "Antofagasta": "II",
    "Atacama": "III",            "Coquimbo": "IV",      "Valparaiso": "V",
    "Metropolitana": "RM",       "OHiggins": "VI",      "Maule": "VII",
    "Nuble": "XVI",              "Biobio": "VIII",      "La Araucania": "IX",
    "Los Rios": "XIV",           "Los Lagos": "X",      "Aysen": "XI",
    "Magallanes": "XII",         "Desconocida": "—",
}
_ABREV = {
    "Arica y Parinacota": "AR",  "Tarapaca": "TA",     "Antofagasta": "AN",
    "Atacama": "AT",             "Coquimbo": "CO",      "Valparaiso": "VA",
    "Metropolitana": "RM",       "OHiggins": "OH",      "Maule": "MA",
    "Nuble": "NB",               "Biobio": "BI",        "La Araucania": "AR2",
    "Los Rios": "LR",            "Los Lagos": "LL",     "Aysen": "AY",
    "Magallanes": "MG",          "Desconocida": "??",
}


def scan_csv(
    input_csv: str, chunk_size: int
) -> tuple[dict[str, int], dict[str, int], int]:
    """
    Lee el CSV de entrada en chunks y extrae los valores únicos de fechas y fuentes,
    asignándoles un ID incremental. También cuenta el número total de filas procesadas.
    """
    unique_dates:   dict[str, int] = {}
    unique_sources: dict[str, int] = {}
    raw_row_count = 0
    reader = pd.read_csv(
        input_csv,
        usecols=["publish_date", "source"],
        chunksize=chunk_size,
        encoding="utf-8-sig",
        on_bad_lines="skip",
    )

    with tqdm(desc="Escaneando CSV", unit=" rows", unit_scale=True, file=sys.stdout, dynamic_ncols=True) as pbar:
        for chunk in reader:
            chunk_len = len(chunk)
            raw_row_count += chunk_len
            pbar.update(chunk_len)

            for d in chunk["publish_date"].dropna().unique():
                if d not in unique_dates:
                    unique_dates[d] = len(unique_dates) + 1

            for s in chunk["source"].dropna().str.lower().unique():
                if s not in unique_sources:
                    unique_sources[s] = len(unique_sources) + 1

    return unique_dates, unique_sources, raw_row_count


def build_dim_date(unique_dates: dict[str, int], warehouse: str) -> pd.DataFrame:
    """
    Construye la dimensión de fechas a partir del diccionario de fechas únicas, extrayendo
    información adicional como año, mes, día, día de la semana y semana del año. Luego guarda
    el DataFrame resultante en un archivo Parquet.
    """
    rows = []
    for date_str, date_id in tqdm(unique_dates.items(), desc="  dim_date", unit=" fecha"):
        try:
            d = datetime.strptime(date_str.strip(), "%Y-%m-%d")
            rows.append({
                "date_id":      date_id,
                "fecha":        date_str,
                "anio":         d.year,
                "mes":          d.month,
                "dia":          d.day,
                "dia_semana":   _DAY_NAMES[d.weekday()],
                "semana_anio":  d.isocalendar()[1],
            })
        except ValueError:
            pass

    dim = pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)
    out = f"{warehouse}/dim_date/dim_date.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dim.to_parquet(out, index=False)
    return dim


def build_dim_source(unique_sources: dict[str, int], warehouse: str) -> pd.DataFrame:
    """
    Construye la dimension de fuentes. Genera el parquet. 
    """
    rows = [
        {"source_id": sid, "source": src}
        for src, sid in tqdm(unique_sources.items(), desc="  dim_source", unit=" source")
    ]
    dim = pd.DataFrame(rows).sort_values("source_id").reset_index(drop=True)

    out = f"{warehouse}/dim_source/dim_source.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dim.to_parquet(out, index=False)
    return dim


def build_dim_region(warehouse: str) -> pd.DataFrame:
    """
    Construye la dimension de las regiones. Crea el parquet
    """
    rows = [
        {
            "region_id":   rid,
            "region_name": rname,
            "n_romano":    _ROMAN.get(rname, ""),
            "abreviacion": _ABREV.get(rname, ""),
        }
        for rname, rid in tqdm(REGION_IDS.items(), desc="  dim_region", unit=" region")
    ]
    dim = pd.DataFrame(rows)

    out = f"{warehouse}/dim_region/dim_region.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dim.to_parquet(out, index=False)
    return dim
