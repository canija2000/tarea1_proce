import os
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import FACT_SCHEMA
from .encoding import fix_encoding
from .regions import assign_regions, REGION_IDS


def word_count(s: pd.Series) -> pd.Series:
    return s.fillna("").str.split().str.len().fillna(0).astype(int)


def build_fact_news(
    input_csv: str,
    unique_dates: dict[str, int],
    unique_sources: dict[str, int],
    warehouse: str,
    chunk_size: int,
) -> tuple[int, int]:
    """
    ETL principal: lee el CSV en chunks y escribe fact_news particionado
    por year/month. Retorna (rows_written, rows_skipped).
    """
    writers = {}
    rows_written = 0
    rows_skipped = 0
    t_start      = time.time()

    try:
        for chunk_num, chunk in enumerate(
            pd.read_csv(
                input_csv,
                chunksize=chunk_size,
                encoding="utf-8-sig",
                on_bad_lines="skip",
            ),
            start=1,
        ):
            before = len(chunk)
            chunk = chunk.dropna(subset=["article_id", "publish_date"]) # eliminamos filas que no tienen article_id o publish_date
            chunk = chunk.drop_duplicates(subset="article_id") # no nos sirven articulos duplicados.
            rows_skipped += before - len(chunk) # filas eliminados x los proceso anteriores.

            chunk["_dt"] = pd.to_datetime(chunk["publish_date"], errors="coerce") # covertimos la fecha a objeto datetime para manipularlo mejor. 
            chunk = chunk.dropna(subset=["_dt"])
            chunk["_year"]  = chunk["_dt"].dt.year.astype(int) 
            chunk["_month"] = chunk["_dt"].dt.month.astype(int)

            chunk["title"] = chunk["title"].fillna("").apply(fix_encoding) # corregimos posibles errores de encoding
            chunk["body"]  = chunk["body"].fillna("").apply(fix_encoding) # same 

            # queremos fechas unicas [para la tabla de dimension], entonces mapeamos cada fecha a un id unico. Si no se encuentra la fecha, asignamos -1.
            chunk["date_id"]   = chunk["publish_date"].map(unique_dates).fillna(-1).astype(int) 
            chunk["source_id"] = chunk["source"].str.lower().map(unique_sources).fillna(-1).astype(int) # same

            chunk["region_name"] = assign_regions(chunk)
            chunk["region_id"]   = chunk["region_name"].map(REGION_IDS).astype(int)

            chunk["title_word_count"] = word_count(chunk["title"])
            chunk["body_word_count"]  = word_count(chunk["body"])

            for (year, month), part in chunk.groupby(["_year", "_month"]):
                # guardamos en el formato "year=XXXX/month=0X/part-X.parquet"
                file = f"{warehouse}/fact_news/year={year}/month={month:02d}/part-0.parquet"
                if file not in writers:
                    os.makedirs(os.path.dirname(file), exist_ok=True)
                    # asignamos un writer de parquet para cada particion (year/month) y lo guardamos en un diccionario para reutilizarlo.
                    writers[file] = pq.ParquetWriter(file, FACT_SCHEMA)

                table = pa.table(
                    {
                        "article_id":       part["article_id"].tolist(),
                        "date_id":          part["date_id"].tolist(),
                        "source_id":        part["source_id"].tolist(),
                        "region_id":        part["region_id"].tolist(),
                        "title":            part["title"].tolist(),
                        "body":             part["body"].tolist(),
                        "title_word_count": part["title_word_count"].tolist(),
                        "body_word_count":  part["body_word_count"].tolist(),
                    },
                    schema=FACT_SCHEMA,
                )
                writers[file].write_table(table)

            #actualizamos la cantidad de filas escritas. 
            rows_written += len(chunk)
            elapsed = time.time() - t_start
            rate    = rows_written / elapsed if elapsed > 0 else 0
            print(
                f"  chunk {chunk_num:4d} | "
                f"acumulado: {rows_written:>10,} | "
                f"descartadas: {rows_skipped:,} | "
                f"{rate:,.0f} filas/s",
                flush=True,
            )

    finally:
        for w in writers.values():
            w.close()

    return rows_written, rows_skipped
