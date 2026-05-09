import time
from tqdm import tqdm

from pipeline.config import CHUNK_SIZE, INPUT_CSV, SAMPLE_CSV, WAREHOUSE
from pipeline.dimensions import build_dim_date, build_dim_region, build_dim_source, scan_csv
from pipeline.fact import build_fact_news
from pipeline.validation import run_all_validations


def main() -> None:

    input_csv  = INPUT_CSV
    warehouse  = WAREHOUSE
    chunk_size = CHUNK_SIZE

    print("=" * 60)
    print("ETL Pipeline — Tarea 1 IIC 2440  **(¨_¨)**".center(60))
    print("=" * 60)
    print(f"  Entrada     : {input_csv:<50}")
    print(f"  Warehouse   : {warehouse:<50}/")
    print(f"  Chunk size  : {chunk_size:,}")
    print()

    t0 = time.time()

    # 1: escaneo indices para dimensiones
    print("[1/4] Escaneando CSV...")
    unique_dates, unique_sources, raw_row_count = scan_csv(input_csv, chunk_size)
    print(f"      Filas totales : {raw_row_count:,}")
    print(f"      Fechas únicas : {len(unique_dates)}")
    print(f"      Fuentes únicas: {len(unique_sources)}")

    # 2: construir dimensiones
    print("[2/4] Construyendo dimensiones...")
    dim_date   = build_dim_date(unique_dates, warehouse)
    print(f"      dim_date   : {len(dim_date)} filas")
    dim_source = build_dim_source(unique_sources, warehouse)
    print(f"      dim_source : {len(dim_source)} filas")
    dim_region = build_dim_region(warehouse)
    print(f"      dim_region : {len(dim_region)} filas")

    # 3: fact_news (ETL principal)
    print("[3/4] Construyendo fact_news...")
    rows_written, rows_skipped = build_fact_news(
        input_csv, unique_dates, unique_sources, warehouse, chunk_size
    )
    print(f"      Filas escritas   : {rows_written:,}")
    print(f"      Filas descartadas: {rows_skipped:,}")

    # 4: validaciones
    print("[4/4] Ejecutando validaciones...")
    run_all_validations(warehouse, raw_row_count)

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
