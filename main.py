"""
ETL Pipeline — Tarea 1 IIC 2440
Uso:
    python main.py                        # dataset completo
    python main.py --sample               # usa noti.csv (50k filas)
    python main.py --input mi_archivo.csv
    python main.py --chunk-size 200000
    python main.py --warehouse mi_wh/
"""
import argparse
import time

from pipeline.config import CHUNK_SIZE, INPUT_CSV, SAMPLE_CSV, WAREHOUSE
from pipeline.dimensions import build_dim_date, build_dim_region, build_dim_source, scan_csv
from pipeline.fact import build_fact_news
from pipeline.validation import run_all_validations


def main() -> None:
    parser = argparse.ArgumentParser(description="ETL pipeline — Tarea 1 IIC 2440")
    parser.add_argument("--input",      default=None,       help="Archivo CSV de entrada")
    parser.add_argument("--warehouse",  default=WAREHOUSE,  help="Directorio de salida del warehouse")
    parser.add_argument("--chunk-size", default=CHUNK_SIZE, type=int, help="Filas por chunk")
    parser.add_argument("--sample",     action="store_true", help="Usar noti.csv en lugar del CSV completo")
    args = parser.parse_args()

    input_csv  = args.input or (SAMPLE_CSV if args.sample else INPUT_CSV)
    warehouse  = args.warehouse
    chunk_size = args.chunk_size

    print("=" * 60)
    print("ETL Pipeline — Tarea 1 IIC 2440")
    print("=" * 60)
    print(f"  Entrada     : {input_csv}")
    print(f"  Warehouse   : {warehouse}/")
    print(f"  Chunk size  : {chunk_size:,}")
    print()

    t0 = time.time()

    # ── Paso 1: escaneo ligero ────────────────────────────────────────────────
    print("[1/4] Escaneando CSV (pass 1: fechas y fuentes)...")
    unique_dates, unique_sources, raw_row_count = scan_csv(input_csv, chunk_size)
    print(f"      Filas totales : {raw_row_count:,}")
    print(f"      Fechas únicas : {len(unique_dates)}")
    print(f"      Fuentes únicas: {len(unique_sources)}")
    print()

    # ── Paso 2: dimensiones ───────────────────────────────────────────────────
    print("[2/4] Construyendo tablas de dimensiones...")
    dim_date   = build_dim_date(unique_dates, warehouse)
    print(f"      dim_date   : {len(dim_date)} filas")
    dim_source = build_dim_source(unique_sources, warehouse)
    print(f"      dim_source : {len(dim_source)} filas")
    dim_region = build_dim_region(warehouse)
    print(f"      dim_region : {len(dim_region)} filas")
    print()

    # ── Paso 3: fact_news ─────────────────────────────────────────────────────
    print("[3/4] Construyendo fact_news (ETL principal)...")
    rows_written, rows_skipped = build_fact_news(
        input_csv, unique_dates, unique_sources, warehouse, chunk_size
    )
    print(f"\n      Filas escritas   : {rows_written:,}")
    print(f"      Filas descartadas: {rows_skipped:,}")
    print()

    # ── Paso 4: validaciones ──────────────────────────────────────────────────
    print("[4/4] Ejecutando validaciones...")
    run_all_validations(warehouse, raw_row_count)

    elapsed = time.time() - t0
    print(f"\nTiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
