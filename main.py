import argparse

from pipeline.pipeline import main as etl_pipeline
from map_reduce.query_total import main as query_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL Pipeline + MapReduce Queries")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--etl",   action="store_true", help="Ejecutar solo el ETL pipeline")
    group.add_argument("--query", action="store_true", help="Ejecutar solo las queries MapReduce")
    args = parser.parse_args()

    # Si no especifica nada, ejecuta ambos
    run_etl   = not args.query  # ejecuta ETL si no pidió solo query
    run_query = not args.etl    # ejecuta query si no pidió solo etl

    if run_etl:
        print("=" * 60)
        print(f"{'Ejecutando ETL Pipeline  ¿(°_°)?':^60}")
        print("=" * 60)
        etl_pipeline()

    if run_query:
        print("\n" + "=" * 60)
        print(f"{'Ejecutando MapReduce  ¿(~.~)?':^60}")
        print("=" * 60)
        query_total()