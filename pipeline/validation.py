import os

import pandas as pd


def _load_warehouse(warehouse: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dim_date   = pd.read_parquet(f"{warehouse}/dim_date/dim_date.parquet")
    dim_source = pd.read_parquet(f"{warehouse}/dim_source/dim_source.parquet")
    dim_region = pd.read_parquet(f"{warehouse}/dim_region/dim_region.parquet")
    fact_keys  = pd.read_parquet(
        f"{warehouse}/fact_news/",
        columns=["article_id", "date_id", "source_id", "region_id"],
    )
    return dim_date, dim_source, dim_region, fact_keys


def _check(condition: bool, name: str, msg_ok: str, msg_fail: str) -> bool:
    if condition:
        print(f"  PASA [{name}]: {msg_ok}")
    else:
        print(f"  FALLO [{name}]: {msg_fail}")
    return condition


def run_all_validations(warehouse: str, raw_row_count: int) -> bool:
    """Corre las 5 validaciones y retorna True si todas pasan."""
    print("Cargando warehouse para validaciones...")
    dim_date, dim_source, dim_region, fact_keys = _load_warehouse(warehouse)
    print(f"  fact_news cargada: {len(fact_keys):,} filas\n")

    results = []

    # 1. Consistencia referencial
    orphan_date   = (~fact_keys["date_id"].isin(dim_date["date_id"])).sum()
    orphan_source = (~fact_keys["source_id"].isin(dim_source["source_id"])).sum()
    orphan_region = (~fact_keys["region_id"].isin(dim_region["region_id"])).sum()
    ok = orphan_date == 0 and orphan_source == 0 and orphan_region == 0
    results.append(_check(
        ok, "1-referencial",
        "sin llaves foráneas huérfanas",
        f"date={orphan_date}, source={orphan_source}, region={orphan_region} huérfanas",
    ))

    # 2. Conteo de filas vs CSV crudo
    results.append(_check(
        len(fact_keys) <= raw_row_count, "2-conteo",
        f"fact_news ({len(fact_keys):,}) ≤ CSV crudo ({raw_row_count:,})",
        f"fact_news ({len(fact_keys):,}) > CSV crudo ({raw_row_count:,}) — imposible",
    ))

    # 3. Sin PKs duplicadas en dimensiones
    dims_ok = True
    for name, df, pk in [
        ("dim_date",   dim_date,   "date_id"),
        ("dim_source", dim_source, "source_id"),
        ("dim_region", dim_region, "region_id"),
    ]:
        dups = df[pk].duplicated().sum()
        ok   = dups == 0
        dims_ok = dims_ok and ok
        _check(ok, f"3-pks/{name}", "sin duplicados", f"{dups} PKs duplicadas")
    results.append(dims_ok)

    # 4. Distribución de particiones correcta
    date_lookup = dim_date.set_index("date_id")[["anio", "mes"]]
    fact_parts  = fact_keys[["date_id"]].join(date_lookup, on="date_id")
    combos      = fact_parts.groupby(["anio", "mes"]).size().reset_index()
    missing     = 0
    for _, row in combos.iterrows():
        p = f"{warehouse}/fact_news/year={int(row['anio'])}/month={int(row['mes']):02d}/part-0.parquet"
        if not os.path.exists(p):
            print(f"    FALTA partición: {p}")
            missing += 1
    results.append(_check(
        missing == 0, "4-particiones",
        f"{len(combos)} particiones verificadas",
        f"{missing} particiones faltantes",
    ))

    # 5. Sin article_id duplicados en fact_news
    dups = len(fact_keys) - fact_keys["article_id"].nunique()
    results.append(_check(
        dups == 0, "5-article_id-unicos",
        "todos los article_id son únicos",
        f"{dups} article_ids duplicados",
    ))

    all_ok = all(results)
    print(f"\n{'✓ Todas las validaciones pasaron.' if all_ok else '✗ Algunas validaciones fallaron.'}")
    return all_ok
