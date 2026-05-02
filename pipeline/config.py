import pyarrow as pa

CHUNK_SIZE = 100_000
WAREHOUSE  = "warehouse_data"
INPUT_CSV  = "noticias_chile_2023_2025.csv"
SAMPLE_CSV = "noti.csv"

FACT_SCHEMA = pa.schema([
    ("article_id",       pa.string()),
    ("date_id",          pa.int32()),
    ("source_id",        pa.int32()),
    ("region_id",        pa.int32()),
    ("title",            pa.string()),
    ("body",             pa.string()),
    ("title_word_count", pa.int32()),
    ("body_word_count",  pa.int32()),
])
