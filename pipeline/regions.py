import re
import pandas as pd

REGION_ALIASES: dict[str, list[str]] = {
    "Arica y Parinacota": [
        "arica", "parinacota", "xv region", "region de arica",
    ],
    "Tarapaca": [
        "tarapaca", "iquique", "alto hospicio", "i region", "primera region",
    ],
    "Antofagasta": [
        "antofagasta", "calama", "tocopilla", "mejillones", "taltal",
        "ii region", "segunda region",
    ],
    "Atacama": [
        "atacama", "copiapo", "vallenar", "chanaral", "huasco",
        "iii region", "tercera region",
    ],
    "Coquimbo": [
        "coquimbo", "la serena", "ovalle", "illapel", "los vilos",
        "iv region", "cuarta region",
    ],
    "Valparaiso": [
        "valparaiso", "vina del mar", "quilpue", "villa alemana",
        "san antonio", "quillota", "los andes", "quintero", "limache",
        "v region", "quinta region",
    ],
    "Metropolitana": [
        "santiago", "region metropolitana", "metropolitana", "rm",
        "maipu", "nunoa", "providencia", "las condes", "la florida",
        "penalolen", "pudahuel", "estacion central", "macul",
        "puente alto", "san bernardo",
    ],
    "OHiggins": [
        "ohiggins", "o'higgins", "rancagua", "san fernando",
        "pichilemu", "santa cruz", "vi region", "sexta region",
    ],
    "Maule": [
        "maule", "talca", "curico", "linares", "cauquenes",
        "constitucion", "vii region", "septima region",
    ],
    "Nuble": [
        "nuble", "chillan", "san carlos", "bulnes",
        "xvi region", "region de nuble",
    ],
    "Biobio": [
        "biobio", "concepcion", "conce", "talcahuano", "los angeles",
        "coronel", "lota", "chiguayante", "hualpen", "tome",
        "viii region", "octava region",
    ],
    "La Araucania": [
        "araucania", "temuco", "villarrica", "pucon", "angol",
        "victoria", "nueva imperial", "ix region", "novena region",
    ],
    "Los Rios": [
        "los rios", "valdivia", "la union", "panguipulli",
        "xiv region", "region de los rios",
    ],
    "Los Lagos": [
        "los lagos", "puerto montt", "osorno", "ancud", "castro",
        "chiloe", "puerto varas", "calbuco",
        "x region", "decima region",
    ],
    "Aysen": [
        "aysen", "coyhaique", "coihaique", "chile chico", "cochrane",
        "xi region", "undecima region",
    ],
    "Magallanes": [
        "magallanes", "punta arenas", "puerto natales",
        "torres del paine", "xii region", "duodecima region",
    ],
    "Desconocida": [],
}

REGION_IDS: dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(REGION_ALIASES)
}

_REGION_PATTERNS: dict[str, str] = {
    region: "|".join(r"\b" + re.escape(a) + r"\b" for a in aliases)
    for region, aliases in REGION_ALIASES.items()
    if aliases
}


def normalize_series(s: pd.Series) -> pd.Series:
    """lowercase → NFD → elimina no-ASCII (vectorizado)."""
    return (
        s.fillna("")
         .str.lower()
         .str.normalize("NFD")
         .str.encode("ascii", errors="ignore")
         .str.decode("ascii")
    )


def assign_regions(chunk: pd.DataFrame) -> pd.Series:
    """
    Infiere la región de cada artículo de forma vectorizada.

    Puntuación: title hit = 3 pts, body hit (primeros 800 chars) = 1 pt.
    Empate o puntaje 0 → 'Desconocida'.
    """
    norm_title = normalize_series(chunk["title"])
    norm_body  = normalize_series(chunk["body"].str[:800])

    score_data = {
        region: norm_title.str.count(pat) * 3 + norm_body.str.count(pat)
        for region, pat in _REGION_PATTERNS.items()
    }

    scores     = pd.DataFrame(score_data, index=chunk.index)
    max_scores = scores.max(axis=1)
    n_winners  = scores.eq(max_scores, axis=0).sum(axis=1)

    result = scores.idxmax(axis=1)
    result[max_scores == 0] = "Desconocida"
    result[n_winners > 1]   = "Desconocida"
    return result
