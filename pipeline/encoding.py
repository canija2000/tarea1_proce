import re
import ftfy

# caracteres CJK que aparecen como sustitución de vocales con tilde
MANUAL_FIXES = {
    "豩": "á",  # á
    "贸": "ó",  # ó
    "贡": "é",  # é
    "赤": "í",  # í
    "资": "ú",  # ú
    "谷": "ü",  # ü
}

# formamos patron de estos caracteres 

MANUAL_PAT = re.compile("|".join(re.escape(k) for k in MANUAL_FIXES))

# encoding roto (evita llamar ftfy innecesariamente [es mas costoso que un simple search])
BAD_ENCODING = re.compile(
    r"[\xc3\xc2豩贸贡赤资]|Ã|Â"
)

def fix_encoding(text: str) -> str:
    """Aplica ftfy solo si hay indicios de mojibake, luego corrige sustituciones CJK."""
    if not isinstance(text, str) or not text:
        return ""
    if BAD_ENCODING.search(text):
        text = ftfy.fix_text(text)
    if MANUAL_PAT.search(text):
        text = MANUAL_PAT.sub(lambda m: MANUAL_FIXES[m.group()], text)
    return text
