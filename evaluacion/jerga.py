"""Detección de expresiones chilenas en un texto, tolerante a conjugación, género y número.

La búsqueda literal de la `clave` no ve *cogotearon* (cogotear), *craneando* (cranear) ni *flaites* (flaite).
Aquí cada palabra de la expresión se convierte en un patrón (sobre texto sin tildes, puntuación ni ñ):

- Verbos en infinitivo (-ar/-er/-ir, también con pronombre: -arse, -arle...): la raíz más una terminación verbal
  (*cranear* -> craneo, craneai, craneando, crearon...). La raíz debe tener al menos 3 letras (4 si la expresión
  es de una sola palabra); si no, se exige la palabra exacta.
- Otras palabras de 5 letras o más: la palabra, o la palabra sin su vocal final más una terminación de género,
  número o diminutivo (*flaite* -> flaites, flaita, flaitito...). Las de 4 letras aceptan solo el plural.
- Palabras cortas: exactas, salvo los artículos, que se aceptan entre sí (*echar la talla* -> *echó una talla*).

También se prueban las variantes ortográficas del diccionario (rag_documentos.jsonl).

Es una aproximación: no ve raíces irregulares (*poner* -> *puso*), cambios ortográficos (*sacar* -> *saqué*)
ni las formas sin *d* (*machucao*). Y algunas expresiones son también palabras neutras (*taco*, *pega*):
`metricas.expresiones_ambiguas()` las identifica.

Medido sobre el dataset: encuentra la expresión en el 89 % de las frases chilenas (la búsqueda literal, 69 %) y da
falsos positivos en el 0,4 % de las frases neutras.
"""
import re
import unicodedata
from functools import lru_cache

ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
TERMINACIONES = r"(?:a|o|e|as|os|es|s|it[oa]s?|cit[oa]s?|ill[oa]s?|on|ona|ones|onas|azo|azos)"
CLITICOS = r"(?:me|te|se|le|lo|la|nos|les|los|las)*"
# Terminaciones verbales de -ar/-er/-ir sin tildes, incluido el voseo chileno (estai, comis); pueden llevar pronombre
VERBALES = (r"(?:(?:ar|er|ir)(?:e|a|emos|an|ia|ias|iamos|ian|is|as)?|o|as|es|is|ai|ei|a|e|i|amos|emos|imos|an|en|"
            r"aste|iste|asteis|aron|ieron|io|aba|abas|abamos|aban|ia|ias|iamos|ian|ando|iendo|ado|ada|ados|adas|"
            r"ido|ida|idos|idas|ao|ia|ara|aras|aramos|aran|iera|ieras|ieramos|ieran|ase|ases|iese|ieses|en|emos)"
            + CLITICOS)


def norm(s):
    """Minúsculas, sin tildes ni puntuación (la ñ se conserva). Igual que `comun.norm` de models/gemma4."""
    s = str(s).lower().replace("ñ", "\0")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("\0", "ñ")
    s = re.sub(r"[^a-z0-9ñ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _plegar(s):
    """Texto normalizado para comparar: además la ñ pasa a n (muchas claves del diccionario vienen sin ñ)."""
    return norm(s).replace("ñ", "n")


def _patron_palabra(p, una_sola):
    if p in ARTICULOS:
        return "(?:" + "|".join(sorted(ARTICULOS, key=len, reverse=True)) + ")"
    m = re.fullmatch(r"(\w+?)(ar|er|ir)(?:se|le|la|lo|les|las|los|me|te)?", p)
    if m and len(p) >= 5 and len(m.group(1)) >= (4 if una_sola else 3):
        return re.escape(m.group(1)) + VERBALES
    if len(p) >= 5:
        base = p[:-1] if p[-1] in "aeo" else p
        return f"(?:{re.escape(p)}|{re.escape(base)}{TERMINACIONES})"
    if len(p) == 4:
        return re.escape(p) + "(?:s|es)?"
    return re.escape(p)


@lru_cache(maxsize=None)
def patron_expresion(forma):
    """Regex compilada (sobre texto normalizado) para una forma de la expresión."""
    palabras = _plegar(forma).split()
    if not palabras:
        return None
    una_sola = len(palabras) == 1
    cuerpo = r"\s+".join(_patron_palabra(p, una_sola) for p in palabras)
    return re.compile(rf"(?<!\S){cuerpo}(?!\S)")


def contiene(texto, formas):
    """True si el texto contiene alguna de las formas (str o lista de str) de la expresión."""
    if isinstance(formas, str):
        formas = [formas]
    t = _plegar(texto)
    return any(p.search(t) for p in map(patron_expresion, formas) if p is not None)


class Lexico:
    """Formas de cada expresión del diccionario: la clave, el término y sus variantes."""

    def __init__(self, ruta_rag):
        import json
        self.formas = {}
        with open(ruta_rag, encoding="utf-8") as f:
            for linea in f:
                if linea.strip():
                    d = json.loads(linea)
                    fs = [d["clave"], d.get("termino", ""), *d.get("variantes", [])]
                    self.formas[d["clave"]] = sorted({norm(x) for x in fs if norm(x)})

    def de(self, clave):
        return self.formas.get(clave) or [norm(clave)]

    def buscar(self, texto, claves=None, excluir=()):
        """Claves del diccionario presentes en el texto (para textos sin `clave` conocida, como Reddit)."""
        t = _plegar(texto)
        hallazgos = []
        for clave in (claves or self.formas):
            if clave in excluir:
                continue
            if any(p.search(t) for p in map(patron_expresion, self.de(clave)) if p is not None):
                hallazgos.append(clave)
        return hallazgos
