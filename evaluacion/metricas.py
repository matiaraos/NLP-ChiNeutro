"""Métricas de la evaluación. Trabajan sobre un CSV de predicciones (ver datos.py), no sobre el modelo.

- BLEU (sacrebleu, de corpus) contra la referencia.
- Términos chilenos retenidos: de las expresiones chilenas presentes en la entrada, % que sigue en la salida
  (más bajo es mejor). Con `clave` se busca esa expresión; sin ella (p. ej. Reddit) se busca todo el diccionario,
  excepto las expresiones que también son palabras neutras (expresiones_ambiguas.json).
- BERTScore F1 (similitud semántica) contra la referencia; si no hay referencia, contra la frase de entrada.
- Juez LLM: Claude Opus 5.5 vía Claude Code (juez.py).
"""
import json
from pathlib import Path

import pandas as pd

from . import jerga
from .datos import RUTA_PARES, RUTA_RAG, completar

BERTSCORE_MODELO = "xlm-roberta-large"   # fijo: si cambia, los números no se comparan entre modelos
RUTA_AMBIGUAS = Path(__file__).with_name("expresiones_ambiguas.json")


# ---------------------------------------------------------------- BLEU

def bleu(df):
    import sacrebleu
    if "frase_neutralizada" not in df or df.frase_neutralizada.isna().all():
        return None
    con_ref = df[df.frase_neutralizada.notna()]
    return round(sacrebleu.corpus_bleu(con_ref.prediccion.tolist(),
                                       [con_ref.frase_neutralizada.astype(str).tolist()]).score, 2)


# ---------------------------------------------------------------- términos chilenos retenidos

_LEXICO = None


def lexico():
    global _LEXICO
    if _LEXICO is None:
        _LEXICO = jerga.Lexico(RUTA_RAG)
    return _LEXICO


def expresiones_ambiguas(recalcular=False, min_apariciones=2):
    """Claves que el detector encuentra en las frases neutras del dataset (p. ej. *taco*, *rico*): también son
    palabras neutras. Solo se excluyen cuando no hay `clave` (Reddit). Se guarda en expresiones_ambiguas.json."""
    if RUTA_AMBIGUAS.exists() and not recalcular:
        return set(json.loads(RUTA_AMBIGUAS.read_text(encoding="utf-8")))
    conteo = {}
    for frase in pd.read_csv(RUTA_PARES, encoding="utf-8-sig").frase_neutralizada.astype(str):
        for clave in lexico().buscar(frase):
            conteo[clave] = conteo.get(clave, 0) + 1
    ambiguas = sorted(c for c, n in conteo.items() if n >= min_apariciones)
    RUTA_AMBIGUAS.write_text(json.dumps(ambiguas, ensure_ascii=False, indent=0), encoding="utf-8")
    return set(ambiguas)


def terminos_retenidos(df):
    """Devuelve (% retenido, columnas por frase)."""
    lex = lexico()
    tiene_clave = df["clave"].notna() if "clave" in df else pd.Series(False, index=df.index)
    ambiguas = expresiones_ambiguas() if (~tiene_clave).any() else set()
    en_fuente, retenidos = [], []
    for i, fila in df.iterrows():
        if tiene_clave[i]:
            fuente = [fila.clave] if jerga.contiene(fila.frase_chilena, lex.de(fila.clave)) else []
        else:
            fuente = lex.buscar(fila.frase_chilena, excluir=ambiguas)
        en_fuente.append(fuente)
        retenidos.append([c for c in fuente if jerga.contiene(fila.prediccion, lex.de(c))])
    n_fuente = sum(map(len, en_fuente))
    pct = round(100 * sum(map(len, retenidos)) / n_fuente, 1) if n_fuente else None
    return pct, pd.DataFrame({"terminos_en_entrada": ["; ".join(x) for x in en_fuente],
                              "terminos_retenidos": ["; ".join(x) for x in retenidos]}, index=df.index)


# ---------------------------------------------------------------- BERTScore

def bertscore(df, modelo=BERTSCORE_MODELO, dispositivo=None):
    """Devuelve (F1 medio, contra qué se comparó, F1 por frase)."""
    from bert_score import BERTScorer
    con_ref = "frase_neutralizada" in df and df.frase_neutralizada.notna().all()
    contra = "referencia" if con_ref else "entrada"
    objetivo = df.frase_neutralizada if con_ref else df.frase_chilena
    scorer = BERTScorer(model_type=modelo, lang="es", rescale_with_baseline=True, device=dispositivo)
    _, _, f1 = scorer.score(df.prediccion.tolist(), objetivo.astype(str).tolist())
    return round(float(f1.mean()), 4), contra, f1.numpy()


# ---------------------------------------------------------------- todo junto

def evaluar(predicciones, juez=True, bertscore_activo=True, guardar=True, **opciones_juez):
    """Calcula todas las métricas de un CSV (ruta) o DataFrame de predicciones.

    Devuelve (resumen: dict, por_frase: DataFrame). Con una ruta y guardar=True, escribe junto al CSV:
    <nombre>_evaluacion.json (resumen) y <nombre>_por_frase.csv (métricas y veredicto del juez por frase).
    opciones_juez: ver juez.juzgar (muestra, effort, paralelo, claude_bin...).
    """
    ruta = Path(predicciones) if isinstance(predicciones, (str, Path)) else None
    df = completar(pd.read_csv(ruta) if ruta else predicciones)
    resumen = {"n": len(df)}
    por_frase = df.copy()

    resumen["bleu"] = bleu(df)
    resumen["terminos_retenidos_%"], cols = terminos_retenidos(df)
    por_frase = por_frase.join(cols)
    if bertscore_activo:
        resumen["bertscore_f1"], resumen["bertscore_contra"], f1 = bertscore(df)
        por_frase["bertscore_f1"] = f1.round(4)
    if juez:
        from .juez import juzgar
        r, veredictos = juzgar(df, **opciones_juez)
        resumen.update(r)
        por_frase = por_frase.join(veredictos[["sentido", "jerga", "fluidez", "aceptable", "comentario"]]
                                   .add_prefix("juez_"))

    if ruta and guardar:
        base = ruta.with_suffix("")
        Path(f"{base}_evaluacion.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
        por_frase.to_csv(f"{base}_por_frase.csv", index=False)
        print(f"-> {base}_evaluacion.json\n-> {base}_por_frase.csv")
    return resumen, por_frase
