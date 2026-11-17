"""Rutas del repo, split oficial y formato del CSV de predicciones.

El CSV que entrega cada modelo necesita dos columnas:
    frase_chilena   la frase de entrada
    prediccion      la traducción del modelo
y, si las frases vienen del dataset, `id_mistral` (con él se agregan la referencia, la clave y la definición).
Sin `id_mistral` se pueden dar directamente `frase_neutralizada` (referencia) y `clave` (expresión chilena).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_PARES = RAIZ / "data/entrenamiento/dataset_chilenismos_opus.csv"
RUTA_SPLIT = RAIZ / "data/entrenamiento/splits/split_oficial.csv"
RUTA_RAG = RAIZ / "scraping/data/rag_documentos.jsonl"


def cargar_split(split="test"):
    """Pares del split oficial ('train', 'val' o 'test'): son las frases que cada modelo debe traducir."""
    pares = pd.read_csv(RUTA_PARES, encoding="utf-8-sig")
    s = pd.read_csv(RUTA_SPLIT)[["id_mistral", "split"]]
    pares = pares.merge(s, on="id_mistral", validate="one_to_one")
    return pares[pares.split == split].reset_index(drop=True)


def completar(df):
    """Valida las columnas obligatorias y agrega desde el dataset las que falten, usando `id_mistral`."""
    for col in ("frase_chilena", "prediccion"):
        if col not in df:
            raise ValueError(f"falta la columna obligatoria '{col}' (columnas: {list(df.columns)})")
    df = df.copy()
    df["prediccion"] = df["prediccion"].fillna("").astype(str)
    faltan = [c for c in ("clave", "termino", "definicion", "frase_neutralizada") if c not in df]
    if "id_mistral" in df and faltan:
        pares = pd.read_csv(RUTA_PARES, encoding="utf-8-sig")[["id_mistral", *faltan]]
        df = df.merge(pares, on="id_mistral", how="left", validate="many_to_one")
    return df.reset_index(drop=True)
