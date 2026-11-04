#!/usr/bin/env python3
"""Recuperación de contexto del diccionario para generar el sintético (T-B1).

El generador ya sabe qué término quiere usar, así que la recuperación es una
búsqueda exacta por término normalizado, con las definiciones ordenadas por votos.

Desde código:
  from rag import RAG
  rag = RAG("data/rag_documentos.jsonl")
  rag.contexto("cachai", k=3, min_votos=0)

Desde la terminal:
  python3 rag.py cachai -k 3 --min-votos 0
"""
import argparse
import json
from pathlib import Path

from inspeccion import DATA, normalizar


class RAG:
    def __init__(self, ruta=DATA / "rag_documentos.jsonl"):
        with Path(ruta).open(encoding="utf-8") as f:
            documentos = [json.loads(linea) for linea in f]
        self.documentos = {d["clave"]: d for d in documentos}

    def terminos(self):
        return [d["termino"] for d in self.documentos.values()]

    def recuperar(self, termino, k=3, min_votos=None, solo_con_ejemplo=False):
        """Devuelve hasta k definiciones. KeyError si el término no está en la base."""
        clave = normalizar(termino)
        if clave not in self.documentos:
            raise KeyError(f"Término fuera de la base del RAG: {termino!r}")
        resultado = []
        for d in self.documentos[clave]["definiciones"]:
            if min_votos is not None and (d["votos"] is None or d["votos"] < min_votos):
                continue
            if solo_con_ejemplo and not d["ejemplo"]:
                continue
            resultado.append(d)
        return resultado[:k]

    def contexto(self, termino, k=3, min_votos=None, solo_con_ejemplo=False):
        """Texto listo para el prompt; queda vacío si ninguna definición pasa los filtros."""
        definiciones = self.recuperar(termino, k, min_votos, solo_con_ejemplo)
        if not definiciones:
            return ""
        lineas = [f"Término: {self.documentos[normalizar(termino)]['termino']}"]
        for i, d in enumerate(definiciones, 1):
            lineas.append(f"{i}. Definición: {d['definicion']}")
            if d["ejemplo"]:
                lineas.append(f"   Ejemplo: {d['ejemplo']}")
        return "\n".join(lineas)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Muestra el contexto de un término")
    ap.add_argument("termino")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--min-votos", type=int)
    ap.add_argument("--base", type=Path, default=DATA / "rag_documentos.jsonl")
    args = ap.parse_args()
    print(RAG(args.base).contexto(args.termino, args.k, args.min_votos))
