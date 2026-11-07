#!/usr/bin/env python3
"""Une las particiones filtradas en data/terminos_filtrados.csv (columna "termino").

Incluye las variantes fusionadas de cada término para que el scraper descargue
todas las URL que el equipo agrupó. Uso: python3 unir_particiones.py
"""
import csv
from pathlib import Path

from inspeccion import DATA, normalizar

salida = DATA / "terminos_filtrados.csv"
vistos, filas = set(), []
for ruta in sorted((DATA / "particiones").glob("terminos_*.csv")):
    with ruta.open(encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            crudos = [fila["termino"]] + (fila.get("variantes_fusionadas") or "").split("|")
            # Algunas celdas traen varias formas separadas por ";".
            nombres = [n.strip() for c in crudos for n in c.split(";") if n.strip()]
            for n in nombres:
                if normalizar(n) not in vistos:
                    vistos.add(normalizar(n))
                    filas.append((n, ruta.stem.removeprefix("terminos_")))
with salida.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["termino", "integrante"])
    w.writerows(filas)
print(f"{len(filas)} términos -> {salida}")
