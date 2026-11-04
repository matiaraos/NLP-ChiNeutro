#!/usr/bin/env python3
"""Scraping de definiciones (T-A2) y construcción de la base del RAG.

Uso:
  python3 scraper.py mis_terminos.txt [--salida data] [--catalogo data/terminos.csv]

La entrada puede ser un .txt (un término por línea; se ignoran las líneas vacías
y las que empiezan con #) o un .csv con la columna "termino". Pasar
data/terminos.csv descarga todo el diccionario.

Cada término se busca en el catálogo sin distinguir mayúsculas, tildes ni
puntuación, y se descargan todas sus variantes ("Raja" y "raja").

Archivos en --salida:
  paginas.jsonl         Caché de páginas descargadas; permite reanudar.
  rag_documentos.jsonl  Un documento por término de la lista; lo lee rag.py.
  no_encontrados.txt    Términos que no están en el catálogo, con sugerencias.

El RAG se reconstruye solo con los términos de la lista, aunque la caché tenga
más. Código de salida: 0 si todo se descargó, 1 si la descarga quedó incompleta.
"""
import argparse
import csv
import difflib
import http.client
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from inspeccion import DATA, Cliente, normalizar, parsear_termino

REINTENTOS_S = (30, 120)
MAX_FALLOS_SEGUIDOS = 5


class Detener(Exception):
    """El sitio pidió bajar el ritmo o falla de forma persistente."""


def leer_lista(ruta):
    if ruta.suffix.lower() == ".csv":
        with ruta.open(encoding="utf-8-sig", newline="") as f:
            lector = csv.DictReader(f)
            if "termino" not in (lector.fieldnames or []):
                sys.exit(f'{ruta}: falta la columna "termino"')
            return [fila["termino"] for fila in lector if fila["termino"].strip()]
    lineas = ruta.read_text(encoding="utf-8-sig").splitlines()
    return [l.strip() for l in lineas if l.strip() and not l.lstrip().startswith("#")]


def leer_catalogo(ruta):
    """clave normalizada -> {url: {"termino", "esperadas"}}"""
    catalogo = {}
    with ruta.open(encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            urls = catalogo.setdefault(fila["clave_normalizada"], {})
            pagina = urls.setdefault(fila["url"], {"termino": fila["termino"], "esperadas": 0})
            pagina["esperadas"] += int(fila["n_definiciones"])
    return catalogo


def resolver(terminos, catalogo):
    claves, faltan = {}, []
    for termino in terminos:
        clave = normalizar(termino)
        if clave in catalogo:
            claves[clave] = True
        else:
            faltan.append(termino)
    return list(claves), faltan


def leer_paginas(ruta):
    paginas = {}
    if ruta.exists():
        with ruta.open(encoding="utf-8") as f:
            for linea in f:
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    continue  # Última línea cortada por una interrupción.
                paginas[registro["url"]] = registro
    return paginas


def descargar(cliente, url):
    status, pagina, error = None, "", None
    for espera in (*REINTENTOS_S, None):
        try:
            status, pagina = cliente.get(url, detener=False)
            error = None
        except (OSError, http.client.HTTPException) as e:
            status, pagina, error = None, "", repr(e)
        if status == 429:
            raise Detener(f"{url} respondió 429")
        if (status is not None and status < 500) or espera is None:
            break
        print(f"  {url}: {error or status}; reintento en {espera} s", flush=True)
        time.sleep(espera)
    definiciones = parsear_termino(pagina) if status == 200 else []
    registro = {
        "url": url,
        "estado": "ok" if definiciones else "error",
        "status": status,
        "fecha_extraccion_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "definiciones": definiciones,
    }
    if error:
        registro["error"] = error
    return registro


def scrapear(pendientes, paginas, ruta):
    """Descarga las URL pendientes. Devuelve el motivo si se detuvo antes de terminar."""
    cliente = Cliente()
    fallos_seguidos = 0
    with ruta.open("a", encoding="utf-8") as f:
        for i, url in enumerate(pendientes, 1):
            try:
                registro = descargar(cliente, url)
            except Detener as e:
                return str(e)
            except KeyboardInterrupt:
                return "interrumpido por el usuario"
            paginas[url] = registro
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
            f.flush()
            fallos_seguidos = 0 if registro["estado"] == "ok" else fallos_seguidos + 1
            print(f"[{i}/{len(pendientes)}] {registro['estado']} "
                  f"({len(registro['definiciones'])} def.) {url}", flush=True)
            if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                return f"{fallos_seguidos} páginas seguidas con error"
    return None


def construir_documentos(claves, catalogo, paginas):
    documentos = []
    for clave in claves:
        variantes = catalogo[clave]
        definiciones, vistos, completo = [], set(), True
        for url in variantes:
            registro = paginas.get(url, {})
            if registro.get("estado") != "ok":
                completo = False
                continue
            for d in registro["definiciones"]:
                # Las variantes de un término podrían compartir definiciones.
                identidad = d["id"] if d["id"] is not None else d["definicion"]
                if identidad not in vistos:
                    vistos.add(identidad)
                    definiciones.append({
                        **d, "url": url, "fecha_extraccion_utc": registro["fecha_extraccion_utc"],
                    })
        if not definiciones:
            continue
        definiciones.sort(key=lambda d: d["votos"] if d["votos"] is not None else float("-inf"),
                          reverse=True)
        principal = max(variantes.values(), key=lambda v: v["esperadas"])["termino"]
        documentos.append({
            "clave": clave,
            "termino": principal,
            "variantes": sorted({v["termino"] for v in variantes.values()}),
            "completo": completo,
            "definiciones": definiciones,
        })
    return documentos


def escribir_atomico(ruta, lineas):
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text("".join(l + "\n" for l in lineas), encoding="utf-8")
    temporal.replace(ruta)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("terminos", type=Path, help="archivo .txt o .csv con los términos")
    ap.add_argument("--catalogo", type=Path, default=DATA / "terminos.csv")
    ap.add_argument("--salida", type=Path, default=DATA)
    args = ap.parse_args()

    catalogo = leer_catalogo(args.catalogo)
    claves, faltan = resolver(leer_lista(args.terminos), catalogo)
    if not claves:
        sys.exit("Ningún término de la lista está en el catálogo.")

    args.salida.mkdir(parents=True, exist_ok=True)
    ruta_paginas = args.salida / "paginas.jsonl"
    paginas = leer_paginas(ruta_paginas)
    urls = list(dict.fromkeys(u for c in claves for u in catalogo[c]))
    pendientes = [u for u in urls if paginas.get(u, {}).get("estado") != "ok"]
    print(f"{len(claves)} términos, {len(urls)} páginas, {len(pendientes)} por descargar "
          f"(unos {len(pendientes) * 3.7 / 60:.0f} min)", flush=True)

    motivo = scrapear(pendientes, paginas, ruta_paginas) if pendientes else None

    documentos = construir_documentos(claves, catalogo, paginas)
    escribir_atomico(args.salida / "rag_documentos.jsonl",
                     [json.dumps(d, ensure_ascii=False) for d in documentos])

    ruta_faltan = args.salida / "no_encontrados.txt"
    ruta_faltan.unlink(missing_ok=True)
    if faltan:
        lineas = []
        for t in faltan:
            sugerencias = difflib.get_close_matches(normalizar(t), catalogo, n=3, cutoff=0.7)
            lineas.append(f"{t}\t{', '.join(sugerencias)}")
        escribir_atomico(ruta_faltan, lineas)

    # Avisa si el parser pudo perder datos sin que la página diera error.
    esperadas = {u: v["esperadas"] for c in claves for u, v in catalogo[c].items()}
    distintas = [u for u in urls if paginas.get(u, {}).get("estado") == "ok" and
                 len(paginas[u]["definiciones"]) != esperadas[u]]
    sin_descargar = [u for u in urls if paginas.get(u, {}).get("estado") != "ok"]
    n_defs = sum(len(d["definiciones"]) for d in documentos)
    print(f"\nRAG: {len(documentos)} términos, {n_defs} definiciones -> "
          f"{args.salida / 'rag_documentos.jsonl'}")
    print(f"No están en el catálogo: {len(faltan)}"
          + (f" (ver {ruta_faltan})" if faltan else ""))
    print(f"Páginas sin descargar: {len(sin_descargar)}")
    for u in sin_descargar[:20]:
        print("  ", u)
    print(f"Páginas cuyo n.º de definiciones difiere del catálogo: {len(distintas)}")
    for u in distintas[:20]:
        print("  ", u)
    if motivo:
        print(f"DETENIDO: {motivo}. Vuelve a ejecutar para reanudar.")
    return 1 if sin_descargar else 0


if __name__ == "__main__":
    sys.exit(main())
