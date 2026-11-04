#!/usr/bin/env python3
"""Inspección de factibilidad de diccionariochileno.cl (T-A1).

Subcomandos:
  listar     Descarga /terms (1 solicitud) y escribe data/terminos.csv
  verificar  Compara /terms con /terms/a ... /terms/z (26 solicitudes)
  muestrear  Lee N páginas de término al azar y escribe data/muestra_definiciones.jsonl

Respeta robots.txt, pausa entre solicitudes y se detiene ante 429 o 5xx.
Solo usa la biblioteca estándar.
"""
import argparse
import csv
import html
import json
import random
import re
import string
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://diccionariochileno.cl"
UA = "Mozilla/5.0 (compatible; UC-NLP-course-project/0.1; academic research)"
DATA = Path(__file__).resolve().parent / "data"
LOG = DATA / "registro_solicitudes.csv"
PAUSA_S = 3.0

ITEM_RE = re.compile(
    r'<li>\s*<a href="(https://diccionariochileno\.cl/term/[^"]*)">(.*?)</a>\s*(?:\((\d+)\))?\s*</li>',
    re.S,
)
SMALL_RE = re.compile(r"<small>Enviado por (.*?)\s*(\d+) (\w+) ago\. votos (-?\d+)</small>", re.S)


class Cliente:
    def __init__(self, pausa=PAUSA_S):
        self.pausa = pausa
        self.ultima = 0.0
        self.robots = urllib.robotparser.RobotFileParser()
        status, robots, _ = self._leer(BASE + "/robots.txt")
        if status != 200:
            sys.exit(f"Detenido: robots.txt respondió {status}")
        self.robots.parse(robots.splitlines())
        DATA.mkdir(exist_ok=True)
        nuevo = not LOG.exists()
        self.log = LOG.open("a", newline="", encoding="utf-8")
        self.w = csv.writer(self.log)
        if nuevo:
            self.w.writerow(["fecha_utc", "url", "status", "segundos", "bytes"])

    def _leer(self, url):
        espera = self.pausa - (time.monotonic() - self.ultima)
        if espera > 0:
            time.sleep(espera)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                cuerpo, status = r.read(), r.status
        except urllib.error.HTTPError as e:
            cuerpo, status = e.read(), e.code
        self.ultima = time.monotonic()
        self.ult_duracion = self.ultima - t0
        return status, cuerpo.decode("utf-8", errors="replace"), len(cuerpo)

    def get(self, url, detener=True):
        if not self.robots.can_fetch(UA, url):
            raise PermissionError(f"robots.txt no permite {url}")
        status, texto, n = self._leer(url)
        self.w.writerow(
            [datetime.now(timezone.utc).isoformat(timespec="seconds"), url, status,
             f"{self.ult_duracion:.3f}", n]
        )
        self.log.flush()
        if detener and (status == 429 or status >= 500):
            sys.exit(f"Detenido: {url} respondió {status}")
        return status, texto


def normalizar(t):
    """Clave para detectar duplicados: sin mayúsculas, tildes ni puntuación."""
    t = unicodedata.normalize("NFKD", t.casefold())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^\w\s]", " ", t)
    return " ".join(t.split())


def parsear_lista(pagina):
    items = []
    for href, nombre, n in ITEM_RE.findall(pagina):
        items.append(
            {
                "termino": " ".join(html.unescape(nombre).split()),
                "url": href,
                # El listado omite el contador cuando hay una sola definición.
                "n_definiciones": int(n) if n else 1,
            }
        )
    return items


def texto_plano(fragmento):
    fragmento = re.sub(r"<br\s*/?>", " ", fragmento)
    return " ".join(html.unescape(re.sub(r"<[^>]+>", "", fragmento)).split())


def parsear_termino(pagina):
    defs = []
    bloques = pagina.split('<div class="definition">')[1:]
    for b in bloques:
        id_def = re.search(r"voto\('(\d+)'", b)
        b = b.split("<button", 1)[0]
        p = re.search(r"</h4>\s*<p>(.*?)</p>", b, re.S)
        ej = re.search(r"<p>\s*<i>(.*?)</i>\s*</p>", b, re.S)
        small = SMALL_RE.search(b)
        defs.append(
            {
                "id": int(id_def.group(1)) if id_def else None,
                "definicion": texto_plano(p.group(1)) if p else "",
                "ejemplo": texto_plano(ej.group(1)) if ej else "",
                "etiquetas": [
                    texto_plano(x)
                    for x in re.findall(r'class="btn btn-mini btn-warning">(.*?)</a>', b, re.S)
                ],
                # No se guarda el seudónimo del autor: no aporta al proyecto.
                "antiguedad": f"{small.group(2)} {small.group(3)}" if small else None,
                "votos": int(small.group(4)) if small else None,
            }
        )
    return defs


def listar(_):
    c = Cliente()
    _, pagina = c.get(BASE + "/terms")
    fecha = datetime.now(timezone.utc).isoformat(timespec="seconds")
    items = parsear_lista(pagina)
    grupos = {}
    for it in items:
        grupos.setdefault(normalizar(it["termino"]), []).append(it)
    with (DATA / "terminos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["termino", "clave_normalizada", "n_definiciones", "n_variantes", "n_palabras",
                    "url", "fuente", "fecha_extraccion_utc"])
        for it in items:
            k = normalizar(it["termino"])
            w.writerow([it["termino"], k, it["n_definiciones"], len(grupos[k]),
                        len(it["termino"].split()), it["url"], BASE + "/terms", fecha])
    print(f"{len(items)} términos, {len(grupos)} claves únicas -> {DATA / 'terminos.csv'}")


def leer_terminos():
    with (DATA / "terminos.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verificar(_):
    c = Cliente()
    completos = {t["url"] for t in leer_terminos()}
    vistos = set()
    print("letra  entradas  fuera_de_/terms")
    for letra in string.ascii_lowercase:
        _, pagina = c.get(f"{BASE}/terms/{letra}")
        urls = {i["url"] for i in parsear_lista(pagina)}
        vistos |= urls
        print(f"{letra:5}  {len(urls):8}  {len(urls - completos):15}")
    faltan = sorted(completos - vistos)
    print(f"\nEn letras: {len(vistos)} | en /terms: {len(completos)}")
    print(f"En letras y no en /terms: {len(vistos - completos)}")
    print(f"En /terms y no en letras: {len(faltan)}")
    for u in faltan[:40]:
        print("  ", urllib.parse.unquote_plus(u.rsplit("/", 1)[1]))


def muestrear(args):
    c = Cliente()
    azar = random.Random(args.semilla)
    muestra = azar.sample(leer_terminos(), args.n)
    salida = DATA / "muestra_definiciones.jsonl"
    with salida.open("w", encoding="utf-8") as f:
        for t in muestra:
            status, pagina = c.get(t["url"])
            f.write(json.dumps(
                {"termino": t["termino"], "url": t["url"], "status": status,
                 "n_en_listado": int(t["n_definiciones"]),
                 "definiciones": parsear_termino(pagina) if status == 200 else []},
                ensure_ascii=False) + "\n")
            f.flush()
    print(f"{args.n} términos -> {salida}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar").set_defaults(fn=listar)
    sub.add_parser("verificar").set_defaults(fn=verificar)
    m = sub.add_parser("muestrear")
    m.add_argument("-n", type=int, default=60)
    m.add_argument("--semilla", type=int, default=42)
    m.set_defaults(fn=muestrear)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
