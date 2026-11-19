"""Muestra de frases de Reddit Chile con expresiones del diccionario, para la evaluación fuera de dominio.

Fuente: API pública de Arctic Shift (archivo de Reddit, https://github.com/ArthurHeitmann/arctic_shift).
Su búsqueda de texto se corta por tiempo, así que se hace un escaneo estocástico: se eligen instantes al azar
y se piden los 100 comentarios siguientes del subreddit (listado por fecha, consulta barata). Después, offline,
se buscan las formas del diccionario (`scraping/data/rag_documentos.jsonl`) y se extraen las oraciones.

Uso, desde la raíz del repo:
    python3 scraping/reddit_muestreo.py recolectar --solicitudes 2500      # reanudable; ~1 solicitud/s
    python3 scraping/reddit_muestreo.py extraer --objetivo 2000 --tope 3

Salidas:
    salidas/reddit/crudo.jsonl                 comentarios descargados (no se versiona)
    data/evaluacion/reddit_chile.json          frases seleccionadas
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "models/gemma4"))
import comun  # noqa: E402

API = "https://arctic-shift.photon-reddit.com/api/comments/search"
CAMPOS = "id,created_utc,author,score,link_id,parent_id,subreddit,body"
# Peso de cada subreddit en el muestreo e instante desde el que se muestrea (antes hay muy poca actividad)
SUBREDDITS = {"chile": (0.7, "2014-01-01"), "RepublicadeChile": (0.3, "2020-06-01")}
CRUDO = RAIZ / "salidas/reddit/crudo.jsonl"
SALIDA = RAIZ / "data/evaluacion/reddit_chile.json"


def ts(fecha):
    return int(datetime.fromisoformat(fecha).replace(tzinfo=timezone.utc).timestamp())


def recolectar(a):
    import requests
    CRUDO.parent.mkdir(parents=True, exist_ok=True)
    vistos = set()
    if CRUDO.exists():
        with open(CRUDO, encoding="utf-8") as f:
            vistos = {json.loads(l)["id"] for l in f if l.strip()}
    registro = CRUDO.with_name("solicitudes.log")
    hechas = sum(1 for _ in open(registro)) if registro.exists() else 0
    rng = random.Random(a.semilla + hechas)
    fin = int(time.time()) - 7 * 86400
    subs, pesos = list(SUBREDDITS), [p for p, _ in SUBREDDITS.values()]
    s = requests.Session()
    s.headers["User-Agent"] = "chineutro-eval/0.1 (investigación académica)"
    print(f"Retomando: {hechas} solicitudes hechas, {len(vistos):,} comentarios", flush=True)
    errores = 0
    with open(CRUDO, "a", encoding="utf-8") as out, open(registro, "a") as log:
        while hechas < a.solicitudes:
            sub = rng.choices(subs, pesos)[0]
            desde = rng.randint(ts(SUBREDDITS[sub][1]), fin)
            try:
                r = s.get(API, params={"subreddit": sub, "after": desde, "sort": "asc", "limit": 100,
                                       "fields": CAMPOS}, timeout=60)
                d = r.json()
                if r.status_code != 200 or d.get("error"):
                    raise RuntimeError(d.get("error") or r.status_code)
            except Exception as e:  # noqa: BLE001
                errores += 1
                espera = min(300, 10 * 2 ** min(errores, 5))
                print(f"error ({e}); espero {espera}s", flush=True)
                time.sleep(espera)
                continue
            errores = 0
            nuevos = [c for c in d["data"] or [] if c["id"] not in vistos]
            for c in nuevos:
                vistos.add(c["id"])
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
            out.flush()
            hechas += 1
            log.write(f"{sub}\t{desde}\t{len(nuevos)}\n")
            log.flush()
            if hechas % 50 == 0:
                print(f"{hechas}/{a.solicitudes} solicitudes, {len(vistos):,} comentarios", flush=True)
            time.sleep(a.pausa)
    print(f"Listo: {len(vistos):,} comentarios en {CRUDO}", flush=True)


# ---------- extracción ----------

BOTS = re.compile(r"(^automoderator$|bot$|^\[deleted\]$)", re.I)
URL = re.compile(r"https?://|www\.|\]\(")
MENCION = re.compile(r"(?<![\w/])/?u/[\w-]+", re.I)
ARROBA = re.compile(r"(?<!\w)@\w+")
CORREO = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
TELEFONO = re.compile(r"(?:\+?56\s?)?(?:9\s?)?\d{4}\s?\d{4}\b")
RUT = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[\dkK]\b")
ORACION = re.compile(r"(?<=[.!?…])\s+|\n+")
FUNCIONALES = set("de la que el en y a los se no un por con las una para es lo como pero más mas me si "
                  "ya o le al mi te yo eso esta este hay porque cuando todo nada así asi tu tú ser".split())


def limpiar(texto):
    texto = re.sub(r"^\s*>.*$", " ", texto, flags=re.M)                 # citas de otros comentarios
    texto = re.sub(r"[*_~`#^|]+", "", texto)                            # marcado de markdown
    texto = texto.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<").replace("&nbsp;", " ")
    texto = CORREO.sub("[correo]", texto)
    texto = RUT.sub("[rut]", texto)
    texto = TELEFONO.sub("[teléfono]", texto)
    texto = MENCION.sub("[usuario]", texto)
    texto = ARROBA.sub("[usuario]", texto)
    return texto


def ngramas(tokens, n_max):
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            yield " ".join(tokens[i:i + n])


def oracion_valida(o, min_p, max_p):
    if URL.search(o):
        return False
    palabras = o.split()
    if not min_p <= len(palabras) <= max_p:
        return False
    letras = sum(ch.isalpha() for ch in o)
    if letras < 0.6 * len(o.replace(" ", "")):
        return False
    if sum(ch.isupper() for ch in o) > 0.5 * max(letras, 1):              # gritos
        return False
    toks = comun.norm(o).split()
    return any(t in FUNCIONALES for t in toks)                           # descarta inglés y listas sueltas


def extraer(a):
    dic = comun.cargar_diccionario(RAIZ)
    por_forma = defaultdict(list)
    for fila in dic.itertuples():
        por_forma[fila.forma].append((fila.clave, fila.termino))
    n_max = max(len(f.split()) for f in por_forma)
    print(f"Diccionario: {dic.clave.nunique()} expresiones, {len(por_forma)} formas")

    candidatas, vistas = [], set()
    comentarios = descartados = 0
    with open(CRUDO, encoding="utf-8") as f:
        for linea in f:
            c = json.loads(linea)
            comentarios += 1
            cuerpo = c.get("body") or ""
            if BOTS.search(c.get("author") or "") or cuerpo.strip() in ("[deleted]", "[removed]", ""):
                descartados += 1
                continue
            contexto = re.sub(r"\s+", " ", limpiar(cuerpo)).strip()
            for o in ORACION.split(limpiar(cuerpo)):
                o = re.sub(r"\s+", " ", o).strip(" -•")
                toks = comun.norm(o).split()
                hallazgos = {}
                for g in ngramas(toks, n_max):
                    for clave, termino in por_forma.get(g, ()):
                        hallazgos.setdefault(clave, {"clave": clave, "termino": termino, "forma": g})
                if not hallazgos or not oracion_valida(o, a.min_palabras, a.max_palabras):
                    continue
                llave = " ".join(toks)
                if llave in vistas:
                    continue
                vistas.add(llave)
                candidatas.append({
                    "id": f"{c['id']}_{len(candidatas)}", "comentario_id": c["id"], "subreddit": c["subreddit"],
                    "fecha": datetime.fromtimestamp(int(c["created_utc"]), timezone.utc).date().isoformat(),
                    "score": c.get("score"), "frase": o, "expresiones": list(hallazgos.values()),
                    "contexto": contexto[:1200],
                })
    print(f"Comentarios: {comentarios:,} ({descartados:,} descartados por bot/borrado); "
          f"oraciones candidatas: {len(candidatas):,}")

    # Selección balanceada: rondas de una frase por expresión, priorizando las de varias palabras y
    # las frases con buen score, hasta el objetivo; cada expresión aporta como máximo `tope` frases.
    rng = random.Random(a.semilla)
    rng.shuffle(candidatas)
    por_clave = defaultdict(list)
    for cand in candidatas:
        for e in cand["expresiones"]:
            por_clave[e["clave"]].append(cand)
    for lista in por_clave.values():
        lista.sort(key=lambda x: -(x["score"] or 0) if (x["score"] or 0) > 0 else 1)
    claves = sorted(por_clave, key=lambda k: (-len(k.split()), len(por_clave[k]), k))
    elegidas, ids, usos = [], set(), Counter()
    for _ in range(a.tope):
        for k in claves:
            if len(elegidas) >= a.objetivo:
                break
            while por_clave[k]:
                cand = por_clave[k].pop(0)
                if cand["id"] not in ids:
                    ids.add(cand["id"])
                    elegidas.append(cand)
                    usos[k] += 1
                    break
    for i, e in enumerate(elegidas):
        e["id"] = f"reddit_{i:05d}"

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    meta = {"fuente": "Reddit (r/chile, r/RepublicadeChile) vía API de Arctic Shift, muestreo aleatorio por fecha",
            "fecha_extraccion": datetime.now(timezone.utc).date().isoformat(),
            "comentarios_leidos": comentarios, "oraciones_candidatas": len(candidatas),
            "expresiones_cubiertas": len(usos), "parametros": vars(a) | {"cmd": None}}
    SALIDA.write_text(json.dumps({"meta": meta, "frases": elegidas}, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(f"Elegidas {len(elegidas):,} frases, {len(usos)} expresiones distintas "
          f"({sum(len(k.split()) > 1 for k in usos)} de varias palabras) → {SALIDA}")
    print("Por subreddit:", dict(Counter(e["subreddit"] for e in elegidas)))
    print("Por año:", dict(sorted(Counter(e["fecha"][:4] for e in elegidas).items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recolectar")
    r.add_argument("--solicitudes", type=int, default=2500)
    r.add_argument("--pausa", type=float, default=1.0, help="segundos entre solicitudes")
    r.add_argument("--semilla", type=int, default=0)
    e = sub.add_parser("extraer")
    e.add_argument("--objetivo", type=int, default=2000)
    e.add_argument("--tope", type=int, default=3, help="máximo de frases por expresión")
    e.add_argument("--min-palabras", type=int, default=5)
    e.add_argument("--max-palabras", type=int, default=40)
    e.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()
    recolectar(a) if a.cmd == "recolectar" else extraer(a)


if __name__ == "__main__":
    main()
