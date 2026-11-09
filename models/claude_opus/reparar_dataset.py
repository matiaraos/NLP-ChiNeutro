"""Repara el dataset generado por Mistral-7B con agentes de Claude Code (`claude -p`, Opus 5.5).

Por cada lote de expresiones se lanza un agente que recibe, para cada expresión, sus definiciones y
ejemplos del RAG y los pares que generó Mistral. El agente devuelve un par corregido por cada par de
entrada (mismo id), usando exactamente una definición por par. La salida se valida con un JSON Schema
(`--json-schema`) y además en Python; si falla, el lote se reintenta.

Uso (desde la raíz del repo):
    python3 models/claude_opus/reparar_dataset.py --limite-lotes 1          # prueba con un lote
    python3 models/claude_opus/reparar_dataset.py --paralelo 4              # todo (retoma lo ya hecho)
    python3 models/claude_opus/reparar_dataset.py --esperar                 # espera renovaciones de cuota
    python3 models/claude_opus/reparar_dataset.py --solo-cuota              # muestra la cuota usada
    python3 models/claude_opus/reparar_dataset.py --solo-unir               # solo arma el CSV final
    python3 models/claude_opus/reparar_dataset.py --mostrar-prompt          # imprime el prompt del lote 0

Requiere Claude Code >= 2.1.280 para Opus 5.5 (`claude update`), o pasar otro binario con --claude-bin
o la variable de entorno CLAUDE_BIN.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
RUTA_RAG = RAIZ / "scraping/data/rag_documentos.jsonl"
RUTA_MISTRAL = RAIZ / "data/entrenamiento/dataset_chilenismos.csv"
DIR_SALIDA = RAIZ / "data/entrenamiento/opus"
DIR_LOTES = DIR_SALIDA / "lotes"
DIR_ERRORES = DIR_SALIDA / "errores"
RUTA_FINAL = RAIZ / "data/entrenamiento/dataset_chilenismos_opus.csv"

MAX_DEFS = 6  # definiciones por expresión que ve el agente (las de más votos + las que usó Mistral)

SYSTEM_PROMPT = """Eres un lingüista experto en el español de Chile (incluido el habla coloquial, juvenil y popular) y en español neutro. Estás reparando un dataset de entrenamiento para un sistema que traduce frases chilenas a español neutro. Los pares los generó un modelo pequeño y muchos están mal: usan la expresión con un sentido equivocado, no la incluyen, tienen errores gramaticales o la versión neutral cambia el significado.

Recibirás varias expresiones. Para cada una tienes sus definiciones numeradas y ejemplos reales, tomados de un diccionario colaborativo chileno (pueden traer errores de ortografía o definiciones de mala calidad), y los pares generados, cada uno con un id.

Por cada par de entrada devuelve exactamente un par de salida con el mismo id:

1. "def": el número de UNA sola definición. La frase chilena debe ilustrar solo ese sentido, sin mezclar varios. Si la expresión tiene más de una definición, reparte los pares entre los sentidos distintos que sean de uso real, empezando por los más votados. Si dos definiciones dicen lo mismo, trátalas como un solo sentido. No uses definiciones que sean claramente bromas, errores o que no correspondan a un uso real en Chile.
2. "frase_chilena": una frase que un chileno diría de verdad en una conversación, que contenga la expresión (puedes conjugarla, ajustar género y número o usar una variante listada). Debe sonar natural: registro coloquial chileno, sin explicar el significado de la expresión dentro de la misma frase. Puede incluir otros rasgos del habla chilena (po, cachai, voseo chileno como "estai" o "tenís") con moderación. Entre 5 y 25 palabras. No copies los ejemplos del diccionario. Las frases de una misma expresión deben ser situaciones distintas entre sí.
3. "frase_neutralizada": la misma frase en español neutro y estándar, comprensible para cualquier hispanohablante. Reemplaza la expresión y TODOS los demás chilenismos, conjugaciones chilenas y muletillas. Evita también regionalismos de otros países (vosotros, coger por tomar, voseo rioplatense). Conserva el significado, la información, la persona gramatical, el tiempo verbal y la intención. Si la frase chilena es grosera o informal, mantén un registro informal o grosero de uso general en vez de volverla formal.
4. "accion": "corregido" si mantuviste la idea del par original y lo arreglaste, "reescrito" si la idea no servía y escribiste un par nuevo, "descartado" solo si la expresión entera no se puede usar (ninguna definición describe un uso real o es incomprensible); en ese caso deja las frases vacías y usa def 0.
5. "nota": máximo 15 palabras sobre qué estaba mal o qué cambiaste.

Revisa cada par antes de responder: ¿la frase chilena contiene la expresión con el sentido de la definición elegida? ¿un chileno la diría así? ¿la neutral dice exactamente lo mismo y no queda ningún chilenismo? Responde solo con el JSON pedido."""

SCHEMA = {
    "type": "object",
    "properties": {
        "pares": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "accion": {"type": "string", "enum": ["corregido", "reescrito", "descartado"]},
                    "def": {"type": "integer"},
                    "frase_chilena": {"type": "string"},
                    "frase_neutralizada": {"type": "string"},
                    "nota": {"type": "string"},
                },
                "required": ["id", "accion", "def", "frase_chilena", "frase_neutralizada", "nota"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pares"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- datos

def votos(x):
    try:
        return int(x.get("votos") or 0)
    except (TypeError, ValueError):
        return 0


def limpio(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def cargar():
    with open(RUTA_RAG, encoding="utf-8") as f:
        rag = {d["clave"]: d for d in (json.loads(l) for l in f if l.strip())}
    with open(RUTA_MISTRAL, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    por_clave = OrderedDict()
    for i, r in enumerate(filas):
        r["id"] = i
        r["def_id"] = int(float(r["def_id"])) if r.get("def_id") else None
        por_clave.setdefault(r["clave"], []).append(r)
    return rag, por_clave


def definiciones_de(doc, filas):
    """Definiciones que ve el agente: las MAX_DEFS más votadas, más las que Mistral dijo usar."""
    defs = sorted([x for x in doc["definiciones"] if limpio(x.get("definicion"))], key=votos, reverse=True)
    usadas = {r["def_id"] for r in filas}
    elegidas = [x for i, x in enumerate(defs) if i < MAX_DEFS or x.get("id") in usadas]
    return elegidas


def bloque_expresion(doc, filas):
    defs = definiciones_de(doc, filas)
    num = {x.get("id"): i for i, x in enumerate(defs, 1)}
    lineas = [f'### EXPRESIÓN: {doc["termino"]}  (clave: {doc["clave"]})']
    otras = [v for v in doc.get("variantes", []) if v != doc["termino"]]
    if otras:
        lineas.append(f"Variantes: {', '.join(otras)}")
    lineas.append("Definiciones:")
    for i, x in enumerate(defs, 1):
        lineas.append(f"  [{i}] ({votos(x)} votos) {limpio(x['definicion'])[:400]}")
        if limpio(x.get("ejemplo")):
            lineas.append(f"      Ejemplo real: {limpio(x['ejemplo'])[:300]}")
    lineas.append("Pares generados por el modelo pequeño:")
    for r in filas:
        d = num.get(r["def_id"], "?")
        lineas.append(f'  - id {r["id"]} (def {d}): chilena: {limpio(r["frase_chilena"])} || neutral: {limpio(r["frase_neutralizada"])}')
    return "\n".join(lineas), defs


def armar_lotes(rag, por_clave, tam):
    claves = list(por_clave)
    return [claves[i:i + tam] for i in range(0, len(claves), tam)]


def prompt_lote(rag, por_clave, claves):
    bloques = [bloque_expresion(rag[c], por_clave[c])[0] for c in claves]
    n = sum(len(por_clave[c]) for c in claves)
    return (f"Repara los {n} pares de estas {len(claves)} expresiones. Devuelve exactamente un par por cada id.\n\n"
            + "\n\n".join(bloques))


# ---------------------------------------------------------------- validación

def norm(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9ñ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


VACIAS = {"ser", "estar", "andar", "ando", "anda", "estoy", "esta", "tener", "tengo", "hacer", "dar", "echar",
          "el", "la", "los", "las", "un", "una", "de", "del", "al", "a", "en", "que", "se", "te", "me", "le", "lo",
          "y", "o", "pa", "con", "por", "para", "como"}


def contiene_expresion(frase, doc, flexible=True):
    f = norm(frase)
    formas = {norm(x) for x in [doc["clave"], doc["termino"], *doc.get("variantes", [])] if norm(x)}
    if any(f" {x} " in f" {f} " for x in formas):
        return True
    if not flexible:
        return False
    palabras = f.split()
    for forma in formas:
        toks = [t for t in forma.split() if t not in VACIAS and len(t) > 2]
        if toks and all(any(w.startswith(t[:max(4, min(len(t) - 2, 6))]) for w in palabras) for t in toks):
            return True
    return False


def validar(pares, rag, por_clave, claves):
    """Devuelve lista de errores (vacía si el lote está bien)."""
    errores = []
    esperado = {r["id"]: c for c in claves for r in por_clave[c]}
    n_defs = {c: len(definiciones_de(rag[c], por_clave[c])) for c in claves}
    vistos = set()
    for p in pares:
        pid = p.get("id")
        if pid not in esperado:
            errores.append(f"id {pid} no existe en el lote")
            continue
        if pid in vistos:
            errores.append(f"id {pid} repetido")
        vistos.add(pid)
        if p["accion"] == "descartado":
            continue
        ch, ne = limpio(p["frase_chilena"]), limpio(p["frase_neutralizada"])
        if not ch or not ne:
            errores.append(f"id {pid}: frase vacía")
        elif norm(ch) == norm(ne):
            errores.append(f"id {pid}: chilena y neutral iguales")
        if not 1 <= p["def"] <= n_defs[esperado[pid]]:
            errores.append(f"id {pid}: def {p['def']} fuera de rango (1-{n_defs[esperado[pid]]})")
    faltan = set(esperado) - vistos
    if faltan:
        errores.append(f"faltan ids: {sorted(faltan)}")
    return errores


# ---------------------------------------------------------------- cuota del plan

class LimiteUso(Exception):
    """Se alcanzó el límite de uso del plan."""


PARAR = threading.Event()


def hora(ts):
    return time.strftime("%d/%m %H:%M", time.localtime(ts)) if ts else "?"


class Cuota:
    """Uso del plan (fracción 0-1) de las ventanas de 5 horas y semanal.
    Se actualiza con los eventos rate_limit_event que emite cada llamada en modo stream-json."""

    NOMBRES = {"five_hour": "5h", "seven_day": "semana"}

    def __init__(self):
        self.lock = threading.Lock()
        self.ventanas = {}        # nombre -> {"utilization", "resetsAt", "status"}
        self.base = None          # (utilización 5h, resetsAt) al empezar, para estimar consumo por lote
        self.lotes_desde_base = 0

    def actualizar(self, info):
        with self.lock:
            for nombre, v in (info.get("unifiedWindows") or {}).items():
                self.ventanas[nombre] = dict(v)

    def desde_usage(self, texto):
        """Lee el texto de `claude -p /usage` (solo porcentajes; las horas de renovación llegan con la 1.a llamada)."""
        for nombre, patron in [("five_hour", r"Current session:\s*(\d+)%"),
                               ("seven_day", r"Current week \(all models\):\s*(\d+)%")]:
            m = re.search(patron, texto)
            if m:
                self.ventanas.setdefault(nombre, {})["utilization"] = int(m.group(1)) / 100
        return bool(self.ventanas)

    def lote_terminado(self):
        with self.lock:
            v = self.ventanas.get("five_hour", {})
            if self.base is None or self.base[1] != v.get("resetsAt"):    # nueva ventana: reinicia la estimación
                self.base, self.lotes_desde_base = (v.get("utilization", 0.0), v.get("resetsAt")), 0
            else:
                self.lotes_desde_base += 1

    def texto(self, tope_5h=None):
        with self.lock:
            partes = [f"{self.NOMBRES.get(n, n)} {v.get('utilization', 0):.0%} (renueva {hora(v.get('resetsAt'))})"
                      for n, v in self.ventanas.items() if "utilization" in v]
            v = self.ventanas.get("five_hour", {})
            if tope_5h and self.base and self.lotes_desde_base >= 2:
                por_lote = (v.get("utilization", 0) - self.base[0]) / self.lotes_desde_base
                if por_lote > 0:
                    quedan = max(0, int((tope_5h - v.get("utilization", 0)) / por_lote))
                    partes.append(f"~{por_lote:.1%}/lote, ~{quedan} lotes más en esta ventana")
        return "cuota: " + " · ".join(partes) if partes else "cuota: sin datos"

    def excedida(self, args):
        """Devuelve (nombre, resetsAt) de la ventana que superó su tope, o None."""
        topes = {"five_hour": args.tope_5h, "seven_day": args.tope_semana}
        ahora = time.time()
        with self.lock:
            for nombre, tope in topes.items():
                v = self.ventanas.get(nombre)
                if not v:
                    continue
                if v.get("resetsAt") and v["resetsAt"] <= ahora:      # la ventana ya se renovó
                    v["utilization"], v["status"] = 0.0, "allowed"
                    continue
                if v.get("utilization", 0) >= tope or v.get("status") == "rejected":
                    return nombre, v.get("resetsAt")
        return None

    def bloquear(self):
        """Tras un error de límite (429): marca como agotada la ventana más cercana a renovarse."""
        with self.lock:
            for nombre in ("five_hour", "seven_day"):
                v = self.ventanas.get(nombre)
                if v and v.get("status") == "rejected":
                    return
            v = self.ventanas.setdefault("five_hour", {})
            v["utilization"], v["status"] = 1.0, "rejected"
            v.setdefault("resetsAt", None)


CUOTA = Cuota()
ESPERA = threading.Lock()   # un solo hilo espera e imprime; los demás esperan detrás


def hay_cuota(args):
    """True si se puede lanzar otra llamada. Con --esperar duerme hasta que se renueve la ventana."""
    while True:
        if PARAR.is_set():
            return False
        exc = CUOTA.excedida(args)
        if not exc:
            return True
        nombre, reset = exc
        espera = (reset - time.time() + 60) if reset else 15 * 60   # sin hora conocida, reintenta en 15 min
        if not args.esperar or espera > args.max_espera_horas * 3600:
            if not PARAR.is_set():
                PARAR.set()
                print(f"Tope de cuota alcanzado ({CUOTA.NOMBRES[nombre]}); se renueva {hora(reset)}. "
                      f"No se lanzan más lotes.", flush=True)
            return False
        with ESPERA:
            if CUOTA.excedida(args):
                print(f"Tope de cuota ({CUOTA.NOMBRES[nombre]}): esperando hasta {hora(time.time() + espera)}…",
                      flush=True)
                time.sleep(max(espera, 1))


def consultar_usage(args):
    try:
        proc = subprocess.run([args.claude_bin, "-p", "/usage", "--output-format", "json"],
                              capture_output=True, text=True, timeout=120)
        texto = str(json.loads(proc.stdout).get("result") or "")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        print(f"No pude consultar /usage ({e}); la cuota se medirá con la primera llamada.")
        return
    if CUOTA.desde_usage(texto):
        print("Antes de empezar,", CUOTA.texto())
    else:
        print("No pude leer los porcentajes de /usage; la cuota se medirá con la primera llamada.")


# ---------------------------------------------------------------- agente

def llamar_agente(prompt, args):
    cmd = [args.claude_bin, "-p", "--model", args.modelo, "--output-format", "stream-json", "--verbose",
           "--tools", "", "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT,
           "--json-schema", json.dumps(SCHEMA)]
    if args.effort:
        cmd += ["--effort", args.effort]
    with tempfile.TemporaryDirectory() as cwd:  # carpeta vacía: el agente no carga contexto del repo
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=args.timeout)
    salida = None
    for linea in proc.stdout.splitlines():
        try:
            ev = json.loads(linea)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "rate_limit_event":
            CUOTA.actualizar(ev.get("rate_limit_info") or {})
        elif ev.get("type") == "result":
            salida = ev
    if salida is None:
        raise RuntimeError(f"sin evento result (código {proc.returncode}): {proc.stdout[-500:]} {proc.stderr[-500:]}")
    if salida.get("is_error"):
        msg = str(salida.get("result"))
        if salida.get("api_error_status") == 429 or re.search(r"usage limit|rate limit|limit reached|out of extra usage",
                                                             msg, re.I):
            raise LimiteUso(msg[:300])
        raise RuntimeError(f"error del agente: {msg[:800]}")
    if not isinstance(salida.get("structured_output"), dict):
        raise RuntimeError(f"el agente no devolvió structured_output: {str(salida.get('result'))[:800]}")
    return salida


def procesar_lote(i, claves, rag, por_clave, args):
    ruta = DIR_LOTES / f"lote_{i:04d}.json"
    prompt = prompt_lote(rag, por_clave, claves)
    ultimo_error, intento = "", 0
    while intento <= args.reintentos:
        if not hay_cuota(args):
            return i, None, 0.0, "omitido por cuota"
        intento += 1
        t0 = time.time()
        texto_prompt = prompt
        if ultimo_error:
            texto_prompt += f"\n\nATENCIÓN: un intento anterior falló la validación: {ultimo_error}. Corrígelo."
        try:
            salida = llamar_agente(texto_prompt, args)
            pares = salida["structured_output"]["pares"]
            errores = validar(pares, rag, por_clave, claves)
            if errores:
                raise RuntimeError("; ".join(errores[:10]))
        except LimiteUso as e:
            CUOTA.bloquear()
            intento -= 1                     # no cuenta como intento: se reintenta cuando haya cuota
            print(f"lote {i:04d}: límite de uso del plan ({e})", flush=True)
            continue
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            ultimo_error = str(e)[:600]
            DIR_ERRORES.mkdir(parents=True, exist_ok=True)
            with open(DIR_ERRORES / f"lote_{i:04d}.log", "a", encoding="utf-8") as f:
                f.write(f"--- intento {intento} ({time.strftime('%H:%M:%S')}): {ultimo_error}\n")
            continue
        ruta.write_text(json.dumps({
            "lote": i, "claves": claves, "modelo": args.modelo, "effort": args.effort, "intentos": intento,
            "costo_usd": salida.get("total_cost_usd"), "duracion_s": round(time.time() - t0, 1),
            "pares": pares,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        CUOTA.lote_terminado()
        return i, True, salida.get("total_cost_usd") or 0.0, ""
    return i, False, 0.0, ultimo_error


# ---------------------------------------------------------------- unión

def unir(rag, por_clave):
    filas_id = {r["id"]: r for fs in por_clave.values() for r in fs}
    salida, descartados, lotes, costo = [], 0, 0, 0.0
    for ruta in sorted(DIR_LOTES.glob("lote_*.json")):
        data = json.loads(ruta.read_text(encoding="utf-8"))
        lotes += 1
        costo += data.get("costo_usd") or 0.0
        for p in data["pares"]:
            orig = filas_id[p["id"]]
            if p["accion"] == "descartado":
                descartados += 1
                continue
            doc = rag[orig["clave"]]
            defs = definiciones_de(doc, por_clave[orig["clave"]])
            d = defs[p["def"] - 1]
            salida.append({
                "id_mistral": p["id"], "clave": orig["clave"], "termino": orig["termino"],
                "def_id": d.get("id"), "definicion": limpio(d["definicion"]),
                "frase_chilena": limpio(p["frase_chilena"]), "frase_neutralizada": limpio(p["frase_neutralizada"]),
                "accion": p["accion"], "nota": limpio(p["nota"]),
                "contiene_clave": contiene_expresion(p["frase_chilena"], doc),
                "neutral_sin_clave": not contiene_expresion(p["frase_neutralizada"], doc, flexible=False),
                "frase_chilena_mistral": orig["frase_chilena"], "frase_neutralizada_mistral": orig["frase_neutralizada"],
                "modelo": data["modelo"],
            })
    if not salida:
        print("No hay lotes terminados todavía.")
        return
    with open(RUTA_FINAL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(salida[0]))
        w.writeheader()
        w.writerows(salida)
    acciones = {}
    for s in salida:
        acciones[s["accion"]] = acciones.get(s["accion"], 0) + 1
    print(f"Lotes: {lotes} | pares: {len(salida)} | descartados: {descartados} | acciones: {acciones}")
    print(f"contiene_clave: {sum(s['contiene_clave'] for s in salida) / len(salida):.1%} | "
          f"neutral_sin_clave: {sum(s['neutral_sin_clave'] for s in salida) / len(salida):.1%}")
    print(f"Equivalente en API (con suscripción no se cobra aparte): US${costo:.2f} -> {RUTA_FINAL}")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lote", type=int, default=8, help="expresiones por agente (default 8, ~40 pares)")
    ap.add_argument("--paralelo", type=int, default=2, help="agentes simultáneos")
    ap.add_argument("--limite-lotes", type=int, default=None, help="procesar solo N lotes pendientes")
    ap.add_argument("--modelo", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high", help="low|medium|high|xhigh|max ('' para no pasarlo)")
    ap.add_argument("--reintentos", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900, help="segundos por llamada")
    ap.add_argument("--claude-bin", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--tope-5h", type=float, default=0.95,
                    help="no lanzar lotes si la ventana de 5 h supera esta fracción (default 0.95)")
    ap.add_argument("--tope-semana", type=float, default=0.95,
                    help="no lanzar lotes si la ventana semanal supera esta fracción (default 0.95)")
    ap.add_argument("--esperar", action="store_true",
                    help="al llegar a un tope, esperar a que se renueve la ventana y seguir")
    ap.add_argument("--max-espera-horas", type=float, default=6,
                    help="con --esperar, espera máxima; si la renovación es más tarde, se detiene (default 6)")
    ap.add_argument("--solo-cuota", action="store_true", help="solo muestra la cuota usada y termina")
    ap.add_argument("--solo-unir", action="store_true")
    ap.add_argument("--mostrar-prompt", action="store_true")
    args = ap.parse_args()

    if args.solo_cuota:
        consultar_usage(args)
        return
    rag, por_clave = cargar()
    lotes = armar_lotes(rag, por_clave, args.lote)
    if args.mostrar_prompt:
        print(SYSTEM_PROMPT, "\n\n=====\n")
        print(prompt_lote(rag, por_clave, lotes[0]))
        return
    if args.solo_unir:
        unir(rag, por_clave)
        return

    DIR_LOTES.mkdir(parents=True, exist_ok=True)
    # al retomar se identifica cada lote por su número; no cambies --lote a mitad de una corrida
    meta = DIR_SALIDA / "config.json"
    if meta.exists() and json.loads(meta.read_text())["lote"] != args.lote:
        sys.exit(f"Ya hay una corrida con --lote {json.loads(meta.read_text())['lote']}; usa ese valor o borra {DIR_SALIDA}")
    meta.write_text(json.dumps({"lote": args.lote, "modelo": args.modelo}))

    pendientes = [(i, c) for i, c in enumerate(lotes) if not (DIR_LOTES / f"lote_{i:04d}.json").exists()]
    if args.limite_lotes:
        pendientes = pendientes[:args.limite_lotes]
    print(f"Lotes: {len(lotes)} en total, {len(pendientes)} por procesar "
          f"({args.lote} expresiones c/u, {args.paralelo} en paralelo, {args.modelo}, effort={args.effort or 'default'})")
    print(f"Topes: 5h {args.tope_5h:.0%}, semana {args.tope_semana:.0%}"
          f"{' (esperando renovaciones)' if args.esperar else ''}")
    consultar_usage(args)

    # Con suscripción (Pro/Max) no se cobra por llamada: el costo es el equivalente en API que reporta claude -p.
    hechos, costo, fallidos, omitidos = 0, 0.0, [], 0
    with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
        futuros = [ex.submit(procesar_lote, i, c, rag, por_clave, args) for i, c in pendientes]
        for fut in as_completed(futuros):
            i, ok, c, err = fut.result()
            if ok is None:
                omitidos += 1
                continue
            hechos += 1
            costo += c
            if not ok:
                fallidos.append(i)
            estado = f"ok (equiv. API US${c:.2f})" if ok else f"FALLÓ: {err[:150]}"
            print(f"[{hechos}/{len(pendientes)}] lote {i:04d} {estado} | {CUOTA.texto(args.tope_5h)}", flush=True)
    if omitidos:
        print(f"Por cuota quedaron {omitidos} lotes pendientes ({CUOTA.texto()}). "
              f"Vuelve a correr el mismo comando cuando se renueve y retoma desde ahí (o usa --esperar).")
    if fallidos:
        print(f"Lotes fallidos ({len(fallidos)}): {fallidos}. Vuelve a correr el script para reintentarlos; "
              f"detalle en {DIR_ERRORES}")
    unir(rag, por_clave)


if __name__ == "__main__":
    main()
