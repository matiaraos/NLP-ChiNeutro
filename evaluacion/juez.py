"""LLM-as-a-judge con Claude Opus 5.5 a través de Claude Code (`claude -p`), igual que models/claude_opus.

Corre en el PC (donde Claude Code tiene la sesión iniciada), no en Colab: lee un CSV de predicciones y no necesita
el modelo evaluado. Cada llamada evalúa un lote de traducciones con una rúbrica fija y devuelve JSON validado con
`--json-schema`.

- Con referencia (splits del dataset): el juez ve la frase chilena, la expresión y su definición, la referencia
  y la traducción. La referencia es *una* traducción válida, no la única.
- Sin referencia (Reddit): solo la frase chilena y la traducción.

Los veredictos se guardan en un caché (evaluacion/cache_juez.jsonl) indexado por el contenido evaluado: repetir una
evaluación, o evaluar otro modelo que produjo la misma traducción, no gasta cuota. Si se corta (cuota, error),
volver a correr retoma lo que falta.

Ojo con el sesgo: Opus reescribió las referencias del dataset, así que puede preferir traducciones con su estilo.
Por eso el plan calibra el juez contra evaluación humana en una muestra.
"""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .datos import completar

VERSION_RUBRICA = 1          # súbela si cambias SYSTEM_PROMPT o SCHEMA: invalida el caché
MODELO_JUEZ = "claude-opus-5-5"
RUTA_CACHE = Path(__file__).with_name("cache_juez.jsonl")
JERGA = ("eliminada", "parcial", "retenida")

SYSTEM_PROMPT = """Eres un lingüista experto en el español de Chile (incluido el habla coloquial, juvenil y popular) y en español neutro. Evalúas la salida de un sistema que traduce frases chilenas a español neutro, es decir, a un español que entienda cualquier hispanohablante, sin chilenismos ni modismos locales.

Para cada caso recibes la frase chilena, la traducción del sistema y, cuando existen, la expresión chilena principal con su definición y una traducción de referencia. La referencia es una traducción correcta, pero no la única: no penalices reformulaciones distintas que sean correctas.

Califica cada traducción, de forma independiente, con estos criterios:

- sentido (1 a 5): ¿conserva el significado completo de la frase chilena, incluida la expresión según su definición?
  5 = mismo significado; 4 = diferencia menor de matiz; 3 = se pierde o cambia una parte relevante; 2 = el sentido principal está mal; 1 = sin relación, vacía, o el sistema no tradujo (respondió, explicó, comentó o repitió la instrucción).
- jerga: "eliminada" si no queda ningún chilenismo; "parcial" si se tradujo la expresión principal pero queda otro chilenismo (por ejemplo «po», «cachai», «al tiro», «la pega», voseo chileno como «estái» o «sabís», o artículo ante nombre propio como «el Martín»); "retenida" si la expresión principal sigue en la traducción. Palabras de uso general en el español no cuentan como jerga.
- fluidez (1 a 5): ¿es español natural y gramatical, independiente del sentido? 5 = como lo escribiría un nativo; 3 = comprensible pero forzado; 1 = agramatical o ilegible.
- aceptable: true solo si un editor la publicaría tal cual: conserva el sentido, no deja jerga chilena y es correcta.
- comentario: el problema principal en 20 palabras como máximo, o "" si no hay.

Devuelve exactamente una evaluación por caso, con su mismo id."""

SCHEMA = {
    "type": "object",
    "properties": {"evaluaciones": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "sentido": {"type": "integer", "minimum": 1, "maximum": 5},
            "jerga": {"type": "string", "enum": list(JERGA)},
            "fluidez": {"type": "integer", "minimum": 1, "maximum": 5},
            "aceptable": {"type": "boolean"},
            "comentario": {"type": "string"},
        },
        "required": ["id", "sentido", "jerga", "fluidez", "aceptable", "comentario"],
        "additionalProperties": False,
    }}},
    "required": ["evaluaciones"],
    "additionalProperties": False,
}


class LimiteUso(RuntimeError):
    """Error que no se arregla reintentando (cuota agotada, sin sesión): detiene todos los lotes."""


def _texto(x):
    return "" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x).strip()


def caso(fila):
    """Lo que ve el juez de una fila (sin el nombre del modelo evaluado)."""
    c = {"frase_chilena": _texto(fila.get("frase_chilena"))}
    expresion = _texto(fila.get("termino")) or _texto(fila.get("clave"))
    if expresion:
        c["expresion"] = expresion
    if _texto(fila.get("definicion")):
        c["definicion"] = _texto(fila.get("definicion"))
    if _texto(fila.get("frase_neutralizada")):
        c["referencia"] = _texto(fila.get("frase_neutralizada"))
    c["traduccion"] = _texto(fila.get("prediccion"))
    return c


def llave(c, modelo):
    datos = json.dumps([VERSION_RUBRICA, modelo, c], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(datos.encode("utf-8")).hexdigest()[:24]


class Cache:
    def __init__(self, ruta=RUTA_CACHE):
        self.ruta = ruta
        self.lock = threading.Lock()
        self.datos = {}
        if ruta.exists():
            with open(ruta, encoding="utf-8") as f:
                for linea in f:
                    if linea.strip():
                        d = json.loads(linea)
                        self.datos[d["llave"]] = d["veredicto"]

    def get(self, k):
        return self.datos.get(k)

    def guardar(self, nuevos):
        with self.lock:
            self.ruta.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ruta, "a", encoding="utf-8") as f:
                for k, v in nuevos.items():
                    self.datos[k] = v
                    f.write(json.dumps({"llave": k, "veredicto": v}, ensure_ascii=False) + "\n")


def llamar_claude(prompt, claude_bin, modelo, effort, timeout):
    cmd = [claude_bin, "-p", "--model", modelo, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--system-prompt", SYSTEM_PROMPT, "--json-schema", json.dumps(SCHEMA)]
    if effort:
        cmd += ["--effort", effort]
    with tempfile.TemporaryDirectory() as cwd:   # carpeta vacía: no carga contexto del repo
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    try:
        salida = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"salida no JSON (código {proc.returncode}): {proc.stdout[-400:]} {proc.stderr[-400:]}")
    if salida.get("is_error"):
        msg = str(salida.get("result"))
        if salida.get("api_error_status") == 429 or re.search(r"usage limit|rate limit|limit reached", msg, re.I):
            raise LimiteUso(msg[:300])
        if re.search(r"not logged in|/login|invalid api key|authentication", msg, re.I):
            raise LimiteUso(f"Claude Code sin sesión: corre `claude` y luego /login ({msg[:200]})")
        raise RuntimeError(f"error de claude: {msg[:600]}")
    out = salida.get("structured_output")
    if not isinstance(out, dict):
        raise RuntimeError(f"sin structured_output: {str(salida.get('result'))[:600]}")
    return out["evaluaciones"], salida.get("total_cost_usd") or 0.0


def juzgar_lote(casos, claude_bin, modelo, effort, timeout, reintentos):
    """casos: {id: caso}. Devuelve ({id: veredicto}, costo)."""
    prompt = ("Evalúa estas traducciones (JSON, una por caso):\n\n"
              + json.dumps([{"id": i, **c} for i, c in casos.items()], ensure_ascii=False, indent=1))
    error = ""
    for _ in range(reintentos + 1):
        try:
            evals, costo = llamar_claude(prompt + (f"\n\nUn intento anterior falló: {error}" if error else ""),
                                         claude_bin, modelo, effort, timeout)
            por_id = {e["id"]: {k: e[k] for k in ("sentido", "jerga", "fluidez", "aceptable", "comentario")}
                      for e in evals}
            if set(por_id) != set(casos):
                raise RuntimeError(f"ids distintos: faltan {sorted(set(casos) - set(por_id))[:5]}, "
                                   f"sobran {sorted(set(por_id) - set(casos))[:5]}")
            return por_id, costo
        except LimiteUso:
            raise
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            error = str(e)[:300]
    raise RuntimeError(f"lote fallido tras {reintentos + 1} intentos: {error}")


def juzgar(df, modelo=MODELO_JUEZ, effort="medium", lote=20, paralelo=3, muestra=None, semilla=42,
           claude_bin=None, timeout=900, reintentos=2, ruta_cache=RUTA_CACHE):
    """Evalúa cada fila con el juez. Devuelve (resumen: dict, veredictos: DataFrame alineado con df o la muestra).

    muestra: evaluar solo N filas al azar (con `semilla`), para ahorrar cuota.
    """
    claude_bin = claude_bin or os.environ.get("CLAUDE_BIN", "claude")
    df = completar(df).reset_index(drop=True)
    if muestra and muestra < len(df):
        df = df.sample(muestra, random_state=semilla).sort_index()
    cache = Cache(ruta_cache)
    casos = {i: caso(f) for i, f in df.iterrows()}
    llaves = {i: llave(c, modelo) for i, c in casos.items()}
    vacias = {llaves[i] for i, c in casos.items() if not c["traduccion"]}
    cache.datos.update({k: {"sentido": 1, "jerga": None, "fluidez": 1, "aceptable": False,
                            "comentario": "salida vacía"} for k in vacias})
    pendientes = {}
    for i, c in casos.items():   # un caso repetido (misma llave) se evalúa una vez
        if cache.get(llaves[i]) is None:
            pendientes.setdefault(llaves[i], c)
    lotes = [dict(list(pendientes.items())[j:j + lote]) for j in range(0, len(pendientes), lote)]
    print(f"Juez {modelo} (effort={effort}): {len(df)} casos, {len(df) - len(pendientes)} en caché, "
          f"{len(pendientes)} por evaluar en {len(lotes)} lotes de hasta {lote}")
    costo, errores, t0 = 0.0, [], time.time()
    parar = threading.Event()

    def trabajar(casos_lote):
        if parar.is_set():
            return None
        try:
            v, c = juzgar_lote(casos_lote, claude_bin, modelo, effort, timeout, reintentos)
        except LimiteUso:
            parar.set()
            raise
        cache.guardar(v)
        return c

    with ThreadPoolExecutor(max_workers=paralelo) as ex:
        futuros = [ex.submit(trabajar, l) for l in lotes]
        for n, fut in enumerate(as_completed(futuros), 1):
            try:
                c = fut.result()
                costo += c or 0.0
            except LimiteUso as e:
                errores.append(f"detenido: {e}")
            except Exception as e:   # noqa: BLE001 - se informa y se sigue con los demás lotes
                errores.append(str(e)[:300])
            print(f"  lote {n}/{len(lotes)} | {time.time() - t0:.0f}s | equivalente API US${costo:.2f}", flush=True)
    for e in errores[:5]:
        print("  ERROR:", e)

    filas = [{"_i": i, **(cache.get(llaves[i]) or {})} for i in casos]
    ver = pd.DataFrame(filas).set_index("_i").reindex(columns=["sentido", "jerga", "fluidez", "aceptable",
                                                              "comentario"])
    ver.index.name = None
    hechos = ver.sentido.notna()
    con_jerga = ver.jerga.notna()
    resumen = {
        "juez_modelo": modelo, "juez_effort": effort, "juez_rubrica": VERSION_RUBRICA,
        "juez_n": int(hechos.sum()), "juez_faltan": int((~hechos).sum()),
        "juez_sentido_media": round(float(ver.sentido[hechos].mean()), 3) if hechos.any() else None,
        "juez_fluidez_media": round(float(ver.fluidez[hechos].mean()), 3) if hechos.any() else None,
        "juez_jerga_eliminada_%": round(100 * float((ver.jerga[con_jerga] == "eliminada").mean()), 1)
        if con_jerga.any() else None,
        "juez_aceptable_%": round(100 * float(ver.aceptable[hechos].astype(bool).mean()), 1) if hechos.any() else None,
    }
    if resumen["juez_faltan"]:
        print(f"Faltan {resumen['juez_faltan']} casos sin veredicto: vuelve a correr para completarlos.")
    return resumen, df.join(ver)
