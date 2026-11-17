"""Utilidades compartidas por los notebooks de Gemma 4 E4B.

- 00_explorar_corpus.ipynb: índice y muestra del Chilean Spanish Corpus.
- 01_paso1_ntp_corpus.ipynb: paso 1, predicción del siguiente token (LoRA solo en la FFN) sobre el corpus.
- 02_paso2_traduccion.ipynb: paso 2, ajuste a la traducción chileno -> español neutro.

El módulo no importa torch ni transformers al cargarse, para que la exploración corra en un entorno sin GPU.
"""
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATASET_CORPUS = "jorgeortizfuentes/chilean-spanish-corpus"
MODELO_BASE = "google/gemma-4-E4B"   # base (pre-entrenado), no la variante -it
SEED = 42

RUTA_RAG = "scraping/data/rag_documentos.jsonl"
RUTA_PARES = "data/entrenamiento/dataset_chilenismos_opus.csv"
RUTA_SPLIT = "data/entrenamiento/splits/split_oficial.csv"   # split oficial del grupo (crear_split_oficial.py)


def ruta_dato(raiz, relativa):
    """Ubica un archivo de datos: con la estructura del repo (raiz/data/...) o suelto en la carpeta (raiz/nombre),
    que es como queda al subir los archivos a mano a Colab o Drive."""
    raiz = Path(raiz)
    for candidata in (raiz / relativa, raiz / Path(relativa).name):
        if candidata.is_file():
            return candidata
    raise FileNotFoundError(f"No encontré {Path(relativa).name} en {raiz} (ni en {raiz / Path(relativa).parent})")


# ---------------------------------------------------------------------------
# Repo del Hugging Face Hub: datos, checkpoints y resultados
# ---------------------------------------------------------------------------

def con_limite(funcion, segundos, descripcion):
    """Corre funcion() en un hilo aparte y espera a lo más `segundos`. Si el Hub limita los commits (429),
    huggingface_hub espera hasta 1 h antes de reintentar: con esto el entrenamiento no queda bloqueado esa hora.
    El hilo es daemon, así que no impide que el proceso termine. Devuelve True si terminó a tiempo y sin error."""
    import threading
    resultado = {}

    def correr():
        try:
            funcion()
            resultado["ok"] = True
        except Exception as e:
            resultado["error"] = e

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(segundos)
    if hilo.is_alive():
        print(f"⚠ {descripcion}: no terminó en {segundos} s (¿límite de commits del Hub?); se sigue sin esperar")
        return False
    if "error" in resultado:
        print(f"⚠ {descripcion}: {resultado['error']}")
        return False
    return True


class Hub:
    """Espejo entre DIR_TRABAJO y un repo privado del Hub: un archivo en DIR_TRABAJO/x/y queda en x/y del repo.

    Así, una sesión nueva de Colab baja lo que necesita (índice, corpus, adaptador del paso 1, último checkpoint)
    y cada notebook sube lo que produce. Con repo=None no hace nada (pruebas locales sin red)."""

    def __init__(self, repo, dir_trabajo, tipo="model"):
        self.repo, self.tipo, self.dir = repo, tipo, Path(dir_trabajo).resolve()
        self.api = None
        if repo:
            from huggingface_hub import HfApi
            self.api = HfApi()
            # Falla aquí si el token es de solo lectura: mejor saberlo antes de entrenar horas sin poder guardar
            self.api.create_repo(repo, repo_type=tipo, private=True, exist_ok=True)

    def rel(self, ruta):
        """Ruta dentro del repo: relativa a DIR_TRABAJO."""
        ruta = Path(ruta)
        return (ruta.resolve().relative_to(self.dir) if ruta.is_absolute() else ruta).as_posix()

    def bajar(self, *rutas):
        """Baja del repo archivos o carpetas (rutas de DIR_TRABAJO; admite comodines). Lo que no está se ignora."""
        if not self.api:
            return
        from huggingface_hub import snapshot_download
        patrones = [p for r in rutas for p in (self.rel(r), f"{self.rel(r)}/*")]
        snapshot_download(self.repo, repo_type=self.tipo, allow_patterns=patrones, local_dir=self.dir)

    def subir(self, ruta, destino=None, reemplazar=False, ignorar=None, mensaje=None, limite_s=1200):
        """Sube un archivo o carpeta a su misma ruta en el repo (o a `destino`). reemplazar=True borra del repo lo
        que ya no está en la carpeta. Un fallo de red o una subida que tarda más de `limite_s` (p. ej. por el límite
        de commits del Hub) avisa y no corta ni bloquea el entrenamiento."""
        if not self.api:
            return
        ruta = Path(ruta)
        destino = destino or self.rel(ruta)
        if con_limite(lambda: self._subir(ruta, destino, reemplazar, ignorar, mensaje), limite_s,
                      f"No pude subir {destino} al Hub"):
            print(f"↑ Hub: {destino}")

    def _subir(self, ruta, destino, reemplazar, ignorar, mensaje):
        if ruta.is_dir():
            self.api.upload_folder(repo_id=self.repo, repo_type=self.tipo, folder_path=ruta,
                                   path_in_repo=destino,
                                   # el README que escribe peft trae metadatos que el Hub puede rechazar
                                   # (base_model con una ruta local); no hace falta para cargar ni retomar
                                   ignore_patterns=[*(ignorar or []), "README.md"],
                                   delete_patterns="*" if reemplazar else None,
                                   commit_message=mensaje or f"Subir {destino}")
        else:
            self.api.upload_file(repo_id=self.repo, repo_type=self.tipo, path_or_fileobj=ruta,
                                 path_in_repo=destino, commit_message=mensaje or f"Subir {destino}")


def huella_archivo(ruta):
    """SHA-256 corto de un archivo: identifica el corpus con que se entrenó un checkpoint."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()[:16]


def huella_texto(*partes):
    """Huella corta de varios valores (p. ej. la huella del corpus más la configuración del entrenamiento)."""
    return hashlib.sha256(repr(partes).encode()).hexdigest()[:16]


def checkpoint_compatible(ck, huella):
    """True si el checkpoint `ck` se entrenó con el corpus de huella `huella` (archivo huella_corpus.txt).
    Un checkpoint sin huella o de otro corpus no se retoma: seguiría entrenando con datos distintos."""
    archivo = Path(ck) / "huella_corpus.txt"
    return archivo.exists() and archivo.read_text().strip() == huella


def callback_subir_checkpoints(hub, destino, huella=None):
    """TrainerCallback que copia cada checkpoint recién guardado a `destino` en el repo, reemplazando el anterior.
    Si la sesión se cae, el notebook lo baja y retoma desde ahí. Con `huella`, la deja en huella_corpus.txt dentro del
    checkpoint para no retomar nunca con un corpus distinto."""
    from transformers import TrainerCallback

    class SubirCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            ck = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if huella and ck.is_dir():
                (ck / "huella_corpus.txt").write_text(huella)
            if state.is_world_process_zero and ck.is_dir():
                hub.subir(ck, destino=destino, reemplazar=True,
                          mensaje=f"Checkpoint del paso {state.global_step}")

    return SubirCheckpoint()


def callback_progreso(ruta=None, extra=None):
    """TrainerCallback que mantiene un JSON corto con el estado del entrenamiento (paso, pérdidas, ETA).

    Por defecto lo escribe en $DIR_EJECUCION/progreso.json, la carpeta de la corrida que crea lanzar.py, que lo
    copia a estado.json y lo sube al Hub. Así se sigue un entrenamiento leyendo un archivo, sin abrir el runtime.
    `extra` agrega campos fijos (p. ej. el número de ensayo y el LR en el barrido)."""
    import os
    import time
    from datetime import datetime, timezone
    from transformers import TrainerCallback

    ruta = Path(ruta or Path(os.environ.get("DIR_EJECUCION", ".")) / "progreso.json")

    class Progreso(TrainerCallback):
        def __init__(self):
            self.t0 = None
            self.datos = {"fase": "preparando", **(extra or {})}

        def _escribir(self, state, **cambios):
            self.datos.update(cambios)
            seg = time.time() - self.t0 if self.t0 else 0
            paso, total = state.global_step, state.max_steps
            hechos = paso - self.datos.get("paso_inicial", 0)
            eta = seg / hechos * (total - paso) if hechos > 0 and total else None
            self.datos.update(
                paso=paso, pasos_totales=total, pct=round(100 * paso / total, 1) if total else None,
                epoca=round(state.epoch or 0, 3), segundos=round(seg),
                eta_min=round(eta / 60, 1) if eta is not None else None,
                actualizado_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            tmp = ruta.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.datos, indent=2, ensure_ascii=False))
            tmp.replace(ruta)   # escritura atómica: el lanzador nunca lee un JSON a medias

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            self._escribir(state, fase="entrenando", paso_inicial=state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            c = {}
            if "loss" in logs:
                c.update(loss=round(logs["loss"], 4), lr=logs.get("learning_rate"))
            if "eval_loss" in logs:
                c["eval_loss"] = round(logs["eval_loss"], 4)
                c["mejor_eval_loss"] = round(min(logs["eval_loss"], self.datos.get("mejor_eval_loss", float("inf"))), 4)
            self._escribir(state, **c)

        def on_save(self, args, state, control, **kwargs):
            self._escribir(state, ultimo_checkpoint=state.global_step)

        def on_train_end(self, args, state, control, **kwargs):
            self._escribir(state, fase="entrenamiento terminado")

    return Progreso()


# ---------------------------------------------------------------------------
# Normalización y limpieza de texto
# ---------------------------------------------------------------------------

def norm(s):
    """Minúsculas, sin tildes ni puntuación (la ñ se conserva)."""
    s = str(s).lower().replace("ñ", "\0")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").replace("\0", "ñ")
    s = re.sub(r"[^a-z0-9ñ\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_MENCION = re.compile(r"(?<!\w)@\w{1,30}")
RE_HASHTAG = re.compile(r"(?<!\w)#\w+")
RE_RT = re.compile(r"^RT\s+@?\w*:?\s*")
RE_MENCIONES_SEGUIDAS = re.compile(r"@usuario(?:[ ,]+@usuario)+")
RE_ESPACIOS = re.compile(r"[ \t ​]+")


def limpiar_texto(t, max_caracteres=4000):
    """Quita URLs y 'RT', anonimiza menciones (una cadena de menciones queda en un solo @usuario), normaliza
    espacios y recorta documentos muy largos."""
    t = unicodedata.normalize("NFC", str(t))
    t = RE_RT.sub("", t)
    t = RE_URL.sub("", t)
    t = RE_MENCION.sub("@usuario", t)
    t = RE_ESPACIOS.sub(" ", t)
    t = RE_MENCIONES_SEGUIDAS.sub("@usuario", t)   # respuestas a varios: "@usuario @usuario @usuario" -> "@usuario"
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    if len(t) > max_caracteres:   # corta en el último fin de oración o línea antes del límite
        corte = max(t.rfind(". ", 0, max_caracteres), t.rfind("\n", 0, max_caracteres))
        t = t[: corte + 1 if corte > max_caracteres // 2 else max_caracteres].strip()
    return t


def es_valido(t, min_palabras=5, min_letras=0.6):
    """Descarta textos muy cortos o con poca proporción de letras (tablas, código, listas de números)."""
    if len(t.split()) < min_palabras:
        return False
    sin_espacios = [c for c in t if not c.isspace()]
    return sum(c.isalpha() for c in sin_espacios) / max(len(sin_espacios), 1) >= min_letras


def huella(t):
    """Hash del texto normalizado, para deduplicar."""
    return hashlib.md5(norm(t).encode("utf-8")).hexdigest()


# Marcadores de registro coloquial chileno (heurísticos, sobre texto normalizado sin tildes).
MARCADORES = {
    "po": r"\bpo\b|\bpoh\b",
    "cachar": r"\bcach(ai|ay|ai\w*|ar|o|ae|aste|amos|ando|ado)\b",
    "weon/hueon/wn": r"\b(hue|we|gue)on(a|es|as|ada|adas|eo)?\b|\bwn\b|\bweas?\b|\bhueas?\b",
    "voseo chileno": r"\b(estai|estay|queris|sabis|podis|vai|vay|eris|soi|hablai|andai|decis|tenis que|cachai|pensai|creis)\b",
    "al tiro": r"\bal ?tiro\b",
    "bacan/fome/cuatico": r"\b(bacan|bacanes|fome|fomes|cuatico|cuatica|filete|la raja)\b",
    "pololo": r"\bpolol(o|a|os|as|ear|eando|eo)\b",
    "carrete": r"\bcarret(e|es|ear|eando|era|ero)\b",
    "cabros": r"\bcabr(o|a|os|as)( chic(o|a|os|as))?\b",
    "harto": r"\bhart(o|a|os|as)\b",
}
_MARCADORES_RE = {k: re.compile(v) for k, v in MARCADORES.items()}


def contar_marcadores(texto):
    t = norm(texto)
    return {k: len(r.findall(t)) for k, r in _MARCADORES_RE.items()}


class DetectorChilenismos:
    """Busca las expresiones del diccionario (clave, término y variantes) en un texto normalizado."""

    def __init__(self, ruta_rag, max_palabras=6, excluir=()):
        self.formas = {}
        with open(ruta_rag, encoding="utf-8") as f:
            for linea in f:
                if not linea.strip():
                    continue
                d = json.loads(linea)
                for forma in [d["clave"], d.get("termino", ""), *d.get("variantes", [])]:
                    fn = norm(forma)
                    if fn and len(fn.split()) <= max_palabras and fn not in excluir:
                        self.formas.setdefault(fn, d["clave"])
        self.primeras = {f.split()[0] for f in self.formas}
        self.max_n = max(len(f.split()) for f in self.formas)

    def buscar(self, texto, max_palabras_texto=None):
        """Lista de claves encontradas (con repeticiones)."""
        toks = norm(texto).split()
        if max_palabras_texto:
            toks = toks[:max_palabras_texto]
        encontradas = []
        for i, tok in enumerate(toks):
            if tok not in self.primeras:
                continue
            for n in range(min(self.max_n, len(toks) - i), 0, -1):   # prioriza la forma más larga
                clave = self.formas.get(" ".join(toks[i:i + n]))
                if clave:
                    encontradas.append(clave)
                    break
        return encontradas


# ---------------------------------------------------------------------------
# Priorización del corpus del paso 1: habla chilena primero, ruido fuera
# ---------------------------------------------------------------------------

# Ruido que se vio en la exploración (HALLAZGOS_DATASET.md): bots de Twitter y navegación de sitios en mc4
RE_RUIDO = [
    re.compile(r"^\d{1,2}:\d{2} en .{2,80}?\(\d{1,2}-\d"),                         # alertas de bomberos
    re.compile(r"acaba de publicar una? (foto|video)|^just posted a (photo|video)|^i'm at ", re.I),
]


def es_ruido(t):
    """Bots y navegación: no aportan habla chilena y se repiten miles de veces."""
    return t.count("»") >= 2 or any(r.search(t) for r in RE_RUIDO)


def claves_coloquiales(muestra, detector, fuente_coloquial="twitter", min_razon=8.0, min_conteo=3,
                       max_palabras_texto=400):
    """Elige qué expresiones del diccionario cuentan como señal de habla chilena al priorizar textos.

    Compara, en la muestra de exploración, la tasa por palabra en la fuente coloquial con la del resto:
    - De varias palabras: cuentan, salvo que sean claramente más frecuentes fuera de la fuente coloquial
      (p. ej. "la misión", que en mc4 es casi siempre literal).
    - De una palabra: cuentan si su tasa en la fuente coloquial es al menos `min_razon` veces la del resto
      (con `min_conteo` apariciones), o si aparecen en la fuente coloquial y nunca fuera. Así quedan fuera
      homónimos del español general como "completo", "feria" u "once". El umbral es alto porque Twitter usa
      más jerga en general: con razón 2-6 pasaban "mina", "hincha", "fresco" o "senda".
    Devuelve (conjunto de claves, DataFrame con el detalle por clave)."""
    import pandas as pd
    conteo = {True: Counter(), False: Counter()}
    palabras = {True: 0, False: 0}
    for t, fuente in zip(muestra.text, muestra.source):
        t = limpiar_texto(str(t))
        col = fuente == fuente_coloquial
        palabras[col] += min(len(t.split()), max_palabras_texto)
        conteo[col].update(detector.buscar(t, max_palabras_texto))
    filas = []
    for clave in sorted(set(detector.formas.values())):
        a, b = conteo[True][clave], conteo[False][clave]
        razon = ((a + 0.5) / max(palabras[True], 1)) / ((b + 0.5) / max(palabras[False], 1))
        varias = len(norm(clave).split()) > 1
        if varias:
            cuenta = not (b >= min_conteo and razon < 1)
        else:
            cuenta = (a >= min_conteo and razon >= min_razon) or (a >= 1 and b == 0)
        filas.append({"clave": clave, "varias_palabras": varias, f"en_{fuente_coloquial}": a, "en_resto": b,
                      "razon_tasas": round(razon, 2), "cuenta": cuenta})
    tabla = pd.DataFrame(filas).sort_values(["cuenta", f"en_{fuente_coloquial}"], ascending=False)
    return set(tabla[tabla.cuenta].clave), tabla


class Priorizador:
    """Marca un texto como prioritario si tiene marcadores coloquiales o expresiones del diccionario elegidas por
    claves_coloquiales con suficiente densidad: al menos una señal cada `palabras_por_senal` palabras. Un tweet
    con un "po" es prioritario; una noticia de 500 palabras con un "harto" no.
    Devuelve (prioritario, n_marcadores, expresiones encontradas)."""

    def __init__(self, detector, claves, max_palabras_texto=400, palabras_por_senal=150):
        self.detector, self.claves, self.max_palabras = detector, set(claves), max_palabras_texto
        self.palabras_por_senal = palabras_por_senal

    def __call__(self, t):
        n_marc = sum(contar_marcadores(t).values())
        expresiones = [c for c in self.detector.buscar(t, self.max_palabras) if c in self.claves]
        n_palabras = min(len(t.split()), self.max_palabras)
        requeridas = max(1, -(-n_palabras // self.palabras_por_senal))   # techo de la división
        return n_marc + len(expresiones) >= requeridas, n_marc, expresiones


# ---------------------------------------------------------------------------
# Corpus en Hugging Face: índice por row group y lectura parcial
# ---------------------------------------------------------------------------

def listar_parquets(repo=DATASET_CORPUS, fs=None):
    from huggingface_hub import HfFileSystem
    fs = fs or HfFileSystem()
    return sorted(fs.glob(f"datasets/{repo}/data/*.parquet")), fs


def _indice_de_archivo(ruta, fs, col_fuente="source"):
    import pyarrow.parquet as pq
    filas = []
    with fs.open(ruta, "rb") as f:
        pf = pq.ParquetFile(f)
        j = pf.schema_arrow.get_field_index(col_fuente)
        for rg in range(pf.num_row_groups):
            md = pf.metadata.row_group(rg)
            st = md.column(j).statistics
            mn = mx = None
            if st is not None and st.has_min_max:
                mn, mx = st.min, st.max
                mn = mn.decode() if isinstance(mn, bytes) else mn
                mx = mx.decode() if isinstance(mx, bytes) else mx
            if mn is not None and mn == mx:   # el row group tiene una sola fuente: basta con los metadatos
                filas.append({"archivo": ruta, "rg": rg, "fuente": mn, "filas": md.num_rows,
                              "bytes_rg": md.total_byte_size, "filas_rg": md.num_rows})
            else:                             # varias fuentes: se lee solo la columna de la fuente
                col = pf.read_row_group(rg, columns=[col_fuente]).column(0)
                for x in col.value_counts():
                    filas.append({"archivo": ruta, "rg": rg, "fuente": x["values"].as_py(),
                                  "filas": x["counts"].as_py(), "bytes_rg": md.total_byte_size,
                                  "filas_rg": md.num_rows})
    return filas


def indice_corpus(archivos, fs, hilos=8):
    """DataFrame con una fila por (archivo, row group, fuente): cuántas filas de esa fuente tiene.
    Lee solo los metadatos de cada parquet (y la columna 'source' si un row group mezcla fuentes)."""
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(hilos) as ex:
        partes = list(ex.map(lambda r: _indice_de_archivo(r, fs), archivos))
    return pd.DataFrame([f for p in partes for f in p])


def leer_row_groups(pares, fs, columnas=("text", "source"), hilos=8):
    """Lee los row groups indicados como [(archivo, rg), ...] y devuelve un DataFrame."""
    import pandas as pd
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor
    por_archivo = defaultdict(list)
    for archivo, rg in pares:
        por_archivo[archivo].append(int(rg))

    def leer(item):
        archivo, rgs = item
        with fs.open(archivo, "rb") as f:
            return pq.ParquetFile(f).read_row_groups(sorted(set(rgs)), columns=list(columnas)).to_pandas()

    with ThreadPoolExecutor(hilos) as ex:
        dfs = list(ex.map(leer, por_archivo.items()))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=list(columnas))


def elegir_row_groups(indice, fuente, filas_objetivo, seed=SEED):
    """Row groups al azar (repartidos por todo el corpus) que suman al menos `filas_objetivo` filas de la fuente."""
    sub = indice[indice.fuente == fuente].sample(frac=1.0, random_state=seed)
    elegidos, total = [], 0
    for r in sub.itertuples():
        elegidos.append((r.archivo, r.rg))
        total += r.filas
        if total >= filas_objetivo:
            break
    return elegidos


def muestra_por_fuente(indice, fs, n_por_fuente, seed=SEED):
    """Muestra estratificada por fuente. Se leen row groups completos (muestra por conglomerados)."""
    import pandas as pd
    partes = []
    for fuente in sorted(indice.fuente.unique()):
        pares = elegir_row_groups(indice, fuente, n_por_fuente * 1.2, seed)
        df = leer_row_groups(pares, fs)
        df = df[df.source == fuente]
        partes.append(df.sample(min(n_por_fuente, len(df)), random_state=seed))
    return pd.concat(partes, ignore_index=True)


def construir_corpus_paso1(indice, fs, tokenizer, pesos, tokens_objetivo, max_caracteres=4000,
                           min_palabras=5, rgs_por_lote=16, seed=SEED, log=print,
                           priorizador=None, fraccion_prioritaria=0.0, max_rgs_por_fuente=5000):
    """Junta textos limpios, sin duplicados ni ruido (bots, navegación), de cada fuente hasta cubrir
    tokens_objetivo * peso.

    Con `priorizador` y `fraccion_prioritaria` > 0, esa fracción de los tokens de la fuente sale de textos
    prioritarios (con marcadores o chilenismos del diccionario) y el resto de textos al azar de la misma fuente.
    `fraccion_prioritaria` es un número (todas las fuentes) o un dict {fuente: fracción}.
    El resto al azar mantiene español general en la mezcla, para que el modelo no olvide el registro neutro que
    después tiene que producir. Si una fuente no tiene suficientes textos prioritarios en `max_rgs_por_fuente`
    row groups, se completa con textos al azar y se avisa.
    Devuelve un DataFrame con columnas text, source, n_tokens, prioritario."""
    import pandas as pd
    vistos, filas = set(), []
    for fuente, peso in pesos.items():
        meta = int(tokens_objetivo * peso)
        if meta <= 0:
            continue
        frac = fraccion_prioritaria.get(fuente, 0.0) if isinstance(fraccion_prioritaria, dict) else fraccion_prioritaria
        meta_prio = int(meta * frac) if priorizador else 0
        sub = indice[indice.fuente == fuente].sample(frac=1.0, random_state=seed)
        pares = list(zip(sub.archivo, sub.rg))
        limite = min(len(pares), max_rgs_por_fuente) if meta_prio else len(pares)
        prio, resto = [], []
        tok_prio = tok_resto = 0
        i = leidos = ruido = n_prio_vistos = 0
        while i < limite and (tok_prio < meta_prio or tok_prio + tok_resto < meta):
            df = leer_row_groups(pares[i:min(i + rgs_por_lote, limite)], fs)
            i += rgs_por_lote
            df = df[df.source == fuente].sample(frac=1.0, random_state=seed)
            candidatos = []
            for t in df.text:
                t = limpiar_texto(t, max_caracteres)
                h = huella(t)
                if h in vistos or not es_valido(t, min_palabras):
                    continue
                leidos += 1
                if es_ruido(t):
                    ruido += 1
                    continue
                vistos.add(h)
                es_prio = bool(meta_prio) and priorizador(t)[0]
                n_prio_vistos += es_prio
                # un prioritario que sobra después de llenar su cuota sirve como texto al azar
                if es_prio and tok_prio < meta_prio:
                    candidatos.append((t, True))
                elif tok_resto < meta:
                    candidatos.append((t, False))
            if not candidatos:
                continue
            largos = [len(x) for x in tokenizer([t for t, _ in candidatos], add_special_tokens=False)["input_ids"]]
            for (t, es_prio), n in zip(candidatos, largos):
                if es_prio and tok_prio < meta_prio:
                    prio.append((t, n))
                    tok_prio += n + 2   # + <bos> y <eos>
                elif not es_prio and tok_resto < meta:
                    resto.append((t, n))
                    tok_resto += n + 2
        # Mezcla final: los prioritarios (hasta su cuota) y textos al azar hasta completar la meta de la fuente
        acumulado = 0
        for t, n in prio:
            filas.append({"text": t, "source": fuente, "n_tokens": n, "prioritario": True})
            acumulado += n + 2
        for t, n in resto:
            if acumulado >= meta:
                break
            filas.append({"text": t, "source": fuente, "n_tokens": n, "prioritario": False})
            acumulado += n + 2
        detalle = (f" | prioritarios: {tok_prio:,} tokens de {meta_prio:,} pedidos; "
                   f"{100 * n_prio_vistos / max(leidos - ruido, 1):.1f} % de los textos leídos lo eran"
                   if meta_prio else "")
        log(f"{fuente}: {acumulado:,} tokens de {meta:,} pedidos ({min(i, limite)} de {len(pares)} row groups "
            f"leídos; ruido descartado: {100 * ruido / max(leidos, 1):.1f} %){detalle}"
            + ("  <- NO ALCANZÓ: baja su peso o el presupuesto" if acumulado < meta else "")
            + ("  <- faltaron prioritarios: se completó al azar" if meta_prio and tok_prio < meta_prio else ""))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Presencia del diccionario en el corpus completo (escaneo con Aho-Corasick)
# ---------------------------------------------------------------------------

def cargar_diccionario(raiz):
    """Formas buscables de las expresiones de rag_documentos.jsonl: una fila por (forma normalizada, clave),
    con el término, si la expresión es de una o de varias palabras y su split en el dataset del paso 2."""
    import pandas as pd
    raiz = Path(raiz)
    filas = []
    with open(ruta_dato(raiz, RUTA_RAG), encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                d = json.loads(linea)
                for forma in [d["clave"], d.get("termino", ""), *d.get("variantes", [])]:
                    if len(norm(forma)) >= 2:
                        filas.append((norm(forma), d["clave"], d.get("termino", d["clave"])))
    dic = pd.DataFrame(filas, columns=["forma", "clave", "termino"]).drop_duplicates(["forma", "clave"])
    dic["varias_palabras"] = dic.clave.str.split().str.len() > 1
    # Con el split oficial toda expresión del dataset tiene pares en train (y uno en val o test)
    en_dataset = set(cargar_pares(raiz).clave)
    dic["split_dataset"] = dic.clave.map(lambda c: "train" if c in en_dataset else None)
    return dic.reset_index(drop=True)


def _normalizar_columna(col):
    """Versión vectorizada (pyarrow) de norm() para columnas de texto."""
    import pyarrow.compute as pc
    s = pc.utf8_lower(col)
    for patron, letra in [("[áàäâã]", "a"), ("[éèëê]", "e"), ("[íìïî]", "i"), ("[óòöôõ]", "o"), ("[úùüû]", "u")]:
        s = pc.replace_substring_regex(s, patron, letra)
    return pc.replace_substring_regex(s, r"[^a-z0-9ñ]+", " ")


def _escanear_archivo(args):
    """Proceso hijo: cuenta en un parquet cuántos textos contienen cada forma, por fuente."""
    archivo, rgs, formas, ruta_salida, lote_rg, protocolo = args
    import ahocorasick
    import fsspec
    import pyarrow.parquet as pq
    if Path(ruta_salida).exists():
        return ruta_salida
    A = ahocorasick.Automaton()
    for forma in formas:
        A.add_word(f" {forma} ", forma)
    A.make_automaton()
    docs, filas = defaultdict(Counter), Counter()
    with fsspec.filesystem(protocolo).open(archivo, "rb") as f:
        pf = pq.ParquetFile(f)
        for i in range(0, len(rgs), lote_rg):
            t = pf.read_row_groups(rgs[i:i + lote_rg], columns=["text", "source"])
            textos = _normalizar_columna(t.column("text")).to_pylist()
            for texto, fuente in zip(textos, t.column("source").to_pylist()):
                filas[fuente] += 1
                if texto:
                    docs[fuente].update({forma for _, forma in A.iter(f" {texto} ")})
    tmp = Path(str(ruta_salida) + ".tmp")
    tmp.write_text(json.dumps({"archivo": archivo, "filas": filas, "docs": docs}), encoding="utf-8")
    tmp.rename(ruta_salida)
    return ruta_salida


def escanear_diccionario(indice, formas, dir_salida, fraccion=1.0, procesos=None, lote_rg=8, seed=SEED,
                         protocolo="hf", log=print):
    """Cuenta en cuántos textos del corpus aparece cada forma (frecuencia de documento), por fuente.
    Lee una fracción `fraccion` de los row groups de cada parquet (1.0 = corpus completo). Cada parquet deja
    un json en dir_salida, así que si se corta se retoma. Devuelve (docs: forma x fuente, filas por fuente)."""
    import os
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    formas = sorted(set(formas))
    rng = random.Random(seed)
    tareas = []
    for archivo, g in indice.drop_duplicates(["archivo", "rg"]).groupby("archivo"):
        rgs = sorted(int(x) for x in g.rg if rng.random() < fraccion)
        if rgs:
            nombre = Path(archivo).stem + f"_f{fraccion}.json"
            tareas.append((archivo, rgs, formas, str(dir_salida / nombre), lote_rg, protocolo))
    procesos = procesos or max(1, (os.cpu_count() or 2) - 1)
    log(f"{len(tareas)} parquets, {sum(len(t[1]) for t in tareas):,} row groups, {procesos} procesos")
    with ProcessPoolExecutor(procesos) as ex:
        futuros = [ex.submit(_escanear_archivo, t) for t in tareas]
        for k, fut in enumerate(as_completed(futuros), 1):
            fut.result()
            if k % 5 == 0 or k == len(futuros):
                log(f"  {k}/{len(futuros)} parquets listos")
    docs, filas = defaultdict(Counter), Counter()
    for t in tareas:
        r = json.loads(Path(t[3]).read_text(encoding="utf-8"))
        filas.update(r["filas"])
        for fuente, c in r["docs"].items():
            docs[fuente].update(c)
    tabla = pd.DataFrame(docs).reindex(formas).fillna(0).astype(int)
    return tabla, pd.Series(filas, name="filas_escaneadas")


def presencia_por_clave(dic, docs_forma, fraccion=1.0):
    """Agrega por clave: textos estimados del corpus que contienen alguna de sus formas (suma de formas: cota
    superior si un texto trae dos variantes de la misma expresión), extrapolado según la fracción leída."""
    por_fuente = docs_forma.reindex(dic.forma).fillna(0).groupby(dic.clave.values).sum() / fraccion
    tabla = dic.drop_duplicates("clave").set_index("clave")[["termino", "varias_palabras", "split_dataset"]]
    tabla["formas"] = dic.groupby("clave").forma.apply(lambda f: " | ".join(sorted(set(f))))
    tabla["textos_estimados"] = por_fuente.sum(axis=1).reindex(tabla.index).fillna(0).round().astype(int)
    for fuente in por_fuente.columns:
        tabla[f"textos_{fuente}"] = por_fuente[fuente].reindex(tabla.index).fillna(0).round().astype(int)
    return tabla.sort_values(["textos_estimados", "termino"], ascending=[False, True])


def resumen_presencia(tabla, umbrales=(1, 10, 100, 1000)):
    """Cuántas expresiones de cada grupo aparecen en al menos u textos (estimados)."""
    import pandas as pd
    grupos = {
        "todas (rag_documentos)": tabla.index == tabla.index,
        "  de una palabra": ~tabla.varias_palabras,
        "  de varias palabras": tabla.varias_palabras,
        "en el dataset del paso 2": tabla.split_dataset.notna(),
    }
    for s in ("train", "val", "test"):
        grupos[f"  split {s}"] = tabla.split_dataset == s
    filas = []
    for nombre, m in grupos.items():
        sub = tabla[m]
        fila = {"grupo": nombre, "expresiones": len(sub)}
        for u in umbrales:
            n = int((sub.textos_estimados >= u).sum())
            fila[f">= {u} textos"] = f"{n} ({100 * n / max(len(sub), 1):.0f} %)"
        fila["mediana_textos"] = int(sub.textos_estimados.median()) if len(sub) else 0
        filas.append(fila)
    return pd.DataFrame(filas).set_index("grupo")


def acotar_pesos(pesos, minimo, maximo):
    """Normaliza los pesos a suma 1 dejando cada uno en [minimo, maximo]: fija primero los que quedan bajo
    el mínimo, luego los que pasan el máximo, y reparte el resto en proporción entre los libres. Si las cotas
    son incompatibles (p. ej. pocas fuentes), el sobrante va a las fuentes que no están en el máximo."""
    import pandas as pd
    p = pd.Series(pesos, dtype=float).clip(lower=0)
    p = p / p.sum() if p.sum() > 0 else pd.Series(1 / len(p), index=p.index)
    fijos = {}
    while len(fijos) < len(p):
        libres = [k for k in p.index if k not in fijos]
        resto = 1 - sum(fijos.values())
        base = p[libres]
        q = base / base.sum() * resto if base.sum() > 0 else pd.Series(resto / len(libres), index=libres)
        if (q < minimo).any():
            fijos.update({k: minimo for k in q[q < minimo].index})
        elif (q > maximo).any():
            fijos.update({k: maximo for k in q[q > maximo].index})
        else:
            fijos.update(q.to_dict())
    r = pd.Series(fijos).reindex(p.index)
    sobra = 1 - r.sum()
    if abs(sobra) > 1e-9:
        receptores = r[r < maximo].index if sobra > 0 else r.index
        r[receptores] += sobra / len(receptores)
    return r


# ---------------------------------------------------------------------------
# Paso 1: empaquetado para NTP
# ---------------------------------------------------------------------------

def empaquetar(lote, tokenizer, largo):
    """Para Dataset.map(batched=True): <bos> texto <eos> concatenados y cortados en bloques de `largo`."""
    ids = tokenizer(lote["text"], add_special_tokens=False)["input_ids"]
    flujo = []
    for x in ids:
        flujo.extend([tokenizer.bos_token_id, *x, tokenizer.eos_token_id])
    n = len(flujo) // largo * largo
    bloques = [flujo[i:i + largo] for i in range(0, n, largo)]
    return {"input_ids": bloques, "labels": [list(b) for b in bloques]}


# ---------------------------------------------------------------------------
# Paso 2: pares de traducción, splits por expresión y formato de prompt
# ---------------------------------------------------------------------------

PROMPT = "Traduce la frase del español de Chile al español neutro.\nChileno: {chilena}\nNeutro:"
# Para modelos instruct (-it) se usa su plantilla de chat con esta instrucción como turno del usuario.
INSTRUCCION_CHAT = ("Traduce la siguiente frase del español de Chile al español neutro. "
                    "Responde solo con la traducción.\n\n{chilena}")


def cargar_pares(raiz):
    """DataFrame de pares con la columna split, del split oficial del grupo (por par, estratificado por expresión:
    4 pares de cada expresión en train y 1 en val o test). Se genera con data/entrenamiento/splits/crear_split_oficial.py."""
    import pandas as pd
    ruta_split = ruta_dato(raiz, RUTA_SPLIT)
    df = pd.read_csv(ruta_dato(raiz, RUTA_PARES), encoding="utf-8-sig")
    df = df.merge(pd.read_csv(ruta_split)[["id_mistral", "split"]], on="id_mistral", how="left")
    df = df.dropna(subset=["frase_chilena", "frase_neutralizada"])
    df = df[(df.frase_chilena.str.strip() != "") & (df.frase_neutralizada.str.strip() != "")]
    faltan = df.split.isna().sum()
    if faltan:
        raise ValueError(f"{faltan} pares no están en {ruta_split}; regenéralo con crear_split_oficial.py")
    return df.reset_index(drop=True)


def ids_prompt(chilena, tokenizer, chat=False):
    """Prompt tokenizado (con <bos>). chat=True usa la plantilla de chat del modelo instruct."""
    if chat:
        texto = tokenizer.apply_chat_template(
            [{"role": "user", "content": INSTRUCCION_CHAT.format(chilena=chilena.strip())}],
            tokenize=False, add_generation_prompt=True)
        return tokenizer(texto, add_special_tokens=False)["input_ids"]   # la plantilla ya trae <bos>
    return [tokenizer.bos_token_id] + tokenizer(PROMPT.format(chilena=chilena.strip()),
                                                add_special_tokens=False)["input_ids"]


def ids_fin_respuesta(tokenizer, chat=False):
    """Token que cierra la respuesta: <eos> en el formato base, fin de turno (<turn|>) en el de chat."""
    if chat:
        fin = getattr(tokenizer, "eot_token", None) or "<turn|>"
        return [tokenizer.convert_tokens_to_ids(fin)]
    return [tokenizer.eos_token_id]


def tokenizar_par(chilena, neutra, tokenizer, max_len, chat=False):
    """input_ids y labels con la pérdida solo sobre la respuesta (el prompt queda en -100)."""
    p = ids_prompt(chilena, tokenizer, chat)
    respuesta = neutra.strip() if chat else " " + neutra.strip()
    r = tokenizer(respuesta, add_special_tokens=False)["input_ids"] + ids_fin_respuesta(tokenizer, chat)
    ids = (p + r)[:max_len]
    labels = ([-100] * len(p) + r)[:max_len]
    return {"input_ids": ids, "labels": labels}


class ColadorPares:
    """Rellena a la derecha (entrenamiento) con pad y -100 en las etiquetas."""

    def __init__(self, pad_id, multiplo=8):
        self.pad_id, self.multiplo = pad_id, multiplo

    def __call__(self, ejemplos):
        import torch
        largo = max(len(e["input_ids"]) for e in ejemplos)
        largo = -(-largo // self.multiplo) * self.multiplo
        ids, lab, att = [], [], []
        for e in ejemplos:
            falta = largo - len(e["input_ids"])
            ids.append(list(e["input_ids"]) + [self.pad_id] * falta)
            lab.append(list(e["labels"]) + [-100] * falta)
            att.append([1] * len(e["input_ids"]) + [0] * falta)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


# ---------------------------------------------------------------------------
# Modelo: carga, LoRA por grupo de módulos, perplejidad y generación
# ---------------------------------------------------------------------------

GRUPOS_LORA = {
    "mlp": ("gate_proj", "up_proj", "down_proj"),        # red feed-forward por posición
    "atencion": ("q_proj", "k_proj", "v_proj", "o_proj"),
}


def regex_lora(grupos):
    """Regex para LoRA: solo capas del decodificador de texto (excluye torres de visión y audio).
    Los embeddings, normalizaciones y embeddings por capa (PLE) quedan siempre congelados."""
    nombres = [n for g in grupos for n in GRUPOS_LORA[g]]
    return r"^(?!.*(?:vision|audio)).*layers\.\d+\..*\.(?:" + "|".join(nombres) + r")$"


def cargar_modelo(modelo_id, adaptador_a_fusionar=None, dtype="bfloat16"):
    """Carga el modelo base en bf16. Si se pasa un adaptador (p. ej. el del paso 1), lo fusiona en los pesos.
    No se fuerza attn_implementation: la torre de audio de Gemma 4 no soporta sdpa y transformers elige por submódulo."""
    import torch
    from transformers import AutoModelForCausalLM
    modelo = AutoModelForCausalLM.from_pretrained(modelo_id, dtype=getattr(torch, dtype),
                                                  device_map={"": 0} if torch.cuda.is_available() else None)
    if adaptador_a_fusionar:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, adaptador_a_fusionar).merge_and_unload()
    modelo.config.use_cache = False
    return modelo


def agregar_lora(modelo, grupos, r, alpha, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none", task_type="CAUSAL_LM",
                     target_modules=regex_lora(grupos))
    modelo = get_peft_model(modelo, cfg)
    modelo.print_trainable_parameters()
    return modelo


def perplejidad(modelo, tokenizer, textos, max_len=256, bs=16):
    """Perplejidad media por token de una lista de textos (cada uno con <bos>, sin <eos>)."""
    import math
    import torch
    modelo.eval()
    disp = next(modelo.parameters()).device
    nll, n = 0.0, 0
    for i in range(0, len(textos), bs):
        ids = [[tokenizer.bos_token_id] + x[: max_len - 1]
               for x in tokenizer(list(textos[i:i + bs]), add_special_tokens=False)["input_ids"]]
        largo = max(map(len, ids))
        x = torch.full((len(ids), largo), tokenizer.pad_token_id)
        m = torch.zeros_like(x)
        for j, s in enumerate(ids):
            x[j, :len(s)] = torch.tensor(s)
            m[j, :len(s)] = 1
        x, m = x.to(disp), m.to(disp)
        with torch.no_grad():
            logits = modelo(input_ids=x, attention_mask=m).logits
            # fila por fila: con 262k de vocabulario, los logits del lote en fp32 más el log_softmax ocupan GB
            for j in range(len(ids)):
                mask = m[j, 1:].bool()
                ce = torch.nn.functional.cross_entropy(logits[j, :-1].float(), x[j, 1:], reduction="none")
                nll += ce[mask].sum().item()
                n += mask.sum().item()
            del logits
    return math.exp(nll / max(n, 1))


def traducir(modelo, tokenizer, frases, bs=32, max_new_tokens=96, chat=False):
    """Generación greedy con el prompt de entrenamiento. Corta en <eos>, fin de turno o el primer salto de línea."""
    import torch
    modelo.eval()
    disp = next(modelo.parameters()).device
    salidas = []
    for i in range(0, len(frases), bs):
        ids = [ids_prompt(f, tokenizer, chat) for f in frases[i:i + bs]]
        largo = max(map(len, ids))
        x = torch.full((len(ids), largo), tokenizer.pad_token_id)
        m = torch.zeros_like(x)
        for j, s in enumerate(ids):   # relleno a la izquierda para generar en lote
            x[j, largo - len(s):] = torch.tensor(s)
            m[j, largo - len(s):] = 1
        with torch.no_grad():
            out = modelo.generate(input_ids=x.to(disp), attention_mask=m.to(disp), do_sample=False,
                                  max_new_tokens=max_new_tokens,
                                  eos_token_id=sorted({tokenizer.eos_token_id, *ids_fin_respuesta(tokenizer, chat)}),
                                  pad_token_id=tokenizer.pad_token_id, use_cache=True)
        for fila in out[:, largo:]:
            texto = tokenizer.decode(fila, skip_special_tokens=True)
            salidas.append(texto.strip().split("\n")[0].strip())
    return salidas


def metricas_traduccion(df, col_pred="prediccion"):
    """BLEU y chrF++ contra la referencia, más dos señales léxicas sobre la expresión chilena."""
    import sacrebleu
    pred, ref = df[col_pred].fillna("").tolist(), df.frase_neutralizada.tolist()

    def contiene(frase, clave):
        return f" {norm(clave)} " in f" {norm(frase)} "

    return {
        "n": len(df),
        "bleu": round(sacrebleu.corpus_bleu(pred, [ref]).score, 2),
        "chrf++": round(sacrebleu.corpus_chrf(pred, [ref], word_order=2).score, 2),
        "expresion_en_chilena_%": round(100 * sum(map(contiene, df.frase_chilena, df.clave)) / len(df), 1),
        "expresion_en_prediccion_%": round(100 * sum(map(contiene, pred, df.clave)) / len(df), 1),
        "copia_literal_%": round(100 * sum(norm(p) == norm(c) for p, c in zip(pred, df.frase_chilena)) / len(df), 1),
    }


# ---------------------------------------------------------------------------
# ¿El modelo instruct sigue obedeciendo instrucciones tras el ajuste?
# ---------------------------------------------------------------------------

def _lineas(r):
    return [l for l in (x.strip(" -*•\t") for x in r.splitlines()) if l]


# Instrucciones generales (ninguna es traducir chileno) con una verificación automática simple. Si tras el paso 2
# el modelo responde todo como si fuera una traducción al español neutro, estas fallan.
INSTRUCCIONES_PRUEBA = [
    ("¿Cuál es la capital de Francia? Responde con una sola palabra.",
     lambda r: "paris" in norm(r) and len(norm(r).split()) <= 3),
    ("¿Cuánto es 7 por 8? Responde solo con el número.",
     lambda r: "56" in r and len(r.split()) <= 3),
    ("Escribe exactamente tres frutas, una por línea, sin nada más.",
     lambda r: len(_lineas(r)) == 3),
    ("Answer in English: what color is the sky on a clear day?",
     lambda r: "blue" in r.lower()),
    ("Escribe la palabra hola en mayúsculas y nada más.",
     lambda r: "HOLA" in r and len(r.split()) <= 2),
    ("Ordena alfabéticamente estas palabras y escríbelas separadas por comas: pera, auto, casa",
     lambda r: norm(r).startswith("auto casa pera")),
    ("Si 10 es mayor que 3 responde sí; si no, responde no. Usa una sola palabra.",
     lambda r: norm(r).split()[:1] == ["si"]),
    ("Traduce al inglés: el perro come pan.",
     lambda r: "dog" in r.lower() and "bread" in r.lower()),
]


def seguir_instrucciones(modelo, tokenizer, max_new_tokens=48):
    """Pasa INSTRUCCIONES_PRUEBA con la plantilla de chat (greedy). Devuelve (aciertos, detalle por instrucción)."""
    import torch
    modelo.eval()
    disp = next(modelo.parameters()).device
    detalle = []
    for instruccion, ok in INSTRUCCIONES_PRUEBA:
        texto = tokenizer.apply_chat_template([{"role": "user", "content": instruccion}],
                                              tokenize=False, add_generation_prompt=True)
        x = tokenizer(texto, add_special_tokens=False, return_tensors="pt").to(disp)
        with torch.no_grad():
            out = modelo.generate(**x, do_sample=False, max_new_tokens=max_new_tokens,
                                  eos_token_id=sorted({tokenizer.eos_token_id, *ids_fin_respuesta(tokenizer, True)}),
                                  pad_token_id=tokenizer.pad_token_id, use_cache=True)
        r = tokenizer.decode(out[0, x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        try:
            acierto = bool(ok(r))
        except Exception:
            acierto = False
        detalle.append({"instruccion": instruccion, "respuesta": r, "ok": acierto})
    return sum(d["ok"] for d in detalle), detalle
