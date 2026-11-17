"""Encadena, sin interfaz, el barrido de LR y los entrenamientos completos con el LR que recomienda el barrido.

    python datos/encadenar.py paso1 --repo <REPO_HF> [--rangos 64 32 128]
    python datos/encadenar.py paso2 solo_paso2 it_solo_paso2 --repo <REPO_HF> [--rangos 16 8 32]
    python datos/encadenar.py paso1+2 --repo <REPO_HF> [--rangos-paso1 64 32 128]
    python datos/encadenar.py paso1 --repo <REPO_HF> --rangos 64 32 128 --lr 3e-4   # LR fijo, sin barrido
    python datos/encadenar.py paso1+2 --repo <REPO_HF> --rangos-paso1 32 --rangos 8 32 --lr 2e-4   # rango del paso 2

- paso1: 01b_barrido_lr (con el primer rango) -> 01_paso1_ntp_corpus para cada rango de --rangos.
- paso2: por cada condición: 02b_barrido_lr_paso2 (con el primer rango) -> 02_paso2_traduccion para cada rango.
- paso1+2: necesita los adaptadores del paso 1. Barrido de 02b con el adaptador del primer rango de --rangos-paso1 y el
  primer rango de --rangos -> 02_paso2_traduccion paso1+2 para cada combinación (adaptador del paso 1) x (rango del
  paso 2, por defecto 16).

Ablación del rango: el LR se elige una vez por condición, con el primer rango (el de referencia), y se usa igual en
los demás; alpha = 2 * r siempre. El primer rango va primero: si la cola se corta, el resultado principal ya quedó.

Cada notebook corre con lanzar.py (papermill + estado.json en el Hub). Si un barrido falla o no encuentra un LR
aceptable, esa condición no se entrena y se sigue con la siguiente. El registro queda en
ejecuciones/cadena_<fase>.json (local y en el Hub).
"""
import argparse
import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path


class Yaml(str):
    """Valor que se pasa con -y (YAML: números, listas) en vez de -p (texto)."""


def lanzar(dir_t, repo, notebook, parametros):
    cmd = [sys.executable, str(dir_t / "datos" / "lanzar.py"), notebook, "--repo", repo, "--dir-trabajo", str(dir_t)]
    for n, v in parametros:
        cmd += ["-p" if isinstance(v, str) and not isinstance(v, Yaml) else "-y", n, str(v)]
    print(">>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=dir_t).returncode


def lr_recomendado(patron):
    rutas = sorted(glob.glob(patron), key=lambda p: Path(p).stat().st_mtime)
    if not rutas:
        return None
    return json.loads(Path(rutas[-1]).read_text()).get("lr_recomendado")


def rango(r):
    return [("LORA_R", Yaml(r)), ("LORA_ALPHA", Yaml(2 * r))]


def nombre_paso1(r, presupuesto_m, prio):
    return f"mlp_r{r}_{presupuesto_m}M{prio}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fase", choices=["paso1", "paso2", "paso1+2"])
    ap.add_argument("condiciones", nargs="*", default=["solo_paso2", "it_solo_paso2"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dir-trabajo", default="/content/chineutro")
    ap.add_argument("--rangos", nargs="+", type=int, default=None,
                    help="rangos LoRA del paso que se entrena (en paso1+2, los del paso 2); el primero es el de referencia (barrido)")
    ap.add_argument("--rangos-paso1", nargs="+", type=int, default=[64],
                    help="paso1+2: rangos de los adaptadores del paso 1 a fusionar; el primero se usa en el barrido")
    ap.add_argument("--nombre-paso1", default="mlp_r{r}_5M_priotw80",
                    help="paso1+2: carpeta del adaptador del paso 1, con {r} para el rango")
    ap.add_argument("--lr", type=float, default=None,
                    help="LR fijo: se salta el barrido y se usa este valor en los entrenamientos")
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("NOMBRE", "VALOR"),
                    help="parámetro de texto para todos los notebooks (p. ej. MODELO_BASE en pruebas)")
    ap.add_argument("-y", nargs=2, action="append", default=[], metavar=("NOMBRE", "YAML"),
                    help="parámetro YAML para todos los notebooks (p. ej. MAX_PASOS en pruebas)")
    a = ap.parse_args()
    extra = [(n, v) for n, v in a.p] + [(n, Yaml(v)) for n, v in a.y]
    dir_t = Path(a.dir_trabajo)

    # (descripción, barrido, parámetros del barrido, patrón del resultado, [(entrenamiento, parámetros), ...])
    if a.fase == "paso1":
        rangos = a.rangos or [64]
        tareas = [("paso1", "01b_barrido_lr.ipynb", rango(rangos[0]),
                   str(dir_t / "paso1" / f"*_r{rangos[0]}_*" / "barrido_lr" / "resultado.json"),
                   [("01_paso1_ntp_corpus.ipynb", rango(r)) for r in rangos])]
    elif a.fase == "paso2":
        rangos = a.rangos or [16]
        tareas = [(c, "02b_barrido_lr_paso2.ipynb", [("CONDICION", c)] + rango(rangos[0]),
                   str(dir_t / "paso2" / "barrido_lr" / f"{c}__*_r{rangos[0]}" / "resultado.json"),
                   [("02_paso2_traduccion.ipynb", [("CONDICION", c)] + rango(r)) for r in rangos])
                  for c in a.condiciones]
    else:
        r1 = a.rangos_paso1
        r2 = a.rangos or [16]
        nombre = a.nombre_paso1
        tareas = [("paso1+2", "02b_barrido_lr_paso2.ipynb",
                   [("CONDICION", "paso1+2"), ("NOMBRE_PASO1", nombre.format(r=r1[0]))] + rango(r2[0]),
                   str(dir_t / "paso2" / "barrido_lr" / f"paso1+2__*_r{r2[0]}__p1-{nombre.format(r=r1[0])}" / "resultado.json"),
                   [("02_paso2_traduccion.ipynb",
                     [("CONDICION", "paso1+2"), ("NOMBRE_PASO1", nombre.format(r=r))] + rango(rr))
                    for r in r1 for rr in r2])]

    registro = []
    salida = dir_t / "ejecuciones" / f"cadena_{a.fase}.json"
    for descripcion, barrido, p_barrido, patron, entrenamientos in tareas:
        r = {"tarea": descripcion, "barrido": barrido, "parametros": {n: str(v) for n, v in p_barrido},
             "inicio_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "entrenamientos": []}
        if a.lr is not None:
            r["codigo_barrido"], r["lr"] = "omitido (--lr)", a.lr
        else:
            r["codigo_barrido"] = lanzar(dir_t, a.repo, barrido, p_barrido + extra)
            r["lr"] = lr_recomendado(patron) if r["codigo_barrido"] == 0 else None
        if r["lr"] is None:
            r["entrenamientos"] = "omitidos: el barrido falló o no encontró un LR aceptable"
            print("!!", descripcion, r["entrenamientos"], flush=True)
        else:
            for nb, p_entrenamiento in entrenamientos:
                codigo = lanzar(dir_t, a.repo, nb, p_entrenamiento + extra + [("LR", Yaml(float(r["lr"])))])
                r["entrenamientos"].append({"parametros": {n: str(v) for n, v in p_entrenamiento},
                                            "resultado": "terminado" if codigo == 0 else f"error ({codigo})"})
                guardar(registro + [r], salida, a.repo, a.fase, f"{descripcion}: {r['entrenamientos'][-1]['resultado']}")
        registro.append(r)
        guardar(registro, salida, a.repo, a.fase, f"{descripcion}: listo")
    print("Cadena terminada:", json.dumps(registro, ensure_ascii=False), flush=True)


def guardar(registro, salida, repo, fase, mensaje):
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(registro, indent=2, ensure_ascii=False))
    sys.path.insert(0, str(Path(__file__).parent))
    from lanzar import con_limite   # no quedar colgado 1 h si el Hub limita los commits
    from huggingface_hub import HfApi
    con_limite(lambda: HfApi().upload_file(repo_id=repo, path_or_fileobj=salida,
                                           path_in_repo=f"ejecuciones/cadena_{fase}.json",
                                           commit_message=f"Cadena {fase}: {mensaje}"),
               60, "No pude subir el registro de la cadena")


if __name__ == "__main__":
    main()
