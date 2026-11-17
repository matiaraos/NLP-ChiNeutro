"""Corre un notebook de models/gemma4 sin interfaz (papermill) y sube al Hub su log y el notebook ejecutado.

Pensado para un runtime de Colab manejado por API, donde no hay navegador ni Drive montado. Preparación:

    pip install -q papermill
    hf download <REPO_HF> --include "datos/*" --local-dir /content/chineutro
    nohup python /content/chineutro/datos/lanzar.py 01_paso1_ntp_corpus.ipynb --repo <REPO_HF> \\
        > /content/chineutro/lanzar.log 2>&1 &

Parámetros del notebook con -p NOMBRE VALOR (texto) o -y NOMBRE VALOR (YAML: números, listas, null), p. ej.:
    lanzar.py 02_paso2_traduccion.ipynb --repo <REPO_HF> -p CONDICION paso1+2

Cada corrida queda en DIR_TRABAJO/ejecuciones/<notebook>_<parámetros>_<fecha>/ (y en esa ruta del repo):
notebook ejecutado, log.txt y estado.json. estado.json es el resumen para seguir la corrida sin abrir el runtime:
estado (corriendo | terminado | error), celda en ejecución y, mientras entrena, el progreso que escribe
comun.callback_progreso (paso, pérdidas, ETA). Se sube cada --estado-cada segundos; el log y el notebook
parcial, cada --subir-cada. models/gemma4/monitor_hub.py lo lee desde el Hub.
Los intervalos por defecto (5 y 20 min) cuidan el límite del Hub de 128 commits por hora por repo, que se comparte
entre todas las corridas que suben al mismo repo.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def con_limite(funcion, segundos, descripcion):
    """Corre funcion() en un hilo daemon con tiempo máximo: si el Hub limita los commits (429), huggingface_hub espera
    hasta 1 h antes de reintentar, y el lanzador no debe quedar colgado esa hora (ni al terminar)."""
    import threading
    error = []

    def correr():
        try:
            funcion()
        except Exception as e:
            error.append(e)

    hilo = threading.Thread(target=correr, daemon=True)
    hilo.start()
    hilo.join(segundos)
    if hilo.is_alive() or error:
        print(f"⚠ {descripcion}: {error[0] if error else f'no terminó en {segundos} s'}", flush=True)


def subir(api, repo, carpeta, destino, mensaje):
    con_limite(lambda: api.upload_folder(repo_id=repo, folder_path=carpeta, path_in_repo=destino, commit_message=mensaje),
               300, f"No pude subir {destino}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", help="nombre del notebook en DIR_TRABAJO/datos, o ruta a él")
    ap.add_argument("--repo", required=True, help="repo privado del Hub (usuario/nombre)")
    ap.add_argument("--dir-trabajo", default="/content/chineutro")
    ap.add_argument("-p", nargs=2, action="append", default=[], metavar=("NOMBRE", "VALOR"))
    ap.add_argument("-y", nargs=2, action="append", default=[], metavar=("NOMBRE", "YAML"))
    ap.add_argument("--subir-cada", type=int, default=1200, help="segundos entre subidas del log y el notebook")
    ap.add_argument("--estado-cada", type=int, default=300, help="segundos entre subidas de estado.json")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()

    dir_t = Path(a.dir_trabajo)
    nb = Path(a.notebook) if Path(a.notebook).is_file() else dir_t / "datos" / a.notebook
    assert nb.is_file(), f"No encontré {nb}"
    fecha = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    sufijo = "_".join(re.sub(r"[^\w.+-]", "-", str(v)) for _, v in a.p + a.y)
    nombre = "_".join(x for x in (nb.stem, sufijo, fecha) if x)
    dir_e = dir_t / "ejecuciones" / nombre
    dir_e.mkdir(parents=True, exist_ok=True)
    destino = f"ejecuciones/{nombre}"

    cmd = [sys.executable, "-m", "papermill", str(nb), str(dir_e / nb.name),
           "--log-output", "--request-save-on-cell-execute", "--cwd", str(dir_t), "-k", "python3",
           "-p", "REPO_HF", a.repo, "-p", "DIR_TRABAJO", str(dir_t)]
    for n, v in a.p:
        cmd += ["-p", n, v]
    for n, v in a.y:
        cmd += ["-y", f"{n}: {v}"]

    n_celdas = len(json.loads(nb.read_text(encoding="utf-8"))["cells"]) + 1   # + la que inyecta papermill

    def estado(e, **extra):
        """Resumen de la corrida: estado, celda en curso y progreso del entrenamiento (si lo hay)."""
        celdas = re.findall(r"Executing Cell (\d+)", (dir_e / "log.txt").read_text(errors="replace")) \
            if (dir_e / "log.txt").exists() else []
        try:
            progreso = json.loads((dir_e / "progreso.json").read_text())
        except (OSError, ValueError):
            progreso = None
        (dir_e / "estado.json").write_text(json.dumps(
            {"estado": e, "notebook": nb.name, "parametros": dict(a.p + a.y), "inicio_utc": fecha,
             "actualizado_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
             "celda": f"{celdas[-1] if celdas else 0}/{n_celdas}", "entrenamiento": progreso,
             **extra}, indent=2, ensure_ascii=False))

    def subir_estado():
        con_limite(lambda: api.upload_file(repo_id=a.repo, path_or_fileobj=dir_e / "estado.json",
                                           path_in_repo=f"{destino}/estado.json", commit_message=f"{nombre}: estado"),
                   60, "No pude subir estado.json")

    estado("corriendo")
    subir_estado()
    print("Corrida:", nombre, flush=True)
    entorno = {**os.environ, "DIR_EJECUCION": str(dir_e)}   # comun.callback_progreso escribe ahí progreso.json
    with open(dir_e / "log.txt", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=dir_t, env=entorno)
        ultima_subida = ultimo_estado = time.time()
        while proc.poll() is None:
            time.sleep(15)
            estado("corriendo")
            if time.time() - ultima_subida >= a.subir_cada:
                subir(api, a.repo, dir_e, destino, f"{nombre}: avance")
                ultima_subida = ultimo_estado = time.time()
            elif time.time() - ultimo_estado >= a.estado_cada:
                subir_estado()
                ultimo_estado = time.time()
    codigo = proc.returncode
    estado("terminado" if codigo == 0 else "error", codigo_salida=codigo)
    subir(api, a.repo, dir_e, destino, f"{nombre}: {'terminado' if codigo == 0 else 'error'}")
    print("Fin:", "terminado" if codigo == 0 else f"error (código {codigo})", flush=True)
    sys.exit(codigo)


if __name__ == "__main__":
    main()
