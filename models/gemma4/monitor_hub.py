"""Sigue desde el computador local las corridas que lanzar.py sube al Hub, leyendo solo sus estado.json.

    python3 models/gemma4/monitor_hub.py --repo usuario/chineutro-gemma4 [--cada 60] [--filtro 01_paso1]

Imprime una línea por corrida cada vez que cambia algo relevante: el estado, la celda, una nueva eval_loss o
cada 10 % de avance del entrenamiento. Termina cuando todas las corridas vigiladas están en terminado o error
(con --siempre sigue indefinidamente). Una corrida se vigila si estaba corriendo al empezar o aparece después.
No toca el runtime de Colab: sirve con la máquina ocupada o caída (en ese caso estado.json deja de actualizarse,
y la línea muestra hace cuántos minutos fue la última actualización).
"""
import argparse
import datetime
import json
import sys
import time


def resumen(e):
    t = e.get("entrenamiento") or {}
    partes = [e["estado"], f"celda {e.get('celda', '?')}"]
    if t:
        partes.append(f"{t.get('fase')} paso {t.get('paso')}/{t.get('pasos_totales')} ({t.get('pct')} %)")
        if t.get("loss") is not None:
            partes.append(f"loss {t['loss']}")
        if t.get("eval_loss") is not None:
            partes.append(f"eval {t['eval_loss']}")
        if t.get("eta_min") is not None:
            partes.append(f"ETA {t['eta_min']} min")
    if e.get("codigo_salida") not in (None, 0):
        partes.append(f"código {e['codigo_salida']}")
    try:
        act = datetime.datetime.fromisoformat(e["actualizado_utc"])
        mins = (datetime.datetime.now(datetime.timezone.utc) - act).total_seconds() / 60
        partes.append(f"act. hace {mins:.0f} min")
    except (KeyError, ValueError):
        pass
    return " | ".join(partes)


def clave_cambio(e):
    t = e.get("entrenamiento") or {}
    return (e["estado"], e.get("celda"), t.get("fase"), t.get("eval_loss"), int((t.get("pct") or 0) // 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cada", type=int, default=60, help="segundos entre consultas al Hub")
    ap.add_argument("--filtro", default="", help="solo corridas cuyo nombre contenga este texto")
    ap.add_argument("--siempre", action="store_true", help="no terminar cuando acaban las corridas")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    vistas, vigiladas, primera = {}, set(), True
    while True:
        try:
            rutas = [f for f in api.list_repo_files(a.repo)
                     if f.startswith("ejecuciones/") and f.endswith("/estado.json") and a.filtro in f]
            for ruta in rutas:
                corrida = ruta.split("/")[1]
                e = json.load(open(hf_hub_download(a.repo, ruta)))
                if primera and e["estado"] != "corriendo":
                    vistas[corrida] = clave_cambio(e)   # ya terminadas al empezar: no se reportan
                    continue
                vigiladas.add(corrida)
                if vistas.get(corrida) != clave_cambio(e):
                    vistas[corrida] = clave_cambio(e)
                    hora = datetime.datetime.now().strftime("%H:%M")
                    print(f"[{hora}] {corrida}: {resumen(e)}", flush=True)
            primera = False
            pendientes = [c for c in vigiladas if vistas[c][0] == "corriendo"]
            if vigiladas and not pendientes and not a.siempre:
                print("Todas las corridas vigiladas terminaron.", flush=True)
                return
        except Exception as ex:   # un fallo de red no corta el monitor
            print(f"⚠ consulta fallida: {ex}", file=sys.stderr, flush=True)
        time.sleep(a.cada)


if __name__ == "__main__":
    main()
