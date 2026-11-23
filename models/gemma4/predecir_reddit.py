"""Traduce las frases de Reddit Chile (evaluación fuera de dominio) con configuraciones ya entrenadas de Gemma 4 E4B.

Pensado para correr sin interfaz en un runtime de Colab con GPU, lanzado en segundo plano:
    nohup python3 predecir_reddit.py [--limite 16] [--configs gemma4_base_p1r32_p2r8 ...] > log.txt 2>&1 &

Baja comun.py de datos/ del repo del Hub, las frases de `mariopedemonte12/chineutro-reddit` y los adaptadores de
paso1/ y paso2/. Carga cada configuración igual que el notebook 02 (base + paso 1 fusionado si corresponde + adaptador
del paso 2) y genera con comun.traducir (greedy, mismo prompt, primera línea). Sube a reddit/ del repo:
    predicciones_<config>_reddit.csv   (id, frase_chilena, prediccion): formato de `python -m evaluacion`
y va dejando ejecuciones/predecir_reddit[_prueba]/estado.json para seguirlo con monitor_hub.py.
"""
import argparse
import datetime
import json
import sys
import time
import traceback
from pathlib import Path

REPO = "mariopedemonte12/chineutro-gemma4"
REPO_REDDIT = "mariopedemonte12/chineutro-reddit"
DIR = Path("/content/chineutro")
PASO1 = "paso1/mlp_r32_5M_priotw80/adaptador"
CONFIGS = {   # nombre en models/gemma4/predicciones -> (modelo base, adaptador del paso 1, corrida del paso 2, chat)
    "gemma4_base_p1r32_p2r8": ("google/gemma-4-E4B", PASO1, "paso1+2__atencion+mlp_r8__p1-mlp_r32_5M_priotw80", False),
    "gemma4_base_p2r16": ("google/gemma-4-E4B", None, "solo_paso2__atencion+mlp_r16", False),
    "gemma4_it_p2r8": ("google/gemma-4-E4B-it", None, "it_solo_paso2__atencion+mlp_r8", True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--limite", type=int, default=None, help="solo las primeras N frases (prueba de humo)")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    nombre_ej = "predecir_reddit" + ("_prueba" if a.limite else "")
    estado = {"estado": "corriendo", "notebook": "predecir_reddit.py", "parametros": vars(a),
              "inicio_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "hechas": []}

    def reportar(**cambios):
        estado.update(cambios, actualizado_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        try:
            api.upload_file(repo_id=REPO, path_or_fileobj=json.dumps(estado, ensure_ascii=False, indent=1).encode(),
                            path_in_repo=f"ejecuciones/{nombre_ej}/estado.json",
                            commit_message=f"{nombre_ej}: {estado['estado']} {estado.get('celda', '')}")
        except Exception as e:  # noqa: BLE001
            print("No pude subir estado.json:", e, flush=True)

    try:
        reportar(celda="preparando")
        snapshot_download(REPO, allow_patterns=["datos/comun.py"], local_dir=DIR)
        sys.path.insert(0, str(DIR / "datos"))
        import comun
        import pandas as pd
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        frases = json.loads(Path(hf_hub_download(REPO_REDDIT, "reddit_chile.json", repo_type="dataset"))
                            .read_text(encoding="utf-8"))["frases"]
        df = pd.DataFrame({"id": [f["id"] for f in frases], "frase_chilena": [f["frase"] for f in frases]})
        if a.limite:
            df = df.head(a.limite)
        print(f"{len(df)} frases | GPU: {torch.cuda.get_device_name(0)}", flush=True)

        for nombre in a.configs:
            base, paso1, corrida, chat = CONFIGS[nombre]
            reportar(celda=nombre)
            t0 = time.time()
            patrones = [f"paso2/{corrida}/adaptador/*"] + ([f"{paso1}/*"] if paso1 else [])
            snapshot_download(REPO, allow_patterns=patrones, local_dir=DIR)
            tok = AutoTokenizer.from_pretrained(base)
            modelo = comun.cargar_modelo(base, adaptador_a_fusionar=str(DIR / paso1) if paso1 else None)
            modelo = PeftModel.from_pretrained(modelo, str(DIR / "paso2" / corrida / "adaptador")).eval()
            modelo.config.use_cache = True
            pred = comun.traducir(modelo, tok, df.frase_chilena.tolist(), bs=a.bs, chat=chat)
            salida = df.assign(prediccion=pred)
            ruta = DIR / "reddit" / f"predicciones_{nombre}_reddit{'_prueba' if a.limite else ''}.csv"
            ruta.parent.mkdir(parents=True, exist_ok=True)
            salida.to_csv(ruta, index=False)
            api.upload_file(repo_id=REPO, path_or_fileobj=ruta, path_in_repo=f"reddit/{ruta.name}",
                            commit_message=f"Predicciones Reddit: {nombre}")
            vacias = int((salida.prediccion.str.strip() == "").sum())
            copia = float((salida.prediccion.map(comun.norm) == salida.frase_chilena.map(comun.norm)).mean())
            estado["hechas"].append({"config": nombre, "segundos": round(time.time() - t0), "vacias": vacias,
                                     "copia_literal_%": round(100 * copia, 1)})
            print(estado["hechas"][-1], flush=True)
            del modelo
            torch.cuda.empty_cache()
        reportar(estado="terminado", celda="fin", codigo_salida=0)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        reportar(estado="error", codigo_salida=1, error=traceback.format_exc()[-1500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
