"""Sube al repo privado del Hub lo que necesitan los notebooks en una sesión de Colab sin interfaz.

    python3 models/gemma4/subir_datos_hub.py --repo usuario/chineutro-gemma4

- datos/: código (comun.py, lanzar.py, notebooks) y datos de entrada. Se vuelve a subir cada vez que cambien.
- raíz del repo: salidas de la exploración (índice del corpus, config del paso 1, muestra, presencia del
  diccionario, hallazgos), si existen en salidas/gemma4.
Necesita un token con permiso de escritura.
"""
import argparse
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS = ["models/gemma4/comun.py", "models/gemma4/lanzar.py", "models/gemma4/encadenar.py",
         "models/gemma4/00_explorar_corpus.ipynb", "models/gemma4/01_paso1_ntp_corpus.ipynb",
         "models/gemma4/01b_barrido_lr.ipynb",
         "models/gemma4/02_paso2_traduccion.ipynb", "models/gemma4/02b_barrido_lr_paso2.ipynb",
         "scraping/data/rag_documentos.jsonl", "data/entrenamiento/dataset_chilenismos_opus.csv",
         "data/entrenamiento/splits/split_oficial.csv"]
EXPLORACION = ["indice_corpus.parquet", "config_paso1.json", "muestra_5000.parquet",
               "presencia_diccionario_f0.02.csv", "00_explorar_corpus_ejecutado.ipynb", "HALLAZGOS_DATASET.md"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--sin-exploracion", action="store_true", help="sube solo datos/")
    a = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(a.repo, private=True, exist_ok=True)
    ops = [CommitOperationAdd(f"datos/{Path(r).name}", str(RAIZ / r)) for r in DATOS]
    if not a.sin_exploracion:
        salidas = RAIZ / "salidas/gemma4"
        ops += [CommitOperationAdd(n, str(salidas / n)) for n in EXPLORACION if (salidas / n).is_file()]
        escaneo = salidas / "escaneo_diccionario"
        ops += [CommitOperationAdd(f"escaneo_diccionario/{f.name}", str(f)) for f in sorted(escaneo.glob("*.json"))]
    for op in ops:
        print("↑", op.path_in_repo)
    api.create_commit(a.repo, operations=ops, commit_message="Datos y código de models/gemma4")
    print(f"Listo: https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
