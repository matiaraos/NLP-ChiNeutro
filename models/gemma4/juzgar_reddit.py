"""Agrega el juez Opus a evaluaciones de Reddit ya hechas con `python -m evaluacion <csv> --sin-juez`.

Las métricas automáticas (términos retenidos y BERTScore contra la entrada) se calculan en una GPU; el juez corre
aquí, donde está Claude Code. Para cada CSV de predicciones lee <base>_evaluacion.json y <base>_por_frase.csv,
juzga con evaluacion.juez.juzgar (misma rúbrica, caché y opciones que `python -m evaluacion`) y reescribe ambos
archivos como si se hubiera corrido la evaluación completa.

    python3 models/gemma4/juzgar_reddit.py models/gemma4/predicciones/reddit/predicciones_*_reddit.csv [--muestra N]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluacion.juez import juzgar  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--muestra", type=int, default=None)
    ap.add_argument("--paralelo", type=int, default=3)
    a = ap.parse_args()
    for csv in map(Path, a.csvs):
        base = csv.with_suffix("")
        ruta_json, ruta_pf = Path(f"{base}_evaluacion.json"), Path(f"{base}_por_frase.csv")
        resumen = json.loads(ruta_json.read_text(encoding="utf-8"))
        por_frase = pd.read_csv(ruta_pf)
        por_frase = por_frase[[c for c in por_frase.columns if not c.startswith("juez_")]]
        print(f"== {csv.name}", flush=True)
        r, ver = juzgar(pd.read_csv(csv), muestra=a.muestra, paralelo=a.paralelo)
        resumen = {k: v for k, v in resumen.items() if not k.startswith("juez_")} | r
        por_frase = por_frase.join(ver[["sentido", "jerga", "fluidez", "aceptable", "comentario"]].add_prefix("juez_"))
        ruta_json.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
        por_frase.to_csv(ruta_pf, index=False)
        print(json.dumps(resumen, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
