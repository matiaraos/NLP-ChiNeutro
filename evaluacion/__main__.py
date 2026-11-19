"""Evalúa un CSV de predicciones (correr desde la raíz del repo):

    python -m evaluacion predicciones.csv                 # BLEU, términos retenidos, BERTScore y juez Opus
    python -m evaluacion predicciones.csv --sin-juez      # sin el juez (no necesita Claude Code)

El CSV necesita las columnas frase_chilena y prediccion, más id_mistral si las frases son del dataset.
Escribe junto al CSV: <nombre>_evaluacion.json (resumen) y <nombre>_por_frase.csv (detalle por frase).
"""
import argparse
import json

from .metricas import evaluar


def main():
    ap = argparse.ArgumentParser(prog="python -m evaluacion", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predicciones", help="CSV con frase_chilena y prediccion (y id_mistral)")
    ap.add_argument("--sin-juez", action="store_true", help="no correr el juez Opus")
    ap.add_argument("--sin-bertscore", action="store_true", help="no calcular BERTScore")
    ap.add_argument("--juez-muestra", type=int, default=None, help="juzgar solo N frases al azar (ahorra cuota)")
    ap.add_argument("--claude-bin", default=None, help="binario de Claude Code (por defecto `claude` o $CLAUDE_BIN)")
    a = ap.parse_args()
    resumen, _ = evaluar(a.predicciones, juez=not a.sin_juez, bertscore_activo=not a.sin_bertscore,
                         muestra=a.juez_muestra, claude_bin=a.claude_bin)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
