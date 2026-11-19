"""Evaluación de traducción chileno -> neutro para cualquier modelo. Ver evaluacion/README.md.

    from evaluacion import evaluar
    resumen, por_frase = evaluar("predicciones.csv")            # BLEU, términos retenidos, BERTScore y juez
"""
from .datos import cargar_split
from .metricas import evaluar

__all__ = ["evaluar", "cargar_split"]
