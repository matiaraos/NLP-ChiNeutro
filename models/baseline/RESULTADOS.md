# Baseline: modelos sin ajuste (split `test`, 397 frases)

Prompt: el mismo del ajuste de Gemma (`models/gemma4/comun.py`): `PROMPT` para Gemma base, `INSTRUCCION_CHAT` para los
instruct y mT0. Generación greedy, primera línea. Juez: Opus 5.5 (`evaluacion/juez.py`, rúbrica 1, effort medium).

| modelo | BLEU | términos retenidos % ↓ | BERTScore F1 | juez sentido (1-5) | juez fluidez (1-5) | juez jerga eliminada % | juez aceptable % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Opus 5.5 (techo) | 77.4 | 1.7 | 0.878 | 4.73 | 4.92 | 98.7 | 93.5 |
| Gemma 4 E4B `-it` | 60.7 | 11.0 | 0.779 | 3.53 | 4.57 | 80.9 | 42.3 |
| Qwen3.5 4B | 51.1 | 36.1 | 0.716 | 2.87 | 3.98 | 49.6 | 16.6 |
| Gemma 4 E4B base | 19.0 | 37.5 | 0.433 | 1.95 | 4.31 | 57.7 | 7.6 |
| mT0 large | 46.1 | 80.6 | 0.623 | 2.39 | 3.94 | 10.8 | 3.0 |

- **Opus es un techo inflado**: reescribió las referencias del dataset y además es el juez.
- **mT0** copia la frase casi tal cual (80 % de los términos chilenos siguen); su BLEU de 46 viene de que la referencia
  repite gran parte de la frase original, por eso BLEU no basta.
- **Gemma base** muchas veces repite la instrucción en vez de traducir: no sabe seguir instrucciones sin ajuste. Su
  versión ajustada usa el mismo prompt, así que la comparación es justa.

Para reproducir:

```bash
uv venv .venv-baseline && VIRTUAL_ENV=.venv-baseline uv pip install -r models/baseline/requirements.txt -r evaluacion/requirements.txt
.venv-baseline/bin/python models/baseline/baseline_sin_ajuste.py   # modelos abiertos (CUDA, mps o CPU)
.venv-baseline/bin/python models/baseline/opus.py                  # Opus vía claude -p
for f in models/baseline/predicciones/predicciones_*_test.csv; do .venv-baseline/bin/python -m evaluacion "$f"; done
```
