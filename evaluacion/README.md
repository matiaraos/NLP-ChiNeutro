# Evaluación

Evalúa las traducciones de **cualquier modelo** con cuatro métricas: BLEU, % de términos chilenos retenidos,
BERTScore y un juez LLM (Claude Opus 5.5 vía Claude Code). No necesita el modelo, solo un CSV con sus traducciones.

## 1. Quien entrena: entregar un CSV

Traduce las frases del split oficial y guarda un CSV con estas columnas:

| columna | contenido |
|---|---|
| `id_mistral` | id del par (viene en el split) |
| `frase_chilena` | la frase de entrada |
| `prediccion` | la traducción de tu modelo |

```python
import sys; sys.path.insert(0, "NLP-ChiNeutro")        # raíz del repo clonado
from evaluacion import cargar_split

test = cargar_split("test")                             # o "val" para comparar configuraciones
test["prediccion"] = [traducir(f) for f in test.frase_chilena]   # tu modelo
test[["id_mistral", "frase_chilena", "prediccion"]].to_csv("predicciones_mimodelo_test.csv", index=False)
```

Para frases que no son del dataset (p. ej. Reddit) basta con `frase_chilena` y `prediccion`.

## 2. Evaluar

Desde la raíz del repo, una vez:

```bash
pip install -r evaluacion/requirements.txt
```

Y para cada CSV:

```bash
python -m evaluacion predicciones_mimodelo_test.csv
```

Escribe junto al CSV:
- `predicciones_mimodelo_test_evaluacion.json`: el resumen.
- `predicciones_mimodelo_test_por_frase.csv`: las métricas y el veredicto del juez por frase.

Opciones:

| opción | para qué |
|---|---|
| `--sin-juez` | si no tienes Claude Code; calcula las otras tres métricas |
| `--juez-muestra N` | juzgar solo N frases al azar (ahorra cuota) |
| `--sin-bertscore` | omitir BERTScore (en CPU tarda unos minutos) |
| `--claude-bin RUTA` | si `claude` no está en el PATH (también `CLAUDE_BIN`) |

También desde Python: `from evaluacion import evaluar; resumen, por_frase = evaluar("archivo.csv", juez=False)`.

## Qué mide cada métrica

| campo | qué es |
|---|---|
| `bleu` | BLEU de corpus contra la referencia (`sacrebleu`). Solo si hay referencia |
| `terminos_retenidos_%` | De las expresiones chilenas detectadas en la entrada, % que sigue en la traducción. **Más bajo es mejor** |
| `bertscore_f1` | Similitud semántica (`xlm-roberta-large`, reescalado). Contra la referencia, o contra la entrada si no hay (`bertscore_contra`) |
| `juez_sentido_media` | Rúbrica 1–5: ¿conserva el significado? |
| `juez_fluidez_media` | Rúbrica 1–5: ¿es español natural y correcto? |
| `juez_jerga_eliminada_%` | % de traducciones sin chilenismos, según el juez |
| `juez_aceptable_%` | % que un editor publicaría tal cual |

**Términos retenidos.** Con datos del dataset se busca la expresión de cada par (`clave`); sin ella se busca todo el
diccionario (`scraping/data/rag_documentos.jsonl`), salvo las palabras que también son neutras
(`expresiones_ambiguas.json`: *taco*, *rico*, *harto*…). El detector (`jerga.py`) reconoce conjugaciones, género y
número (*cranear* → *craneando*, *flaite* → *flaites*, *cachar* → *cachai*): encuentra la expresión en el 89 % de las
frases chilenas del dataset (la búsqueda literal, 69 %) con 0,4 % de falsos positivos.

**Juez.** Necesita Claude Code instalado y con sesión iniciada (`claude`, luego `/login`). Evalúa en lotes de 20
frases con una rúbrica fija (`juez.py`) y no sabe qué modelo generó cada traducción. Los veredictos quedan en
`evaluacion/cache_juez.jsonl`: repetir una evaluación no gasta cuota, y si se corta, volver a correr retoma lo que
falta. Opus reescribió las referencias del dataset, así que conviene calibrarlo con evaluación humana en una muestra.
