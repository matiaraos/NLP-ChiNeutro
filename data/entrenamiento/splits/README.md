# Split oficial del dataset de pares

`split_oficial.csv` es el split que usa todo el grupo para entrenar y evaluar con `data/entrenamiento/dataset_chilenismos_opus.csv` (3.980 pares chileno → neutro, 796 expresiones). **No se cambia**: si cada modelo usa otro split, los resultados no se pueden comparar.

## Cómo usarlo

El archivo tiene una fila por par: `id_mistral, clave, def_id, split`, donde `split` es `train`, `val` o `test`. Se une al dataset por `id_mistral`:

```python
import pandas as pd
pares = pd.read_csv("data/entrenamiento/dataset_chilenismos_opus.csv", encoding="utf-8-sig")
split = pd.read_csv("data/entrenamiento/splits/split_oficial.csv")[["id_mistral", "split"]]
pares = pares.merge(split, on="id_mistral")
train, val, test = (pares[pares.split == s] for s in ("train", "val", "test"))
```

- **train**: para entrenar.
- **val**: para elegir hiperparámetros, comparar configuraciones y decidir cuándo parar.
- **test**: solo para el resultado final, con la configuración ya elegida. Mirarlo para decidir algo es ajustar sobre test.

## Cómo se construyó

`crear_split_oficial.py` (solo biblioteca estándar, semilla 42, determinista):

1. **Estratificado por expresión.** Cada expresión (`clave`) tiene exactamente 5 pares. De cada una, 4 van a train y 1 va a val o a test. Toda expresión aparece en train.
2. **Mitad a val y mitad a test.** El par que sale de cada expresión va a val o a test, alternando entre expresiones barajadas. Se hace por separado para las expresiones de una palabra y las de varias, así los dos splits quedan con la misma proporción.
3. **El significado también se ve en train.** Algunas expresiones tienen varios significados (`def_id`). El par que sale se elige entre los significados que tienen otro par en train. Solo en las 4 expresiones con 5 significados distintos no se puede, y queda 1 par de val o test con un significado que no está en train.

| | train | val | test |
|---|---|---|---|
| Pares | 3.184 | 399 | 397 |
| Expresiones | 796 (todas) | 399 | 397 |
| Expresiones de varias palabras | 42,8 % | 42,9 % | 42,8 % |
| Pares `corregido` / `reescrito` | 54,2 / 45,8 % | 53,6 / 46,4 % | 54,2 / 45,8 % |
| Palabras por frase chilena (media) | 13,5 | 13,3 | 13,3 |

## Qué mide y qué no

Val y test miden **traducir frases nuevas con expresiones que el modelo ya vio** en entrenamiento, en otro contexto. **No** miden generalización a expresiones nuevas. Eso es intencional: con 5 ejemplos por expresión, aprender una expresión sin haberla visto no es realista. Las expresiones nuevas tendrían que venir del pre-entrenamiento continuado sobre el corpus (paso 1 de Gemma) o del RAG, y eso se evalúa aparte.

Para regenerarlo (debe dar el mismo archivo): `python data/entrenamiento/splits/crear_split_oficial.py`.
