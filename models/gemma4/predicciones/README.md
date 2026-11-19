# Predicciones y evaluación oficial de Gemma 4 E4B

Contiene las 11 configuraciones de la ablación de rango (`salidas/gemma4/ABLACION_RANGO.md`), cada una en val (399 frases) y test (397), evaluadas con el pipeline del grupo `evaluacion/` (rama `evaluacion-generica`, commit `ae8c5b1`): BLEU (sacrebleu 2.4.3), % de términos chilenos retenidos y BERTScore F1 (xlm-roberta-large, reescalado, contra la referencia). La evaluación se corrió el 2026-11-19 en una GPU L4 de Colab.

**Juez** (Opus 5.5, esfuerzo medio, rúbrica 1, igual que la baseline del grupo): solo en test y en tres configuraciones: `gemma4_base_p1r32_p2r8` (elegida en val), `gemma4_base_p2r16` (mejor `solo_paso2` en val) y `gemma4_it_p2r8` (mejor instruct en val). Sus veredictos están en los `_evaluacion.json` y en las columnas `juez_*` de los `_por_frase.csv` de esas tres. Las demás filas del resumen no tienen juez.

## Archivos

- `predicciones_<modelo>_<split>.csv`: `id_mistral, frase_chilena, prediccion`. Son el formato de `evaluacion/` y salen de `paso2/<corrida>/predicciones_<split>.csv` en el Hub (`mariopedemonte12/chineutro-gemma4`).
- `predicciones_<modelo>_<split>_evaluacion.json` y `_por_frase.csv`: la salida de `python -m evaluacion <csv> --sin-juez`.
- `resumen_gemma4.csv` / `.json`: una fila por configuración y split.

## Nombres

| Nombre | Corrida en el Hub | Partida | LR del paso 2 |
|---|---|---|---|
| `gemma4_base_p2r{r}` | `solo_paso2__atencion+mlp_r{r}` | `google/gemma-4-E4B` | 2e-4 |
| `gemma4_it_p2r{r}` | `it_solo_paso2__atencion+mlp_r{r}` | `google/gemma-4-E4B-it` | 7e-4 |
| `gemma4_base_p1r{r1}_p2r{r2}` | `paso1+2__atencion+mlp_r{r2}__p1-mlp_r{r1}_5M_priotw80` | base + paso 1 (FFN, LR 3e-4) fusionado | 2e-4 |

Todas: LoRA en atención + FFN en el paso 2, alpha = 2r, 3 épocas, lote 16, semilla 42, split oficial. La **configuración elegida en val** es `gemma4_base_p1r32_p2r8`. Los números de test no se usaron para elegir.

## Para reproducir

```bash
python -m evaluacion models/gemma4/predicciones/predicciones_gemma4_base_p1r32_p2r8_test.csv --sin-juez
```
