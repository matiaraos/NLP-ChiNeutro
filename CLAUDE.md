# NLP-ChiNeutro

Traducción del español de Chile al español neutro con Gemma 4 E4B. Pipeline en `models/gemma4/` (notebooks 00–02 generados por `salidas/gemma4/herramientas/construir_notebooks.py`; no editar los `.ipynb` a mano). Estado y decisiones: `salidas/gemma4/HANDOFF.md` y `salidas/gemma4/HALLAZGOS_DATASET.md`. Documentos de trabajo van como `.md` en la carpeta correspondiente.

## Colab por MCP: nunca bloquear

- **`execute_code` siempre en segundo plano.** El código que se envía debe lanzar el trabajo con `subprocess.Popen(..., start_new_session=True)` o `nohup ... &` y volver de inmediato. Nada de `time.sleep`, bucles de espera, `trainer.train()` ni descargas largas dentro de `execute_code`: la llamada tiene un límite de 600 s, se corta con `DEADLINE_EXCEEDED` y deja la sesión ocupada para las siguientes.
- Los notebooks se corren sin interfaz con `models/gemma4/lanzar.py` (papermill), lanzado en segundo plano en el runtime.
- **Para seguir una corrida, leer solo `estado.json` desde el Hub**, no el runtime: `python3 models/gemma4/monitor_hub.py --repo <repo>` como `Monitor` local (o `run_in_background`). `estado.json` trae estado, celda en curso y el progreso del entrenamiento (paso, loss, eval_loss, ETA) que escribe `comun.callback_progreso`. Solo si el estado indica error, mirar `log.txt` de la corrida en el Hub.
- Cada `execute_code` dirigido a un `runtime` abre una sesión nueva: cerrar con `delete_session` las que se abran.
- El token de Hugging Face se copia al runtime por la API de contenidos (`PUT {url}/api/contents/root/.cache/huggingface/token`) leyendo `~/.cache/huggingface/token` en el mismo comando, sin mostrar su valor.

## Hugging Face Hub

- Repo privado `mariopedemonte12/chineutro-gemma4`: espejo de `DIR_TRABAJO` (`comun.Hub`). `datos/` tiene código y datos de entrada; se actualiza con `models/gemma4/subir_datos_hub.py` después de cambiar `comun.py`, `lanzar.py` o los notebooks.
- `mariopedemonte12/chineutro-gemma4-prueba` y `-prueba2`: repo para pruebas de humo (modelo diminuto en `/content/tiny_g4`); no mezclar con el real.
- **Límite del Hub: 128 commits por hora por repo**, compartido por todas las corridas que suben ahí. Si se supera, las subidas fallan (también los checkpoints) durante ~1 h. `lanzar.py` sube `estado.json` cada 5 min y el log cada 20 min por defecto; no bajar esos intervalos con varias corridas en paralelo. Para pruebas seguidas, usar un repo de prueba aparte.
- **No actualizar `datos/` en el Hub con una cola corriendo**: cada notebook baja `comun.py` y los datos nuevos al empezar, pero papermill ejecuta la copia del notebook que la cola bajó al inicio. Quedan código y notebook desfasados. Cambios de código → relanzar la cola.
- `models/gemma4/encadenar.py` encadena barrido de LR → entrenamiento con el LR recomendado (`paso1`, o `paso2 <condiciones...>`).
