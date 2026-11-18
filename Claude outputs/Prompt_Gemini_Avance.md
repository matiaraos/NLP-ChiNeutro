Necesito que crees una presentación de PowerPoint (.pptx) para el video de avance de nuestro proyecto del curso IIC3670 Procesamiento del Lenguaje Natural (Pontificia Universidad Católica de Chile). El video dura como máximo 5 minutos, así que la presentación debe tener unas 10 diapositivas, con poco texto por diapositiva y un elemento visual en cada una.

## Archivos adjuntos

1. `ChiNeutro - Presentación de Propuesta.pptx`: nuestra presentación anterior. **El nuevo PPT debe copiar su estilo visual**: misma paleta, tipografía (Plus Jakarta Sans), tamaños, tarjetas blancas con borde gris claro y esquinas redondeadas sobre fondo gris muy claro, "eyebrow" en mayúsculas azul sobre cada título y portada y cierre con fondo azul oscuro.
2. `Propuesta de proyecto.md`: documento de diseño del proyecto.
3. `Plan de trabajo.md`: plan del equipo, incluido el feedback que recibimos en la propuesta y cómo lo incorporamos. Ojo: lo que diga este archivo sobre un "núcleo" de frases traducidas a mano ya no aplica (ver abajo).

## Lo que pide el curso

"El avance del proyecto debe ser coherente con la propuesta. Se espera que al menos cuente con la etapa de recolección de datos y con el diseño experimental (códigos/herramientas). Idealmente, resultados preliminares. Es importante que incorpore el feedback recibido en la propuesta si corresponde."

Temario mínimo, en este orden: Graphical abstract, Recolección de datos, Diseño experimental, Resultados experimentales, Próximos pasos.

## El proyecto (ChiNeutro)

Ajustar un modelo de lenguaje para traducir jerga chilena a español neutro, conservando el significado. La propuesta usaba Mistral-7B-Instruct, pero **no nos sirvió** (ver resultados) y vamos a cambiar a modelos que manejen mejor el español. El éxito se juzga en un dominio no visto (conversaciones de Reddit Chile, solo para evaluación). **No habrá núcleo de frases traducidas a mano**: el entrenamiento usa el corpus reparado.

## Avance real (usa estas cifras tal cual, no inventes otras)

**Recolección de datos (scraping de diccionariochileno.cl, ya hecho):**
- 3.172 términos en el catálogo del sitio; el equipo seleccionó a mano 806 (se quitaron nombres propios, términos de internet no chilenos y duplicados).
- Base RAG final: 802 términos con 1.621 definiciones, todas con un ejemplo real escrito por un chileno; definiciones ordenadas por votos, con fuente y fecha de extracción.
- Permisos: robots.txt solo prohíbe /admin y /vote; no se guardan seudónimos de autores. Extracción a 1 solicitud cada 3 s, reanudable, detenida ante errores 429 o del servidor. 1.001 solicitudes registradas, 100 % exitosas.

**Corpus paralelo sintético (ya hecho):**
- Etapa A, generación con Mistral-7B-Instruct v0.3 (cuantizado en 4 bits): RAG con hasta 4 definiciones por votos y su ejemplo; 5 pares (frase chilena, frase neutral) por término, en JSON. Resultado: 4.010 pares. Filtro léxico (la chilena contiene la expresión, la neutral no la conserva, largo razonable): 77,1 % lo pasa.
- Etapa B, reparación con Claude Opus 5.5: revisa cada par contra las definiciones del RAG, usa un solo sentido por par y elimina todos los chilenismos de la frase neutral; lotes de 8 términos con salida validada por JSON Schema. Resultado: 3.980 pares finales de 796 términos.
- Par de ejemplo (para el graphical abstract): "Me salió una pega en Antofagasta, así que me voy a vivir pa'l norte." → "Me salió un trabajo en Antofagasta, así que me voy a vivir al norte."

**Código (solo una pincelada: bloques de 4-5 líneas, fuente monoespaciada y fondo oscuro, sin saturar las diapositivas):**
- `scraping/scraper.py`: `for url in pendientes:` / `registro = descargar(cliente, url)` (pausa de 3 s) / `paginas[url] = registro` (ante 429 se detiene) / `f.write(json.dumps(registro) + "\n")` (reanudable).
- Notebook de Mistral: `prompt = construir_prompt(doc, k=5)` (definiciones + ejemplos) / `salida = model.generate(**enc, do_sample=True, temperature=0.8, top_p=0.9)` / `for obj in extraer_objetos(salida):` / `fila = validar_par(obj, doc, defs)` (filtros).
- Reparación (`models/claude_opus/reparar_dataset.py`): `cmd = ["claude", "-p", "--model", "claude-opus-5-5", "--system-prompt", SYSTEM_PROMPT, "--json-schema", json.dumps(SCHEMA)]` / `proc = subprocess.run(cmd, input=prompt_lote(...))` / `pares = validar(...)` (si falla, se reintenta el lote).

**Resultados preliminares: Mistral-7B no nos sirvió (enfatízalo):**
- Ejemplo del notebook, "fierro pelao" (significado: tener relaciones sexuales sin condón). Mistral generó la chilena "Tú que siempre eres tan cautelo, ¿por qué no lo hiciste antes?" y la neutral "Tú que siempre eres tan cauteloso, ¿por qué no lo hiciste antes?". No usa la expresión ni tiene relación con su significado; solo "corrige" cautelo por cauteloso.
- 46 % de los pares de Mistral tuvo que reescribirse desde cero; 26 % usaba la expresión con el sentido equivocado; 77 % pasaba igual el filtro léxico (los filtros automáticos no detectan estos errores).
- Problemas en los 4.010 pares, según las notas del revisor (Opus), por palabras clave; un par puede tener varios: la neutral cambia el significado 31,9 %; sentido equivocado 25,8 %; incoherente o poco natural 22,5 %; error gramatical 13,5 %; no usa la expresión 13,0 %. Muéstralo como gráfico de barras horizontales.
- Conclusión: Mistral-7B no domina el español chileno, ni como generador ni como base; cambiamos a modelos con mejor español.

**Diseño experimental (definido, aún no ejecutado):**
- Fine-tuning en dos fases: fase 1 con el corpus reparado completo (varias épocas, LR de partida); fase 2 con un subconjunto de mayor calidad (pocas épocas, LR menor).
- Variables: arquitectura (encoder-decoder vs LLM); rango r (8, 16, 32, 64); LoRA vs rsLoRA; LoRA vs QLoRA 4 bits; una fase vs dos fases.
- Comparaciones: modelo sin ajuste vs ajustado; mT5/mT0 vs Qwen3.5/Gemma 4; sintético crudo (Mistral) vs reparado (Opus); una fase vs dos fases.
- Evaluación: splits internos por término sobre el corpus reparado (definidos antes de entrenar, para evitar fugas) y Reddit Chile fuera de dominio (sin referencias: juez + rúbrica humana de significado, jerga eliminada y fluidez, y comparación pareada).

**Feedback de la propuesta y cómo lo incorporamos (tabla de 2 columnas):**
- Justificar que el modelo debe entender modismos → medimos a Mistral-7B: no entendía los chilenismos, así que cambiamos a modelos con mejor español.
- Justificar las métricas → BLEU y ROUGE-L pasan a ser secundarias porque miden coincidencia de palabras, no significado.
- Umbral mínimo para BERTScore → se fija comparando pares correctos con incorrectos conocidos (traducciones de Mistral que hubo que corregir).
- LLM-as-a-Judge distinto del modelo entrenado → juez de otra familia; compara salida vs referencia (prueba) y fuente vs salida (Reddit).
- Evaluación humana si no hay juez → hacemos ambas: rúbrica humana en una fracción y medición del acuerdo juez-humano.

**Modelos candidatos (diapositiva propia en Próximos pasos):**
- Opción principal, encoder-decoder: mT5-large o mT0-large (~1,2B). La tarea es corta, de entrada a salida y fija, justo para lo que sirve un encoder-decoder; es rápido, barato y "inventa" menos que un LLM conversacional.
- Alternativa, LLM pequeño con LoRA: Qwen3.5-2B/4B o Gemma 4 E4B. Probablemente entiende mejor el contexto y los chilenismos menos comunes, a cambio de más costo de inferencia.
- Cómo elegimos: primero medimos el dominio del español de cada candidato sin ajuste; luego ajustamos con el corpus reparado y comparamos contra Mistral-7B como referencia.

**Plan (próximos pasos, con fechas):**
- 11-14 nov: probar el dominio del español de mT5/mT0, Qwen3.5 y Gemma 4 sin ajuste.
- 11-14 nov: splits por término, juez LLM y umbral de BERTScore.
- 14-18 nov: fine-tuning en dos fases y grilla repartida entre las GPUs del equipo.
- 14-18 nov: extraer y anonimizar la muestra de Reddit Chile (usuarios y enlaces).
- 18-21 nov: ablaciones y evaluación en Reddit (juez + humanos, calibración juez-humano).
- 25 nov: entrega final (informe, código y dataset).

## Estructura sugerida

1. Portada (fondo oscuro): título "ChiNeutro: Traducción de Dialecto Chileno a Español Neutro", subtítulo de avance, curso, universidad, "Grupo ChiNeutro" y los integrantes (Mario Pedemonte, Agustin Consolo, Matias Araos, Danko Leiva).
2. Graphical abstract: ejemplo de entrada chilena → salida neutra, y debajo el pipeline de 5 pasos (Diccionario, Generación con Mistral, Reparación con Opus, Fine-tuning con modelos de mejor español, Evaluación) con una etiqueta de estado en cada uno (Listo / Diseñado).
3. Recolección de datos I: scraping (cifras grandes, método y bloque de código del scraper).
4. Recolección de datos II: corpus sintético (etapas A y B, y bloques de código de generación y reparación).
5. Diseño experimental I: fine-tuning en dos fases, tabla de variables y comparaciones.
6. Diseño experimental II: evaluación y feedback incorporado.
7. Resultados preliminares: "Mistral-7B no nos sirvió", con el ejemplo de "fierro pelao", el gráfico de barras y las cifras.
8. Próximos pasos I: modelos candidatos.
9. Próximos pasos II: línea de tiempo con fechas.
10. Cierre (fondo oscuro): mensaje final, hitos Avance (11/11) y Entrega Final (25/11), agradecimiento.

## Requisitos

- Todo el texto en español de Chile, con decimales con coma (77,1 %).
- Títulos de 27 pt aprox. y texto de cuerpo de al menos 11 pt; nada de texto que se salga de su caja.
- Gráficos nativos de PowerPoint (editables), no imágenes.
- No agregues cifras, resultados ni afirmaciones que no estén en este mensaje o en los archivos adjuntos.
- No pongas el guion en las diapositivas; lo pediremos aparte.
- Entrega el archivo .pptx descargable.
