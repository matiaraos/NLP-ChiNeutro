# Plan de trabajo: Traductor de jerga chilena a español neutro

Nov 2, 2026 · Equipo: Mario, Agustín, Martín, Danko, Antonio

## Fechas clave

| Fecha | Hito |
| --- | --- |
| Lun 2 nov | Inicio del plan |
| **Mar 10 nov, 23:59** | **Subir video de avance (5 min) a Canvas** (meta interna: subirlo el 10 a las 18:00) |
| Mié 11 nov, bloques 3-4 | Presentación del avance en clase |
| **Mié 25 nov** | **Entrega final** (por confirmar el formato exacto: informe, código, etc.) |

**Qué exige el avance:** etapa de recolección de datos, diseño experimental (código y herramientas) e idealmente resultados preliminares, coherente con la propuesta e incorporando el feedback.

## Cómo incorporamos el feedback

| Feedback | Qué haremos | Dónde queda |
| --- | --- | --- |
| Justificar por qué el modelo debe entender modismos chilenos | Sección corta que justifique Mistral-7B (tamaño, ajustable en Colab, casos de uso de normalizar texto coloquial) y mida cuánto entiende ya el modelo base | Línea base (T-E1) y video |
| Justificar las métricas | Tabla que explique qué mide cada métrica y su límite. BLEU/ROUGE-L miden coincidencia de palabras, no significado, así que pasan a ser secundarias | Diseño de evaluación (T-D1) |
| BERTScore necesita un umbral mínimo justificado | Calcular BERTScore entre pares correctos y pares incorrectos conocidos del núcleo, y fijar el umbral con esa distribución | T-D3 |
| Usar LLM-as-a-Judge con un modelo distinto al entrenado | Juez distinto de Mistral (API gratuita o modelo local mayor). Se compara la generación con la referencia (núcleo) y la fuente con la salida (Reddit) | T-D2 |
| Si no hay juez, evaluación humana parcial | Haremos ambas: evaluación humana en una fracción y **calibración del juez contra los humanos** (acuerdo juez-humano), que responde la duda de si el sentido se conserva | T-D4 |

## Reparto: modelo híbrido

Cada persona **lidera un área** (colunna "Líder") y todas las **tareas transversales se reparten por igual**. Las asignaciones de área son una propuesta y se pueden intercambiar; lo que importa es que cada área tenga un líder y un respaldo.

| Área | Líder | Respaldo | Responsabilidad |
| --- | --- | --- | --- |
| A. Diccionario y curación de datos | Mario | Agustín | Scraping, limpieza, selección de términos y ejemplos, splits |
| B. Datos sintéticos | Agustín | Martín | RAG, prompts few-shot, generación, filtros, deduplicación |
| C. Pipeline de ajuste | Martín | Danko | LoRA/rsLoRA/QLoRA, dos fases, registro de experimentos |
| D. Evaluación | Danko | Antonio | Métricas, umbral BERTScore, juez LLM, rúbrica humana, calibración |
| E. Reddit e infraestructura | Antonio | Mario | Fuente y anonimización de Reddit, repo, formato común, línea base, coordinación del video e informe |

**Tareas transversales (iguales para las 5 personas):**

- Escribir ~50 frases núcleo a mano con su traducción neutra (≈250 en total).
- Revisar 1/5 de la muestra humana del sintético y 1/5 de la muestra humana de las salidas del modelo.
- Escribir la sección del video y del informe que corresponda a su área.
- Correr su porción de la grilla de experimentos en su propia GPU.

## Fase 0: arranque (2-3 nov)

| ID | Tarea | Responsable |
| --- | --- | --- |
| T-0.1 | Reunión de arranque: confirmar áreas, calendario y hardware de cada integrante (GPU y VRAM) | Todos |
| T-0.2 | Crear repo, estructura de carpetas, formato único de datos y plantilla de prompt | Antonio |
| T-0.3 | Definir guía de traducción manual: qué es "español neutro", qué hacer con insultos, expresiones sin equivalente y frases ambiguas | Todos (borrador: Mario) |
| T-0.4 | Decidir el tamaño de la muestra humana (sugerencia inicial: 100 salidas de Reddit y 60 del núcleo, repartidas) | Danko |

## Fase 1: datos (3-6 nov)

| ID | Tarea | Responsable | Entregable |
| --- | --- | --- | --- |
| T-A1 | Inspeccionar diccionariochileno.cl: se organiza por letra (`/terms/a`…`/terms/z`) y por término (`/term/<palabra>`). Confirmar qué campos tiene cada entrada y si tiene frase de ejemplo | Mario | Nota con campos disponibles |
| T-A2 | Scraper con pausas entre solicitudes; guardar fuente y fecha de extracción | Mario | Datos crudos + registro |
| T-A3 | Limpiar y seleccionar términos. El sitio es escrito por la comunidad y hay términos con varias definiciones, así que se necesitan criterios: definición clara, con ejemplo, popularidad, sin duplicados. **Criterio sobre términos vulgares u ofensivos: por decidir** | Mario + Agustín | Lista curada de términos |
| T-A4 | Definir splits por término (entrenamiento, validación, prueba) **antes** de generar sintético, para evitar fugas | Mario | Archivo de splits |
| T-N1 | Cada persona escribe ~50 frases núcleo (con términos distintos, repartidos por letra para no repetir) | Todos | ~250 pares |
| T-N2 | Revisión cruzada: cada par lo revisa otra persona | Todos | Núcleo validado |
| T-R1 | Investigar fuentes para Reddit Chile (API, dataset existente en Hugging Face/Kaggle, scraping), comparar términos de uso, esfuerzo y volumen | Antonio | Recomendación en 1 página |
| T-R2 | Definir método de anonimización (usuarios, enlaces) y tamaño de la muestra | Antonio + Danko | Protocolo |

## Fase 2: sintético, línea base y diseño experimental (6-9 nov)

| ID | Tarea | Responsable | Entregable |
| --- | --- | --- | --- |
| T-B1 | Montar el RAG: por término, definición y ejemplo del diccionario como contexto | Agustín | Código |
| T-B2 | Prompt few-shot usando ejemplos del núcleo (solo los de entrenamiento) | Agustín | Plantilla |
| T-B3 | Piloto: ~200 pares sintéticos; filtros (contiene el término, no conserva jerga, longitud) y deduplicación | Agustín + Martín | Piloto filtrado |
| T-B4 | Revisión humana del piloto para estimar la tasa de error antes de generar a escala | Todos (40 pares c/u) | Tasa de error |
| T-C1 | Script de entrenamiento: plantilla de chat, pérdida solo sobre la respuesta, LoRA/QLoRA, semillas fijas, checkpoints en Drive | Martín | Script probado con datos pequeños |
| T-C2 | Verificar que el pipeline corre en la GPU más modesta del equipo (ver notas de hardware) | Martín + Danko | Confirmación |
| T-D1 | Redactar el diseño de evaluación y justificar cada métrica (ver feedback) | Danko | 1 página |
| T-D2 | Elegir y probar el juez LLM (modelo distinto de Mistral; evaluar capas gratuitas por API o modelo local); redactar el prompt del juez | Danko + Antonio | Juez funcionando |
| T-D3 | Fijar umbral de BERTScore con la distribución de pares correctos e incorrectos del núcleo | Danko | Umbral justificado |
| T-E1 | **Línea base:** Mistral-7B sin ajuste sobre el núcleo (y sobre Reddit si ya hay muestra), con métricas automáticas y juez | Antonio + Danko | Resultados preliminares |

## Video y presentación (9-11 nov)

| Fecha | Tarea | Responsable |
| --- | --- | --- |
| 9 nov | Guion de 5 minutos: problema, datos, diseño experimental, feedback incorporado, resultados preliminares. Una sección por área | Antonio coordina; cada líder escribe la suya |
| 9-10 nov | Grabación y edición | Antonio + 1 persona rotativa |
| 10 nov | Revisión final y **subida a Canvas antes de las 18:00** | Todos |
| 11 nov | Presentación y preguntas. Cada líder prepara respuestas de su área | Todos |

## Fase 3: experimentos (11-21 nov)

| ID | Tarea | Responsable |
| --- | --- | --- |
| T-B5 | Generar el sintético completo hasta 15.000 pares de entrenamiento, **repartido en 5 partes** (una por GPU) y luego fusionar | Agustín coordina; cada uno corre su parte |
| T-B6 | Filtros, deduplicación y revisión humana de una muestra del sintético completo | Agustín + todos |
| T-C3 | Fase 1 (núcleo + sintético) y fase 2 (solo núcleo, LR menor) con LoRA | Martín |
| T-C4 | **Grilla repartida en 5 porciones**, una por persona (rango r, LoRA vs rsLoRA, QLoRA, dropout), empezando por un subconjunto | Todos |
| T-C5 | Ablaciones: núcleo solo, sintético solo, ambos en una fase, ambos en dos fases | Todos (repartidas) |
| T-R3 | Extraer y anonimizar la muestra de Reddit | Antonio |
| T-D4 | Evaluación en Reddit: juez + evaluación humana (rúbrica: significado conservado, jerga eliminada, fluidez) + comparación pareada base vs ajustado; **calibración del juez contra los humanos** | Danko coordina; todos evalúan |
| T-D5 | Análisis del sesgo de autogeneración: núcleo vs Reddit | Danko + Martín |

## Fase 4: cierre (21-25 nov)

| Fecha | Tarea | Responsable |
| --- | --- | --- |
| 21-22 nov | Tablas, gráficos y análisis de resultados | Danko + Martín |
| 22-24 nov | Redacción del informe: cada líder escribe su sección | Cada líder |
| 24 nov | Revisión cruzada y consistencia | Todos |
| 25 nov | **Entrega final** | Antonio |

## Cuadro de carga (equilibrio)

| Persona | Área líder | Respaldo | Transversales |
| --- | --- | --- | --- |
| Mario | A. Diccionario | E | 50 frases, 1/5 revisiones, 1/5 grilla |
| Agustín | B. Sintético | A | 50 frases, 1/5 revisiones, 1/5 grilla |
| Martín | C. Ajuste | B | 50 frases, 1/5 revisiones, 1/5 grilla |
| Danko | D. Evaluación | C | 50 frases, 1/5 revisiones, 1/5 grilla |
| Antonio | E. Reddit e infra | D | 50 frases, 1/5 revisiones, 1/5 grilla |

Las áreas A y E pesan más en la primera semana, y C y D más en la segunda. Para compensar, después del 11 de nov se puede **intercambiar el respaldo** o mover tareas entre personas.

## Notas sobre hardware

- Mistral-7B con QLoRA (4 bits) cabe en una T4 de 16 GB. Una RTX 3070 tiene normalmente 8 GB de VRAM, así que ahí **solo cabría QLoRA con secuencias cortas**; confirmar la VRAM de cada integrante en T-0.1.
- La grilla con LoRA en precisión completa probablemente solo corra en las GPUs más grandes; ver T-C2.
- Los límites de sesión de Colab exigen checkpoints frecuentes en Drive.

## Decisiones pendientes

1. Criterio para términos vulgares u ofensivos del diccionario.
2. Fuente de Reddit (T-R1) y tamaño de la muestra.
3. Qué juez LLM usar y con qué acceso (T-D2).
4. Formato de la entrega final del 25 de nov.
5. Horas semanales realistas de cada integrante (no se informaron; el plan reparte carga relativa, no horas).
6. VRAM y GPU de cada integrante.
7. Cómo se evalúa el núcleo si ya se usa para entrenar la fase 2: definir con claridad la parte de núcleo reservada para validación y prueba.

## Riesgos específicos

| Riesgo | Mitigación |
| --- | --- |
| Solo 8 días hasta el video | Priorizar datos, línea base y diseño; el entrenamiento completo puede quedar para después del 11 |
| Calidad desigual del diccionario (escrito por la comunidad) | Criterios de selección y revisión humana |
| Humanos evaluando frases de su propia autoría | Que revisen salidas que no escribieron |
| Juez LLM con límites de uso en capas gratuitas | Probar pronto y guardar las salidas ya evaluadas |
