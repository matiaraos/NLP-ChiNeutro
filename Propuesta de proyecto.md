# Documento de diseño: Traductor de jerga chilena a español neutro

Oct 25, 2026 · @Mario

## Resumen y objetivos

El proyecto ajusta Mistral-7B-Instruct para traducir jerga chilena a español neutro, y su éxito se juzga por el desempeño en un dominio no visto (conversaciones de Reddit Chile), no por las métricas sobre el propio dataset.

**Problema.** El español chileno usa un léxico coloquial (modismos, verbos y expresiones) que los modelos generales traducen o explican de forma irregular. Se busca un modelo pequeño y ajustable en Colab que lo normalice de forma consistente.

**Alcance.** Entrada: frase en español chileno informal. Salida: la misma frase en español neutro, conservando el significado y evitando agregar o quitar información.

**Hipótesis a contrastar:**

1. Un dataset núcleo pequeño y de alta calidad, combinado con datos sintéticos, mejora la traducción frente al modelo base.
2. El entrenamiento en dos fases (datos mezclados, luego solo datos de calidad con LR menor) supera al entrenamiento en una sola fase.
3. El sesgo de usar el mismo modelo para generar y para entrenar es medible y acotable con la evaluación en Reddit.

**Criterio de éxito.** Mejora clara sobre el modelo base (sin ajuste) en el conjunto de Reddit, con revisión humana de una muestra. Los umbrales numéricos quedan por definir (ver Evaluación).

## Datos

El dato es la parte más delicada: tres conjuntos con roles distintos, y el objetivo es medir cuánto aporta cada uno.

| Conjunto | Origen | Rol | Tamaño objetivo |
| --- | --- | --- | --- |
| Diccionario | Scraping de un diccionario de chilenismos en línea (revisión legal ya hecha: sin impedimentos) | Términos, definiciones en español neutro y una frase de ejemplo; insumo del RAG | Según lo disponible |
| Núcleo | Frases en español chileno traducidas a mano por el equipo y amigos | Calidad: fase final del ajuste con LR reducido y base de validación | A definir |
| Sintético | Mistral-7B-Instruct con few-shot + RAG del diccionario | Volumen: fases iniciales del ajuste | Completar hasta 15.000 pares de entrenamiento en total |
| Reddit Chile | Conversaciones de Reddit Chile | Solo evaluación, fuera de dominio; nunca entra al entrenamiento | Muestra a definir |

**Generación sintética.** Para cada término, el modelo recibe como contexto (RAG) su definición y la frase de ejemplo del diccionario, más unos pocos ejemplos del núcleo como few-shot, y produce nuevas frases chilenas con su traducción neutra.

**Controles de calidad propuestos:**

- Deduplicación exacta y aproximada entre sintético, núcleo y diccionario.
- Filtros automáticos: que la frase contenga el término objetivo, que la traducción no conserve la jerga y que la longitud sea razonable.
- Revisión humana de una muestra del sintético para estimar su tasa de error antes de usarlo.
- Etiquetar cada par con su origen (núcleo / sintético) para poder hacer ablaciones.

**Separación de datos.** Los términos y frases usados en validación y prueba se excluyen del sintético para evitar fugas. El núcleo se parte en entrenamiento, validación y prueba; el resultado final se decide con Reddit.

**Pregunta abierta:** cuántas frases núcleo se pueden conseguir de forma realista, ya que de eso depende cuánto pesa la fase final.

## Modelo

Se usa Mistral-7B-Instruct como base y como generador de datos sintéticos, aceptando un riesgo de sesgo que la evaluación en Reddit está diseñada para medir.

**Por qué Mistral-7B.** Es un LLM de poder medio, con conocimiento y capacidad de razonamiento suficientes para entender modismos y reformularlos, y es ajustable con LoRA en el hardware de Colab.

**Formato de entrada.** Se propone un prompt de instrucción fijo (tarea + frase chilena) y una respuesta que contiene solo la traducción, con la plantilla de chat del modelo y la pérdida calculada únicamente sobre la respuesta.

**Riesgo: el generador y el modelo ajustado son el mismo.** Los pares sintéticos reflejan los errores y preferencias estilísticas de Mistral-7B, así que el modelo podría aprender a imitarse en lugar de aprender el español chileno real. Mitigaciones:

- Anclar con el núcleo humano: la fase final usa solo ese dato y ese dato fija la validación.
- Medir el sesgo: comparar el desempeño en el núcleo (cercano al dominio sintético) con el de Reddit (distinto).
- Ablación: entrenar con núcleo solo, con sintético solo y con ambos, para aislar la contribución de cada fuente.

## Finetuning en dos fases

El ajuste usa adaptadores LoRA (o rsLoRA, con QLoRA si la memoria lo exige) para que quepa en Colab, y se entrena en dos fases sucesivas sobre los mismos adaptadores.

1. **Fase 1, mezcla.** Varias épocas sobre núcleo + sintético, con el LR de partida. Busca que el modelo aprenda la tarea y cubra el vocabulario.
2. **Fase 2, calidad.** Pocas épocas solo sobre el núcleo, con un LR menor. Busca corregir el sesgo del sintético y alinear el estilo con las traducciones humanas.

**Variables a explorar.**

| Variable | Opciones | Qué se quiere saber |
| --- | --- | --- |
| Rango r | Varios valores (p. ej. 8, 16, 32, 64; por confirmar) | Capacidad necesaria frente a riesgo de sobreajuste |
| Escalado | LoRA estándar (alfa / r) frente a rsLoRA (alfa / raíz de r) | Si el escalado por raíz de r estabiliza rangos altos |
| Cuantización | LoRA en precisión completa frente a QLoRA | Costo en memoria y su efecto en la calidad |
| Regularización | Dropout de LoRA y decaimiento de pesos | Control del sobreajuste al núcleo pequeño |
| Plan de fases | Una sola fase frente a dos fases | Si la fase final con LR menor aporta |

**Restricciones prácticas.** Colab limita la memoria y la duración de cada sesión, por lo que conviene guardar puntos de control en Drive, fijar semillas y registrar cada configuración. La grilla completa puede ser inviable: se propone empezar con un subconjunto y ampliar según los resultados.

## Evaluación

El resultado final se decide con el desempeño en Reddit (dominio distinto); las evaluaciones por splits sirven para comparar configuraciones y detectar sobreajuste.

| Evaluación | Datos | Métricas | Para qué sirve |
| --- | --- | --- | --- |
| Splits internos | Validación y prueba del núcleo y del sintético, por separado | Perplejidad y métricas de traducción | Comparar configuraciones y detectar sobreajuste |
| Fuera de dominio | Muestra de conversaciones de Reddit Chile | Métricas automáticas y revisión humana | Resultado final y medición del sesgo del sintético |

**Métricas propuestas.** Perplejidad sobre la traducción de referencia; métricas de similitud con la referencia (por ejemplo BLEU, chrF y una métrica semántica como BERTScore) donde haya referencias; y una tasa de jerga residual, es decir, cuántos términos chilenos permanecen en la salida.

**Reddit sin referencias.** Los comentarios no traen traducción, por lo que se propone:

- Revisión humana de una muestra, con una escala simple (significado conservado, jerga eliminada, fluidez).
- Comparación pareada entre el modelo base y el ajustado.
- Como alternativa, usar un LLM como juez, sabiendo que comparte sesgos con el generador.

**Comparaciones obligatorias.** Modelo base sin ajuste; núcleo solo; sintético solo; ambos en una fase; ambos en dos fases. Con ellas se responde si el sintético ayuda y si la fase final con LR menor aporta.

**Pregunta abierta:** el tamaño de la muestra de Reddit y quién hace la revisión humana.

## Riesgos y mitigaciones

| Riesgo | Efecto | Mitigación |
| --- | --- | --- |
| Sesgo por autogeneración (mismo modelo genera y se entrena) | El modelo imita a Mistral y no el uso real | Fase final solo con núcleo; evaluación en Reddit; ablaciones por origen de datos |
| Calidad desigual del sintético | Traducciones erróneas o que conservan jerga | Filtros automáticos y revisión humana de una muestra |
| Núcleo pequeño | Sobreajuste en la fase final | Pocas épocas, LR menor, regularización y validación separada |
| Fuga entre splits | Métricas infladas | Excluir de entrenamiento los términos y frases de validación y prueba |
| Evaluación sin referencias en Reddit | Medición ruidosa | Revisión humana pareada y rúbrica clara |
| Límites de Colab | Entrenamientos interrumpidos y grilla incompleta | Puntos de control en Drive y priorización de experimentos |
| Datos de redes sociales | Privacidad y términos de uso | Anonimizar usuarios y enlaces, no redistribuir el texto crudo y confirmar las condiciones de uso de Reddit |

**Nota sobre licencias.** El scraping del diccionario ya se revisó sin impedimentos legales; conviene dejar por escrito la fuente y la fecha de extracción, y revisar las condiciones de Reddit antes de extraer.

## Próximos pasos

El orden propuesto parte por los datos, porque de ellos depende todo lo demás; las fechas quedan por definir con el equipo.

- [ ] Extraer el diccionario y dejar registro de la fuente y la fecha
- [ ] Armar el núcleo: traducir a mano las frases con el equipo y amigos
- [ ] Definir el formato del prompt y los splits (entrenamiento, validación, prueba)
- [ ] Generar el sintético con Mistral-7B + RAG, filtrarlo y revisar una muestra
- [ ] Medir el modelo base en el núcleo y en una muestra de Reddit (línea base)
- [ ] Extraer y anonimizar la muestra de Reddit Chile
- [ ] Entrenar la fase 1 y la fase 2 con un subconjunto de la grilla de r y escalado
- [ ] Ejecutar las ablaciones (núcleo solo, sintético solo, ambos)
- [ ] Evaluar en Reddit con revisión humana y redactar los resultados

**Preguntas abiertas.** Tamaño del núcleo; tamaño de la muestra de Reddit y responsables de la revisión humana; umbrales de éxito; y qué parte de la grilla de hiperparámetros cabe en el cómputo disponible.
