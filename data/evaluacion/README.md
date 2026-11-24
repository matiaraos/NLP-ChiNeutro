# Reddit Chile: conjunto de evaluación fuera de dominio

`reddit_chile.json` tiene **1.900 frases** de comentarios de r/chile (1.364) y r/RepublicadeChile (536), de 2014 a
2026. Cada una contiene al menos una expresión del diccionario de trabajo (`scraping/data/rag_documentos.jsonl`).
Solo sirve para evaluar: nunca entra al entrenamiento. No trae traducción de referencia.

## Cómo se armó

Script: `scraping/reddit_muestreo.py`, en la rama `reddit-muestreo`.

1. **Recolección.** Se usó la API pública de [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift), un
   archivo de Reddit. Su búsqueda de texto se corta por tiempo, así que se muestreó al azar: 1.500 solicitudes, cada
   una con los 100 comentarios siguientes a un instante uniforme. El 70 % de las solicitudes fue a r/chile (desde
   2014) y el 30 % a r/RepublicadeChile (desde mediados de 2020). En total se leyeron 143.074 comentarios, entre el
   2026-11-19 y la noche del mismo día, con una rutina de Claude Code en la nube.
2. **Extracción.** Se descartaron los bots y los comentarios borrados, y se quitaron las citas y el markdown. Se
   anonimizaron usuarios, correos, RUT y teléfonos. Cada comentario se partió en oraciones y se buscaron las 823
   formas normalizadas del diccionario, con coincidencia exacta por palabras completas. Se dejaron oraciones de 5 a
   40 palabras, sin URL y en español, y se eliminaron las repetidas. Quedaron 20.379 candidatas.
3. **Selección balanceada.** Se tomó como máximo 5 frases por expresión, en rondas, empezando por las de varias
   palabras y las más raras; dentro de cada expresión se prefirió el mayor score de Reddit. El comando fue
   `extraer --objetivo 2000 --tope 5`.

Campos por frase: `id`, `comentario_id`, `subreddit`, `fecha`, `score`, `frase`, `expresiones`
(clave, término y forma encontrada) y `contexto` (comentario completo, limpio y anonimizado).

**`reddit_chile.json` no está en GitHub** (repo público): se baja del repo privado de Hugging Face
`mariopedemonte12/chineutro-reddit` (pedir acceso), junto con el crudo (143 mil comentarios). Está en `.gitignore`. Ahí también queda la versión anterior (tope 3, 1.289 frases) en el historial.

## Cobertura del diccionario

`cobertura_diccionario_reddit.csv`: una fila por expresión (802), con `comentarios_en_crudo`,
`frases_en_reddit_chile`, `textos_corpus_chileno_est` (frecuencia estimada en el Chilean Spanish Corpus, de
`contar_diccionario.py`) y `estado`:

| estado | expresiones | qué significa |
|---|---:|---|
| `en_frases` | 519 (161 de varias palabras) | aparece en al menos una frase del conjunto |
| `en_crudo_filtrada` | 26 | aparece en 1 a 3 comentarios, pero sus oraciones no pasaron los filtros |
| `ausente` | 257 | no aparece en ningún comentario del crudo; 133 de ellas tampoco salen en el corpus chileno |

Entre las ausentes más frecuentes en el corpus chileno están *de cajón*, *sacar pica*, *a poto pelao*,
*sacar la cresta*, *aperrado*, *atorrante*, *pelolais*, *tengo tuto* y *hacer la cimarra*. Algunas de las
primeras de esa lista (*tbt*, *atraque*, *puelche*, *buitre*) deben su frecuencia sobre todo al sentido neutro.

## Limitaciones

- **Muestreo uniforme en el tiempo, no por comentario.** Los periodos con poca actividad pesan más que su volumen
  real. Además, cada ventana trae 100 comentarios seguidos de pocos hilos, así que las frases no son independientes.
- **La búsqueda es literal sobre las formas del diccionario.** No reconoce flexiones que no estén entre las
  variantes ni distingue sentidos. Formas que también son palabras corrientes (*dale*, *micro*, *colectivo*,
  *completo*) entran aunque se usen en su sentido neutro.
- **El corte de oraciones es simple.** Puede partir oraciones en abreviaturas.
- **Uso de los datos.** El contenido es de Reddit. Usarlo solo para evaluación académica, sin redistribuirlo en
  público, y citar Arctic Shift como fuente.
