# T-A1: factibilidad del scraping de diccionariochileno.cl

Inspección del 2 de noviembre de 2026 (unas 98 solicitudes). **El scraping es factible:** los datos están en HTML simple, sin login ni JavaScript, y `robots.txt` permite las rutas que necesitamos.

## Permisos

- `robots.txt` solo prohíbe `/admin` y `/vote`. No define `Crawl-delay` ni sitemap, y `/sitemap.xml` da 404.
- El pie de página indica "Copyright" y no enlaza a términos de uso. Esto no reemplaza la revisión legal de la propuesta.
- Conviene no guardar los seudónimos de los autores, porque no los usamos.

## Endpoints

| Ruta | Contenido | Uso |
| --- | --- | --- |
| `/terms` | Lista completa en una sola página (382 KB), con el n.º de definiciones por término | Lista de términos |
| `/terms/<letra>` | Lista por letra; no incluye 11 términos que empiezan con símbolo o número | Solo para verificar |
| `/term/<término>` | Todas las definiciones del término | Scraper (T-A2) |
| `/search?s=`, `/random`, `/submit` | Búsqueda, término al azar, envío de definiciones | No usar |
| `/moderacion` | Respondió 500; sus botones registran votos | No usar |
| `/vote`, `/admin` | Prohibidas por `robots.txt` | No usar |

No vimos una API en el HTML. No probamos rutas no enlazadas.

## Campos por definición

| Campo | Disponible | Observación |
| --- | --- | --- |
| Término | Siempre | Puede agrupar variantes ("Sutro o sutra") |
| Definición | 92 de 92 | Es una explicación, no siempre un reemplazo directo |
| Ejemplo | 92 de 92 | Solo 61 contienen el término literal |
| Votos | 92 de 92 | Pueden ser negativos; solo están en la página del término |
| Etiquetas | 37 de 92 | Términos relacionados |
| Antigüedad | 92 de 92 | Solo relativa ("12 years ago") |
| ID | Sí | Está en los botones de voto; el scraper lo usa para no repetir definiciones |

Muestra aleatoria: 60 términos y 92 definiciones (semilla 42). Los 60 términos traen ejemplo (IC 95 %: 94-100 %). Los contadores del listado coincidieron con las definiciones de cada página. En los 21 términos con varias definiciones, estas aparecen ordenadas por votos.

## Frecuencia de extracción

- **Ritmo:** 1 solicitud a la vez, con 3 s de pausa. Es prudente, no un máximo medido.
- **Respuesta observada a ese ritmo:** 87 de 87 con estado 200 en las solicitudes registradas, con mediana de 0,70 s y máximo de 1,62 s. No apareció ningún error 429 ni cabeceras de límite de uso.
- **Sin prueba de carga:** es un sitio pequeño, así que no buscamos el límite real. No sabemos cuánto más rápido se podría ir.
- **Duración:** 3171 páginas a unos 3,7 s cada una toman **unas 3,25 horas**, unos 46 MB de HTML.
- **Veces:** una sola. Una foto fija evita que los splits cambien.
- **Reglas del scraper:** detenerse ante 429 o 5xx, poder reanudar y no repartir la descarga entre varias personas.

## Lista de términos

Archivo: `data/terminos.csv`, con fuente y fecha de extracción.

| Medida | Valor |
| --- | --- |
| Filas | 3172 |
| URL únicas | 3171 ("Cachai" aparece dos veces con la misma URL) |
| Claves normalizadas únicas | 3152 |
| Definiciones según el listado | 5006 |
| Términos con 1 definición | 2140 |
| Términos con 2 o más | 1032 |
| Términos con 3 o más | 384 |
| Términos con 5 o más | 94 |
| Términos de una palabra | 1960 |

## Implicancias para la selección (T-A3)

1. **Los votos exigen el scraping completo.** La lista solo trae el n.º de definiciones, y ese dato premia términos de internet: el primero es "UwU", con 20.
2. **Calidad.** En 11 de los 60 términos, la definición mejor votada tiene 0 votos o menos. Hay nombres propios, como "OLGITA MARINA (PERSONAJE DE TELESERIE )", y términos de internet que no son propiamente chilenos, como "fomo" o "looksmaxxing".
3. **Duplicados.** Hay 19 grupos de variantes (mayúsculas, tildes o puntuación) que se fusionan por `clave_normalizada`. Las URL no distinguen mayúsculas: `/term/raja` muestra las mismas 10 definiciones que `/term/Raja`, aunque el listado le atribuye 3 a "raja". Lo mismo ocurre con la segunda fila de "Cachai". Son como máximo 4 de 5006 definiciones que el sitio no muestra.
4. **El filtro "la frase contiene el término" se queda corto.** Un tercio de los ejemplos humanos no pasaría una búsqueda literal ("Pegarse la mision" aparece como "se pega la misión"). El filtro del sintético debe aceptar variantes o conjugaciones.
5. **Vulgares.** Están entre los términos más votados, así que el criterio pendiente afecta mucho a la lista final.

## Reproducir

```bash
python3 -I inspeccion.py listar      # 1 solicitud
python3 -I inspeccion.py verificar   # 26 solicitudes
python3 -I inspeccion.py muestrear -n 60 --semilla 42
```

## Scraping y RAG

```bash
python3 scraper.py mis_terminos.txt      # .txt (un término por línea) o .csv con columna "termino"
python3 rag.py cachai -k 3 --min-votos 0 # muestra el contexto de un término
```

Para el sintético, la lista debe traer solo términos de entrenamiento: el RAG se construye únicamente con los términos de la lista, lo que evita fugas hacia validación y prueba.
