# Gemma 4 E4B en Reddit Chile (evaluación fuera de dominio)

Conjunto: `data/evaluacion/reddit_chile.json` (1.900 frases de r/chile y r/RepublicadeChile, sin referencia; ver el
README de esa carpeta). Predicciones generadas con `models/gemma4/predecir_reddit.py` (GPU de Colab, greedy, mismo
prompt que en test) y evaluadas con `python -m evaluacion --sin-juez` más `models/gemma4/juzgar_reddit.py` (juez Opus
5.5, esfuerzo medio, rúbrica 1). Sin referencia, BERTScore se calcula **contra la entrada**, así que mide cuánto se
parece la salida a la frase chilena y no es comparable con el de test.

## Resultados

| Configuración | Términos retenidos % ↓ | Sentido (1–5) | Fluidez (1–5) | Jerga eliminada % | **Aceptable %** | Aceptable % en test |
|---|---:|---:|---:|---:|---:|---:|
| `gemma4_base_p2r16` (solo paso 2) | 8,2 | 3,78 | 4,36 | 96,0 | **49,2** | 77,6 |
| `gemma4_base_p1r32_p2r8` (paso 1 + 2, elegida en val) | 8,4 | 3,75 | 4,36 | 96,1 | **48,8** | 76,8 |
| `gemma4_it_p2r8` (instruct; juez en 1.053 de 1.900) | 10,5 | 3,64 | 4,28 | 93,5 | **45,3** | 72,3 |

En test el sentido medio era ~4,4; en Reddit cae a ~3,75. La eliminación de jerga casi no cambia. Copia literal de la
entrada: 3,4 % (base) y 3,5 % (instruct).

Distribución del sentido (`base_p2r16`): 5 → 730, 4 → 445, 3 → 341, 2 → 353, 1 → 31. Uno de cada cinco casos (sentido
≤ 2) cambia o pierde lo que decía la frase.

## Qué errores comete

Muestra al azar de casos con sentido ≤ 2, según los comentarios del juez:

1. **Reemplaza la expresión por otra con otro significado**: *a la hora del pico* → "a la hora de la merienda";
   *el encañao de Gómez* → "el loco"; *rusio* (rubio) → "ruso"; *detonao* → "explosivo".
2. **Borra la expresión y con ella el núcleo de la frase**: *habla en serio con funar la PSU* → "habla en serio sobre
   la PSU".
3. **Sentido neutro o referencia local mal leída**: *UDI* (partido) → "universidad"; *paco* → "su novia";
   *taco* (atasco) queda como "taco".
4. **Inventa contenido**: *inti o tótem de guanaco* → "cocaína"; gritos de estadio reescritos como otra frase.
5. Minoritario: deja la jerga sin tocar (*dar jugo*).

**Paso 1 vs. solo paso 2.** Fallan en la misma proporción (379 y 384 casos con sentido ≤ 2) y comparten 231 de esos
errores. Solo coinciden palabra por palabra en el 22 % de las frases, y el sentido de ambos correlaciona 0,63. El
paso 1 arregla algunas expresiones (*funar*, *encañao*, *paco*) y rompe otras (*sapear* → "buscar marihuana",
*me aweoné* → "me cago"). No da una ventaja neta fuera de dominio.

## Lecciones aprendidas

**El problema está en la forma del dataset de traducción, no en el ajuste.** Los 3.980 pares del paso 2 tienen una
estructura única: cada frase chilena se escribió alrededor de **una** expresión y la neutra **siempre** la elimina
(99,7 %). El modelo aprendió la regla "hay exactamente un chilenismo y hay que reemplazarlo". En test, que tiene la
misma forma, funciona (~77 % aceptable). En Reddit falla porque:

- **Hay palabras ambiguas usadas en su sentido corriente** (*pico*, *taco*, *paco*, *completo*). En train siempre eran
  jerga, así que el modelo las reemplaza por reflejo.
- **Hay frases con varias expresiones, garabatos y escritura fonética** (*podi*, *vendío*, *detonao*). Train no tiene
  ninguna.
- **No existe una salida segura.** Ningún par enseña a conservar el sentido con una paráfrasis prudente cuando la
  expresión es opaca. Ante la duda, el modelo produce algo concreto y plausible, aunque sea falso.

**Qué habría que haber incluido** (unos 15–25 % de pares adicionales, sacados del corpus del paso 1 y nunca de Reddit):

| Tipo de par | Objetivo |
|---|---|
| Palabra ambigua en su sentido neutro | dejar la frase igual |
| Frase sin jerga | copia exacta |
| Varias expresiones, con ruido y garabatos | neutralizarlas todas sin perder el sentido |
| Expresión rara u opaca | paráfrasis conservadora; ante la duda, prima el sentido sobre la eliminación de la jerga |

**Lo que el dataset no resuelve.** Algunas fallas son de conocimiento del léxico (*rusio*, *detonao*, *inti*, la
UDI): ningún formato de pares las arregla sin ver la palabra. Eso corresponde al diccionario (RAG) o a un paso 1 con
más cobertura.

**Sobre la evaluación.** Test comparte la distribución de train y sobreestima la calidad. La evaluación fuera de
dominio fue lo que destapó el sesgo, así que debería formar parte del diseño desde el principio. Además, "jerga
eliminada" y "términos retenidos" premian justo el comportamiento que falla aquí: un modelo puede sacar la jerga en el
96 % de los casos y aun así cambiar el sentido en uno de cada cinco.

## Archivos

Por configuración: `predicciones_<cfg>_reddit.csv` (`id, frase_chilena, prediccion`), `_evaluacion.json` (resumen) y
`_por_frase.csv` (métricas y veredicto del juez por frase). En GitHub solo están los `_evaluacion.json`: los CSV
contienen texto de Reddit y están en `.gitignore`. Los CSV completos están en `resultados/gemma4/` del repo privado de
Hugging Face `mariopedemonte12/chineutro-reddit`.
