# NLP-ChiNeutro

## Reparto de términos

Los 3172 términos de `scraping/data/terminos.csv` están repartidos en 5 particiones contiguas, una por integrante, en `scraping/data/particiones/`:

| Integrante | Archivo | Términos |
| --- | --- | --- |
| Mario | `terminos_mario.csv` | 635 |
| Agustín | `terminos_agustin.csv` | 635 |
| Martín | `terminos_martin.csv` | 634 |
| Danko | `terminos_danko.csv` | 634 |
| Antonio | `terminos_antonio.csv` | 634 |

Para regenerarlas (por ejemplo, si cambia `terminos.csv`):

```bash
python3 scraping/dividir_terminos.py
# o con otros integrantes u otro orden:
python3 scraping/dividir_terminos.py --integrantes Mario Agustin Martin Danko Antonio
```
