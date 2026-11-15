"""Split oficial del dataset de pares (train / val / test), estratificado por expresión.

    python data/entrenamiento/splits/crear_split_oficial.py

Lee data/entrenamiento/dataset_chilenismos_opus.csv y escribe data/entrenamiento/splits/split_oficial.csv
(id_mistral, clave, def_id, split). Solo usa la biblioteca estándar y es determinista (semilla 42).

Método (ver README.md en esta carpeta):
- Cada expresión (`clave`) tiene 5 pares. De cada una, 4 van a train y 1 queda fuera; ese par va a val o a test,
  alternando entre expresiones barajadas (por separado las de una y las de varias palabras), así que val y test
  tienen la mitad de las expresiones cada uno, con la misma proporción de expresiones de varias palabras.
- Toda expresión aparece en train. El par que queda fuera se elige entre los de un significado (`def_id`) que
  tiene otro par en la misma expresión, para que su significado también se vea en train (si se puede).
"""
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

SEMILLA = 42
FUERA_POR_CLAVE = 1          # pares de cada expresión que van a val/test (con 5 pares: 80 / 10 / 10)
AQUI = Path(__file__).resolve().parent
PARES = AQUI.parent / "dataset_chilenismos_opus.csv"
SALIDA = AQUI / "split_oficial.csv"


def crear_split(pares, semilla=SEMILLA):
    rng = random.Random(semilla)
    por_clave = defaultdict(list)
    for p in pares:
        por_clave[p["clave"]].append(p)
    claves = sorted(por_clave)
    split = {}
    fuera = []
    for clave in claves:
        grupo = sorted(por_clave[clave], key=lambda p: int(p["id_mistral"]))
        n_def = Counter(p["def_id"] for p in grupo)
        candidatos = [p for p in grupo if n_def[p["def_id"]] >= 2] or grupo
        elegidos = rng.sample(candidatos, min(FUERA_POR_CLAVE, len(grupo) - 1))
        for p in grupo:
            split[p["id_mistral"]] = "train"
        fuera.append([p["id_mistral"] for p in elegidos])
    # val / test: alterna entre expresiones barajadas, por separado para las de una y de varias palabras
    for varias in (False, True):
        orden = [i for i, c in enumerate(claves) if (len(c.split()) > 1) == varias]
        rng.shuffle(orden)
        for k, i in enumerate(orden):
            for id_ in fuera[i]:
                split[id_] = "val" if k % 2 == 0 else "test"
    return split


def main():
    with open(PARES, encoding="utf-8-sig") as f:
        pares = list(csv.DictReader(f))
    split = crear_split(pares)
    filas = sorted(((int(p["id_mistral"]), p["clave"], p["def_id"], split[p["id_mistral"]]) for p in pares))
    with open(SALIDA, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_mistral", "clave", "def_id", "split"])
        w.writerows(filas)

    # Verificaciones
    total = Counter(s for *_, s in filas)
    por_clave = defaultdict(Counter)
    for _, clave, _, s in filas:
        por_clave[clave][s] += 1
    assert all(c["train"] >= 1 for c in por_clave.values()), "hay una expresión sin pares en train"
    defs_train = {(c, d) for _, c, d, s in filas if s == "train"}
    sin_ver = sum(1 for _, c, d, s in filas if s != "train" and (c, d) not in defs_train)
    print(f"{SALIDA.name}: {len(filas)} pares | " + " / ".join(f"{s} {total[s]}" for s in ("train", "val", "test")))
    print(f"Expresiones: {len(por_clave)} (todas en train) | en val {sum(1 for c in por_clave.values() if c['val'])} "
          f"| en test {sum(1 for c in por_clave.values() if c['test'])}")
    print(f"Pares de val/test cuyo significado (def_id) no aparece en train: {sin_ver}")


if __name__ == "__main__":
    main()
