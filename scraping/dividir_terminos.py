"""Divide data/terminos.csv en particiones contiguas, una por integrante."""
import argparse
import csv
from pathlib import Path

BASE = Path(__file__).parent
INTEGRANTES = ["Mario", "Agustin", "Martin", "Danko", "Antonio"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entrada", default=BASE / "data" / "terminos.csv", type=Path)
    parser.add_argument("--salida", default=BASE / "data" / "particiones", type=Path)
    parser.add_argument("--integrantes", nargs="+", default=INTEGRANTES)
    args = parser.parse_args()

    with open(args.entrada, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        filas = list(reader)

    args.salida.mkdir(parents=True, exist_ok=True)
    n = len(args.integrantes)
    base, resto = divmod(len(filas), n)
    inicio = 0
    for i in range(n):
        tam = base + (1 if i < resto else 0)
        parte = filas[inicio:inicio + tam]
        destino = args.salida / f"terminos_{args.integrantes[i].lower()}.csv"
        with open(destino, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(parte)
        print(f"{destino.name}: {len(parte)} términos (filas {inicio + 1}-{inicio + tam})")
        inicio += tam

    print(f"Total: {len(filas)} términos en {n} particiones")


if __name__ == "__main__":
    main()
