"""Cuenta cuántas expresiones de nuestro diccionario (scraping/data/rag_documentos.jsonl) aparecen en el
Chilean Spanish Corpus.

Para cada expresión (clave, término y variantes) cuenta en cuántos textos del corpus aparece, por fuente, y
resume la cobertura: todas, de una palabra frente a varias, y las del dataset del paso 2 por split.

Uso, desde la raíz del repo, con las condiciones del dataset aceptadas en Hugging Face y sesión iniciada
(`hf auth login` o variable HF_TOKEN):
    python3 models/gemma4/contar_diccionario.py --fraccion 0.05             # 5 % de los row groups (rápido)
    python3 models/gemma4/contar_diccionario.py --fraccion 1.0 --procesos 8 # corpus completo

Dependencias: pip install huggingface_hub pyarrow pandas pyahocorasick
Si se corta, al volver a ejecutarlo retoma (cada parquet deja su resultado en <salida>/escaneo_diccionario/).
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import comun  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraccion", type=float, default=0.05, help="fracción de row groups a leer (1.0 = todo)")
    ap.add_argument("--procesos", type=int, default=None)
    ap.add_argument("--salida", type=Path, default=RAIZ / "salidas/gemma4")
    ap.add_argument("--glob-local", default=None, help=argparse.SUPPRESS)   # pruebas con parquets locales
    a = ap.parse_args()

    import pandas as pd
    a.salida.mkdir(parents=True, exist_ok=True)
    ruta_indice = a.salida / "indice_corpus.parquet"
    if a.glob_local:
        import fsspec
        fs, protocolo = fsspec.filesystem("file"), "file"
        indice = comun.indice_corpus(sorted(fs.glob(a.glob_local)), fs)
    else:
        protocolo = "hf"
        if ruta_indice.exists():
            indice = pd.read_parquet(ruta_indice)
        else:
            print("Armando el índice del corpus (solo metadatos)...")
            indice = comun.indice_corpus(*comun.listar_parquets())
            indice.to_parquet(ruta_indice)

    dic = comun.cargar_diccionario(RAIZ)
    print(f"rag_documentos: {dic.clave.nunique():,} expresiones, {dic.forma.nunique():,} formas buscables")
    docs, filas = comun.escanear_diccionario(indice, dic.forma, a.salida / "escaneo_diccionario", a.fraccion,
                                             a.procesos, protocolo=protocolo)
    print(f"\nTextos leídos: {filas.sum():,} ({100 * filas.sum() / indice.filas.sum():.1f} % del corpus)")
    tabla = comun.presencia_por_clave(dic, docs, a.fraccion)
    ruta = a.salida / f"presencia_diccionario_f{a.fraccion}.csv"
    tabla.to_csv(ruta)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\nCobertura (textos estimados en el corpus completo):")
    print(comun.resumen_presencia(tabla).to_string())
    print("\nMás frecuentes:")
    print(tabla.head(30)[["termino", "textos_estimados", "varias_palabras", "split_dataset"]].to_string())
    ausentes = tabla[tabla.textos_estimados == 0]
    print(f"\nExpresiones sin apariciones ({len(ausentes)}): " + ", ".join(ausentes.termino.head(100)))
    print("\nTabla completa:", ruta)


if __name__ == "__main__":
    main()
