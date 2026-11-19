"""Baseline: traduce las 397 frases de `test` del split oficial con cada modelo SIN ajuste.

Guarda un CSV por modelo (id_mistral, frase_chilena, prediccion) en models/baseline/predicciones/, listo para
`python -m evaluacion`. Usa el mismo prompt que el ajuste de Gemma (models/gemma4/comun.py) para que la comparación
con los modelos ajustados sea justa. Corre en CUDA, en la GPU de un Mac (mps) o en CPU.

    python models/baseline/baseline_sin_ajuste.py mt0_large qwen35_4b      # solo esos
    python models/baseline/baseline_sin_ajuste.py                          # todos
"""
import gc
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "models/gemma4"))
from comun import INSTRUCCION_CHAT, PROMPT  # noqa: E402

# mt5-large no va: sin ajuste solo rellena huecos y no sigue instrucciones; mt0-large es la misma arquitectura
# ajustada a instrucciones.
MODELOS = {
    "mt0_large": ("bigscience/mt0-large", "seq2seq"),
    "qwen35_4b": ("Qwen/Qwen3.5-4B", "chat"),
    "gemma4_e4b_it": ("google/gemma-4-E4B-it", "chat"),
    "gemma4_e4b_base": ("google/gemma-4-E4B", "base"),
}
SALIDA = RAIZ / "models/baseline/predicciones"
DISP = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def cargar_test():
    pares = pd.read_csv(RAIZ / "data/entrenamiento/dataset_chilenismos_opus.csv", encoding="utf-8-sig")
    split = pd.read_csv(RAIZ / "data/entrenamiento/splits/split_oficial.csv")[["id_mistral", "split"]]
    test = pares.merge(split, on="id_mistral", validate="one_to_one").query("split == 'test'").reset_index(drop=True)
    assert len(test) == 397, len(test)
    return test


def cargar(modelo_id, modo):
    tok = AutoTokenizer.from_pretrained(modelo_id)
    if modo == "seq2seq":   # mT5/mT0 dan NaN en fp16
        dtype = torch.bfloat16 if DISP == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        return tok, AutoModelForSeq2SeqLM.from_pretrained(modelo_id, dtype=dtype).to(DISP).eval()
    tok.padding_side = "left"   # relleno a la izquierda para generar en lote
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if DISP == "cuda" and not torch.cuda.is_bf16_supported() else torch.bfloat16
    return tok, AutoModelForCausalLM.from_pretrained(modelo_id, dtype=dtype).to(DISP).eval()


def prompt(tok, frase, modo):
    frase = frase.strip()
    if modo == "base":     # mismo formato que el ajuste de Gemma base
        return tok.bos_token + PROMPT.format(chilena=frase)
    if modo == "chat":     # enable_thinking=False: Qwen3.5 piensa por defecto; Gemma lo ignora
        return tok.apply_chat_template([{"role": "user", "content": INSTRUCCION_CHAT.format(chilena=frase)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return INSTRUCCION_CHAT.format(chilena=frase)   # seq2seq


def traducir(frases, modelo_id, modo, bs=8, max_new_tokens=96):
    tok, modelo = cargar(modelo_id, modo)
    salidas = []
    for i in range(0, len(frases), bs):
        textos = [prompt(tok, f, modo) for f in frases[i:i + bs]]
        x = tok(textos, return_tensors="pt", padding=True, add_special_tokens=(modo == "seq2seq")).to(DISP)
        with torch.no_grad():
            out = modelo.generate(**x, do_sample=False, max_new_tokens=max_new_tokens)
        if modo != "seq2seq":
            out = out[:, x["input_ids"].shape[1]:]   # solo lo generado, sin el prompt
        # igual que comun.traducir: se queda con la primera línea
        salidas += [s.strip().split("\n")[0].strip() for s in tok.batch_decode(out, skip_special_tokens=True)]
        print(f"  {len(salidas)}/{len(frases)}", flush=True)
    del modelo
    gc.collect()
    if DISP == "mps":
        torch.mps.empty_cache()
    return salidas


def main(nombres):
    test = cargar_test()
    SALIDA.mkdir(parents=True, exist_ok=True)
    for nombre in nombres:
        ruta = SALIDA / f"predicciones_{nombre}_test.csv"
        if ruta.exists():
            print(f"{nombre}: ya existe {ruta.name}, lo salto")
            continue
        print(f"{nombre} en {DISP}", flush=True)
        df = test[["id_mistral", "frase_chilena"]].copy()
        df["prediccion"] = traducir(df.frase_chilena.tolist(), *MODELOS[nombre])
        df.to_csv(ruta, index=False)
        print(f"{nombre}: {ruta.name} | vacías: {(df.prediccion == '').sum()}\n{df.head(3).to_string()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(MODELOS))
