"""Referencia: Claude Opus traduce las 397 frases de `test` con la misma instrucción que los modelos sin ajuste.

Va por Claude Code (`claude -p`, igual que evaluacion/juez.py), en lotes de 40 frases. Guarda
models/baseline/predicciones/predicciones_opus_test.csv. Ojo: Opus también reescribió las referencias del dataset y
es el juez, así que su puntaje sale inflado; sirve como techo, no como competidor.

    python models/baseline/opus.py
"""
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from baseline_sin_ajuste import INSTRUCCION_CHAT, SALIDA, cargar_test

MODELO = "claude-opus-5-5"
LOTE = 40
SCHEMA = {"type": "object", "required": ["traducciones"], "properties": {"traducciones": {"type": "array", "items": {
    "type": "object", "required": ["id", "traduccion"],
    "properties": {"id": {"type": "integer"}, "traduccion": {"type": "string"}}}}}}


def traducir_lote(frases):
    """frases: {id: frase_chilena} -> {id: traducción}"""
    prompt = (INSTRUCCION_CHAT.format(chilena="").strip()
              + "\nHaz esto con cada frase de la lista y devuelve una traducción por id.\n\n"
              + json.dumps([{"id": i, "frase": f} for i, f in frases.items()], ensure_ascii=False, indent=1))
    cmd = ["claude", "-p", "--model", MODELO, "--output-format", "json", "--tools", "",
           "--no-session-persistence", "--json-schema", json.dumps(SCHEMA)]
    with tempfile.TemporaryDirectory() as cwd:   # carpeta vacía: no carga contexto del repo
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=900)
    salida = json.loads(proc.stdout)
    if salida.get("is_error") or not isinstance(salida.get("structured_output"), dict):
        raise RuntimeError(str(salida.get("result"))[:500])
    out = {t["id"]: t["traduccion"].strip() for t in salida["structured_output"]["traducciones"]}
    faltan = set(frases) - set(out)
    if faltan:
        raise RuntimeError(f"faltan ids {sorted(faltan)}")
    return out


if __name__ == "__main__":
    test = cargar_test()
    frases = dict(zip(test.id_mistral, test.frase_chilena))
    ids = list(frases)
    lotes = [{i: frases[i] for i in ids[k:k + LOTE]} for k in range(0, len(ids), LOTE)]
    with ThreadPoolExecutor(3) as ex:
        trad = {k: v for r in ex.map(traducir_lote, lotes) for k, v in r.items()}
    df = test[["id_mistral", "frase_chilena"]].copy()
    df["prediccion"] = df.id_mistral.map(trad)
    SALIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(SALIDA / "predicciones_opus_test.csv", index=False)
    print(f"opus: {len(df)} | vacías: {df.prediccion.isna().sum()}\n{df.head(3).to_string()}")
