"""Convierte una COPIA de notebook en un script ejecutable para la verificación.

Uso: python nb2py.py <notebook.ipynb> <salida.py> <subs.json>

subs.json = {"skip_cells": [3, 17], "subs": [[celda, "texto viejo", "texto nuevo"], ...]}
- Las celdas se numeran 1-based contando markdown (igual que conciliacion.md).
- Cada sustitución debe encontrarse exactamente una vez: si no, el script aborta.
- Se eliminan líneas que empiezan con % o ! (magias de IPython).
- Si la última sentencia de una celda es una expresión, se imprime (emula el output).
"""
import ast
import json
import sys

src, dst, subs_path = sys.argv[1:4]
nb = json.load(open(src, encoding="utf-8"))
cfg = json.load(open(subs_path, encoding="utf-8"))
skip = set(cfg.get("skip_cells", []))
subs = cfg.get("subs", [])
used = [0] * len(subs)

out = ["from __future__ import annotations", "import matplotlib", "matplotlib.use('Agg')", "import matplotlib.pyplot as _plt",
       "_plt.show = lambda *a, **k: _plt.close('all')", ""]
for i, cell in enumerate(nb["cells"], 1):
    if cell["cell_type"] != "code":
        continue
    code = "".join(cell["source"])
    if i in skip:
        out.append(f"# ===== CELDA {i} OMITIDA (neutralizada para la verificación) =====\n")
        continue
    for k, (ci, old, new) in enumerate(subs):
        if ci == i:
            n = code.count(old)
            if n != 1:
                sys.exit(f"Sustitución {k} en celda {i}: encontrada {n} veces: {old!r}")
            code = code.replace(old, new)
            used[k] += 1
    code = "\n".join(l for l in code.splitlines()
                     if not l.lstrip().startswith(("%", "!", "from __future__")))
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        sys.exit(f"Celda {i} no parsea: {e}")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        lines = code.splitlines()
        start, end = last.lineno - 1, last.end_lineno
        expr = "\n".join(lines[start:end])
        lines[start:end] = [f"print({expr.strip()})"]
        code = "\n".join(lines)
    out.append(f"# ===== CELDA {i} =====\nprint('\\n######## CELDA {i}', flush=True)\n{code}\n")

missing = [subs[k] for k, u in enumerate(used) if u == 0]
if missing:
    sys.exit(f"Sustituciones sin aplicar: {missing}")
open(dst, "w", encoding="utf-8").write("\n".join(out))
print(f"{src} -> {dst} ({len(subs)} sustituciones, celdas omitidas: {sorted(skip)})")
