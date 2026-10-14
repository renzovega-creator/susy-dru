"""Evaluación común de la comparación uniforme (reglas fijadas antes de ver resultados).

- Todos los códigos evaluados sobre el MISMO test (split_comun.npz, 5000 eventos, mismo orden).
- Ensemble: regla propia de cada código (media de probabilidades; Sebastián: media de rangos, su celda 30).
- Representante de cada código = método cuántico con mayor AUC de VALIDACIÓN media por semilla
  (sin M4, que no tiene validación, ni variantes exploratorias). El test no interviene en la elección.
- Comparaciones por DeLong sobre los scores de test del ensemble, Holm dentro de cada familia.
  DeLong, Holm y bootstrap: estadistica.py (verificada idéntica a la de DRC_SUSY_protocolo_encoding_v2,
  celda 13: diferencia máxima 3e-13 en 276 pares).
"""
import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from estadistica import bootstrap_auc_ci as boot_ci
from estadistica import delong_test as delong
from estadistica import holm_adjust as holm

HERE = Path(__file__).resolve().parent
S = np.load(HERE / "split_comun.npz")
y_val, y_test = S["y_val"], S["y_test"]
SEEDS = (101, 202, 303)

# ---------------- carga de scores ----------------
runs = {}   # (codigo, metodo) -> list of dict(seed, val, test, val_auc)
for f in sorted((HERE / "scores").glob("*.npz")):
    code, method, seed = re.match(r"(\w+?)__(.+)__seed(\d+)\.npz", f.name).groups()
    z = np.load(f)
    val = z["val"] if z["val"].size else None
    runs.setdefault((code, method), []).append(dict(
        seed=int(seed), val=val, test=z["test"],
        val_auc=roc_auc_score(y_val, val) if val is not None else np.nan,
        n_params=int(z["n_params"]) if "n_params" in z else np.nan))
for modo in ("receta", "explora"):
    d = HERE / "scores_sebastian" / modo
    if not (d / "runs.csv").exists():
        continue
    r = pd.read_csv(d / "runs.csv")
    for _, row in r.iterrows():
        runs.setdefault(("sebastian", row.variant), []).append(dict(
            seed=int(row.seed), val=None, test=np.load(d / "scores" / f"seed{int(row.seed)}_{row.variant}.npy"),
            val_auc=float(row.val_auc), n_params=62 if "_7" not in row.variant else 72))

CLASSICAL = {"LogReg", "RF", "MLP-73", "Random_forest", "Logistic_regression", "Linear_SVM"}
EXPLORA = {("sebastian", "E0_7"), ("sebastian", "E2_7")}


def ensemble(code, members, which):
    arrs = [m[which] for m in members if m[which] is not None]
    if not arrs:
        return None
    if code == "sebastian":
        return np.mean([stats.rankdata(a) / len(a) for a in arrs], axis=0)
    return np.mean(arrs, axis=0)


rows, ens = [], {}
for (code, method), members in sorted(runs.items()):
    members = sorted(members, key=lambda m: m["seed"])
    te = ensemble(code, members, "test"); va = ensemble(code, members, "val")
    ens[(code, method)] = te
    lo, hi = boot_ci(y_test, te, seed=2026)
    seed_test = [roc_auc_score(y_test, m["test"]) for m in members]
    rows.append(dict(codigo=code, metodo=method, tipo="clasico" if method in CLASSICAL else
                     ("exploratorio" if (code, method) in EXPLORA else "cuantico"),
                     semillas=len(members), n_params=members[0]["n_params"],
                     val_auc_media=(np.nanmean([m["val_auc"] for m in members])
                                    if np.isfinite([m["val_auc"] for m in members]).any() else np.nan),  # M4: sin val
                     val_auc_ens=roc_auc_score(y_val, va) if va is not None else np.nan,
                     test_auc_media=np.mean(seed_test), test_auc_sd=np.std(seed_test, ddof=1) if len(seed_test) > 1 else np.nan,
                     test_auc_ens=roc_auc_score(y_test, te), ic95_low=lo, ic95_high=hi))
tab = pd.DataFrame(rows)
tab.to_csv(HERE / "resultados_uniforme.csv", index=False)

# ---------------- selección por validación ----------------
q = tab[(tab.tipo == "cuantico") & tab.val_auc_media.notna()]
reps = q.loc[q.groupby("codigo").val_auc_media.idxmax()]
cl = tab[tab.tipo == "clasico"]
best_cl = cl.loc[cl.val_auc_media.idxmax()]

# ---------------- comparaciones ----------------
fam = {
    "1_representantes": [((a.codigo, a.metodo), (b.codigo, b.metodo))
                         for a, b in itertools.combinations(reps.itertuples(), 2)],
    "2_contrato_four": [(("four", "M2"), ("four", "M1")), (("four", "M3"), ("four", "M2")), (("four", "M4"), ("four", "M3"))],
    "3_vs_mejor_clasico": [((r.codigo, r.metodo), (best_cl.codigo, best_cl.metodo)) for r in reps.itertuples()],
    "4_exploratorias": [(("sebastian", "E2_7"), ("sebastian", "E0_7")), (("sebastian", "E2_7"), ("sebastian", "E0"))],
}
comp = []
for fname, pairs in fam.items():
    pairs = [(a, b) for a, b in pairs if a in ens and b in ens]
    out = []
    for a, b in pairs:
        auc_a, auc_b, z, p = delong(y_test, ens[a], ens[b])
        out.append(dict(familia=fname, a="/".join(a), b="/".join(b), auc_a=auc_a, auc_b=auc_b, delta=auc_a - auc_b, z=z, p=p))
    if out:
        for o, ph in zip(out, holm([o["p"] for o in out])):
            o["p_holm"] = ph; o["significativo_5pct"] = bool(ph < 0.05)
    comp += out
comp = pd.DataFrame(comp)
comp.to_csv(HERE / "comparaciones_uniforme.csv", index=False)

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = lambda v: f"{v:.4f}"
print("== Resultados por método (test común: 5000 eventos) ==")
print(tab.sort_values(["tipo", "test_auc_ens"], ascending=[True, False]).to_string(index=False, float_format=fmt))
print("\n== Representantes (elegidos por AUC de validación media) ==")
print(reps[["codigo", "metodo", "val_auc_media", "test_auc_ens", "ic95_low", "ic95_high"]].to_string(index=False, float_format=fmt))
print(f"\nMejor clásico por validación: {best_cl.codigo}/{best_cl.metodo} (val {best_cl.val_auc_media:.4f}, test {best_cl.test_auc_ens:.4f})")
print("\n== Comparaciones DeLong (Holm por familia) ==")
print(comp.to_string(index=False, float_format=lambda v: f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"))
