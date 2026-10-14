"""Split común para la comparación uniforme: el del contrato (26 000 filas -> 2500/500/2500 por clase).

Usa el algoritmo LITERAL de final_quantum_susy_1q_four_models.ipynb (celda 19, SPLIT_SEED=2021),
que es el código que implementa lo firmado. Guarda índices posicionales sobre las filas de SUSY
(válidos tanto en las primeras 26 000 como en las primeras 60 000 filas) y su verificación.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPLICA = HERE.parents[1] / "replica_dru"
CACHE = REPLICA / "data" / "susy_head_60000.csv"
if not CACHE.exists():  # la primera vez descarga las 60 000 filas de UCI con replica_dru/susy_data.py
    import sys
    sys.path.insert(0, str(REPLICA))
    from susy_data import load_susy
    load_susy(60_000)
COLS = ["label", "lep1_pt", "lep1_eta", "lep1_phi", "lep2_pt", "lep2_eta", "lep2_phi", "met", "met_phi",
        "met_rel", "axial_met", "m_r", "m_tr_2", "r", "mt2", "s_r", "m_delta_r", "dphi_r_b", "cos_theta_r1"]
N_ROWS, SEED, TR, VA, TE = 26_000, 2021, 2500, 500, 2500


def make_balanced_split_indices(labels, seed, train_per_class, val_per_class, test_per_class):
    # copia literal de four_models, celda 19 (sin el chequeo de tamaño)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    splits = {"train": [], "val": [], "test": []}
    for class_id in [0, 1]:
        class_indices = np.where(labels == class_id)[0]
        rng.shuffle(class_indices)
        selected = class_indices[:train_per_class + val_per_class + test_per_class]
        train_end = train_per_class
        val_end = train_per_class + val_per_class
        splits["train"].extend(selected[:train_end])
        splits["val"].extend(selected[train_end:val_end])
        splits["test"].extend(selected[val_end:])
    for split_name in splits:
        splits[split_name] = np.asarray(splits[split_name], dtype=int)
        rng.shuffle(splits[split_name])
    return splits


lines = CACHE.read_text(encoding="utf-8").splitlines()[:N_ROWS]
sha26 = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
df = pd.read_csv(CACHE, header=None, names=COLS, nrows=N_ROWS)
y = df["label"].to_numpy(dtype=int)
sp = make_balanced_split_indices(y, SEED, TR, VA, TE)
for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    assert not set(sp[a]) & set(sp[b])
np.savez(HERE / "split_comun.npz", train=sp["train"], val=sp["val"], test=sp["test"],
         y_train=y[sp["train"]], y_val=y[sp["val"]], y_test=y[sp["test"]],
         n_rows=N_ROWS, seed=SEED, sha256_26000=sha26)
print(f"filas 0..{N_ROWS-1}  sha256={sha26}")
for k in ("train", "val", "test"):
    print(f"{k:5s} n={len(sp[k])}  señal={int(y[sp[k]].sum())}  índices {sp[k].min()}-{sp[k].max()}")
