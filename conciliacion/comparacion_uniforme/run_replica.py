"""replica_dru sobre el split común: su receta (run_grid.py, preset paper) con los índices fijados.

Métodos: los del grid con ansatz muten (bloques A, B, C), sin los duplicados exactos a 1 qubit
(A-mse = A-fid, A-cerezo = A-infid) ni las variantes perez_* (aprenden al azar con Quantile).
Baselines: LogReg, RF y MLP-73 de run_grid.py:L177-L209 (sin tuning).
Semillas 101/202/303 (las mismas en los cuatro códigos).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "replica_dru"))
from susy_data import FEATURES_7, SusySplit, load_susy  # noqa: E402
from train import TrainConfig, train_one  # noqa: E402
from dru_core import DRUModel, build_circuit  # noqa: E402

OUT = HERE / "scores"; OUT.mkdir(exist_ok=True)
S = np.load(HERE / "split_comun.npz")
df = load_susy(60_000)  # cache en replica_dru/data/ (se descarga de UCI la primera vez)
labels = df["label"].to_numpy(dtype=int)
assert np.array_equal(labels[S["test"]], S["y_test"])
values = df[FEATURES_7].to_numpy(dtype=np.float64)


def split(scaler, lo=-np.pi, hi=np.pi):
    """Igual que susy_data.make_split:L189-L212, pero con los índices del split común."""
    Xtr, Xva, Xte = values[S["train"]], values[S["val"]], values[S["test"]]
    if scaler == "minmax":
        tf = MinMaxScaler(feature_range=(lo, hi))
        Xtr, Xva, Xte = tf.fit_transform(Xtr), tf.transform(Xva), tf.transform(Xte)
    elif scaler == "quantile":
        tf = QuantileTransformer(n_quantiles=min(1000, len(Xtr)), output_distribution="uniform", random_state=2021)
        f = lambda v: lo + (hi - lo) * v
        Xtr, Xva, Xte = f(tf.fit_transform(Xtr)), f(tf.transform(Xva)), f(tf.transform(Xte))
    return SusySplit(Xtr, S["y_train"].astype(float), Xva, S["y_val"].astype(float), Xte,
                     S["y_test"].astype(float), list(FEATURES_7), scaler)


GRID = [("A-fid", "fidelity", "z", "quantile"), ("A-infid", "infidelity", "z", "quantile"),
        ("A-bce", "bce", "z", "quantile"), ("A-hybrid", "hybrid", "z", "quantile"),
        ("B-bce-mm", "bce", "z", "minmax"), ("B-fid-mm", "fidelity", "z", "minmax"),
        ("C-xyz-bce", "bce", "xyz", "quantile")]
cfg = TrainConfig(max_epochs=25, patience=5, layerwise_epochs=1, batch_size=128, lr=0.05, l2=1e-5)
splits = {s: split(s) for s in ("quantile", "minmax")}
for mid, loss, readout, scaler in GRID:
    sp = splits[scaler]
    for seed in (101, 202, 303):
        r = train_one("muten", sp, loss=loss, readout=readout, seed=seed, cfg=cfg, method_id=mid, verbose=False)
        m = DRUModel(build_circuit("muten", 7, 5, 1, readout), loss=loss)
        np.savez(OUT / f"replica__{mid}__seed{seed}.npz", val=m.predict_proba(r.params, sp.X_val),
                 test=m.predict_proba(r.params, sp.X_test), n_params=r.n_params, best_epoch=r.best_epoch)
        print(f"{mid:10s} seed={seed} val_auc={r.best_val_auc:.4f} ({r.seconds:.0f}s)", flush=True)

Xtr, Xva, Xte = values[S["train"]], values[S["val"]], values[S["test"]]
for name, model in [("LogReg", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000))),
                    ("RF", RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=10,
                                                  random_state=2021, n_jobs=-1))]:
    m = clone(model).fit(Xtr, S["y_train"])
    np.savez(OUT / f"replica__{name}__seed0.npz", val=m.predict_proba(Xva)[:, 1], test=m.predict_proba(Xte)[:, 1])
for seed in (101, 202, 303):
    m = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(8,), max_iter=1200, early_stopping=True,
                                                      n_iter_no_change=30, random_state=seed)).fit(Xtr, S["y_train"])
    np.savez(OUT / f"replica__MLP-73__seed{seed}.npz", val=m.predict_proba(Xva)[:, 1], test=m.predict_proba(Xte)[:, 1])
print("FIN replica")
