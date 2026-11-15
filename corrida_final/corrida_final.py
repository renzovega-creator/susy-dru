"""Corrida final SUSY_DRU: diseño firmado M1–M4 + familias del objetivo 2, evaluadas UNA vez en test.

Motor: el del notebook `conciliacion_codigos/hiperparametros_SUSY_DRU.ipynb` (celdas 13, 16, 17 y 18),
copiado sin cambios salvo UNO, que recomienda la guía `_comprension` (§12 y §17):
  * early stopping con la semántica de four_models (celda 34, L124-128): la paciencia agotada corta
    SOLO la fase en curso y la siguiente fase se ejecuta igual. El notebook cortaba todas las fases.

Protocolo (pre-registrado en config_corrida_final.json, que se escribe ANTES de entrenar):
  * split firmado: 26 000 filas -> 5000/1000/5000, SPLIT_SEED 2021 (split_comun.npz, verificado).
  * 5 semillas de modelo (101-505). Ensemble = media de probabilidades.
  * Todo lo que se elige (hiperparámetros de los clásicos, "mejor cuántico") se elige con VALIDACIÓN.
  * El test se puntúa una sola vez por corrida; IC bootstrap estratificado (2000) y DeLong + Holm por
    familia de comparaciones definida de antemano.
  * Si ya existen resultados, el script se niega a sobrescribirlos (no se re-toca el test).

Uso (WSL, venv ~/qml_workshop/.venv):  python corrida_final.py [--n-jobs 4]
En el repo los resultados ya están en esta carpeta, así que el script se niega a correr; para repetirla,
copia este script a una carpeta hermana dentro del repo (p. ej. corrida_final_rep/) y córrelo allí (ver README.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.svm import LinearSVC

# RUTAS. Único cambio frente al script que produjo los resultados (sha256 1cff9a6b… en config_corrida_final.json):
# el original vivía en <proyecto>/conciliacion_codigos/corrida_final/; aquí se adaptan las rutas al layout del repo.
# La caché del notebook de hiperparámetros no está en el repo: el cotejo de validación se salta solo.
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPLICA = PROJECT / "replica_dru"
CSV = REPLICA / "data" / "susy_head_60000.csv"
SPLIT_NPZ = PROJECT / "conciliacion" / "comparacion_uniforme" / "split_comun.npz"
NB_CACHE = PROJECT / "conciliacion" / "hiperparametros_cache"  # no versionada
SCORES = HERE / "scores"

sys.path.insert(0, str(REPLICA))
os.environ["PYTHONPATH"] = str(REPLICA) + os.pathsep + os.environ.get("PYTHONPATH", "")
from dru_core import DRUModel, build_circuit, Adam, layer_mask  # noqa: E402

SEEDS = (101, 202, 303, 404, 505)
PI = math.pi


# ----------------------------------------------------------------------------- HP (= notebook, celda 13)
@dataclass(frozen=True)
class HP:
    features: tuple = ("lep1_pt", "lep2_pt", "met", "m_tr_2", "m_delta_r", "lep1_eta", "lep2_eta")
    scaler: str = "quantile"
    angle_half_range: float = math.pi
    split_protocol: str = "train_val_test"
    n_layers: int = 5
    readout: str = "xyz"
    init: str = "normal"
    init_scale: float = 0.05
    readout_init: tuple = (0.0, 0.0, 1.0, 0.0)
    loss: str = "bce"
    hybrid_alpha: float = 0.5
    l2: float = 0.0
    l2_on_readout: bool = False
    batch_size: int = 128
    layerwise_epochs: int = 1
    layerwise_lr: float = 0.05
    stages: tuple = ((12, 0.02), (18, 0.01))
    adam_reset: str = "stage"
    patience: int = 7
    patience_per_stage: bool = True
    min_delta: float = 1e-4


def scaled_lr(hp, f):
    return replace(hp, layerwise_lr=hp.layerwise_lr * f, stages=tuple((e, lr * f) for e, lr in hp.stages))


def hp_key(hp):  # misma clave que el notebook (permite cotejar con su caché de validación)
    return hashlib.sha1(json.dumps(asdict(hp), sort_keys=True, default=str).encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- configuraciones pre-registradas
M2 = HP()
M1 = replace(M2, scaler="minmax")
M3_LIN = replace(M2, loss="infidelity", readout="z")          # four_models, tal cual (fidelidad lineal, ec. 7)
M3_CUA = replace(M2, loss="fidelity", readout="z")            # fidelidad cuadrática (ec. 9, α_c = 1)
M4_LIN = replace(M3_LIN, split_protocol="train_test")
M4_CUA = replace(M3_CUA, split_protocol="train_test")

CONFIGS: dict[str, HP] = {
    # A · núcleo firmado (las dos lecturas de "fidelidad" para que la decisión 1 no bloquee)
    "M1": M1, "M2": M2, "M3-lin": M3_LIN, "M4-lin": M4_LIN, "M3-cua": M3_CUA, "M4-cua": M4_CUA,
    # B · pérdidas del objetivo 2 (lectura Z salvo BCE-XYZ = M2); α=1 ≡ M3-lin, α=0 ≡ M3-cua
    "Z híbrida α=0.25": replace(M3_LIN, loss="hybrid", hybrid_alpha=0.25),
    "Z híbrida α=0.5": replace(M3_LIN, loss="hybrid", hybrid_alpha=0.5),
    "Z híbrida α=0.75": replace(M3_LIN, loss="hybrid", hybrid_alpha=0.75),
    "Z BCE": replace(M3_LIN, loss="bce"),
    # C · rango h × inicialización (base M2)
    **{f"h={lab} {il}": replace(M2, angle_half_range=v, init=ii)
       for lab, v in (("π/4", PI / 4), ("π/2", PI / 2), ("π", PI))
       for il, ii in (("0.05·N", "normal"), ("U[0,1)", "uniform01"))
       if not (lab == "π" and ii == "normal")},
    # D · número de capas (base M2, exploratorio)
    **{f"L={L}": replace(M2, n_layers=L) for L in (1, 2, 3, 8)},
    # E · pérdida × lr (la cuadrática se había barrido con el lr de la lineal)
    **{f"{n} lr×{f}": scaled_lr(h, f) for n, h in (("M3-lin", M3_LIN), ("M3-cua", M3_CUA)) for f in (0.5, 2.0)},
}
ALIAS = {"h=π 0.05·N": "M2", "L=5": "M2", "Z híbrida α=1": "M3-lin", "Z híbrida α=0": "M3-cua",
         "BCE-XYZ": "M2", "M3-lin lr×1.0": "M3-lin", "M3-cua lr×1.0": "M3-cua"}

# Familias de comparaciones (Holm dentro de cada una). (modelo, referencia)
FAMILIAS = {
    "F1 núcleo vs M2": [(m, "M2") for m in ("M1", "M3-lin", "M4-lin", "M3-cua", "M4-cua")],
    "F1b fidelidad cuadrática vs lineal": [("M3-cua", "M3-lin"), ("M4-cua", "M4-lin")],
    "F2 pérdidas vs lineal (lectura Z)": [(m, "M3-lin") for m in
                                          ("Z híbrida α=0.75", "Z híbrida α=0.5", "Z híbrida α=0.25",
                                           "M3-cua", "Z BCE", "M2")],
    "F3 rango × init vs M2": [(m, "M2") for m in CONFIGS if m.startswith("h=")],
    "F4 capas vs L=5": [(m, "M2") for m in ("L=1", "L=2", "L=3", "L=8")],
    "F5 pérdida × lr": [("M3-lin lr×0.5", "M3-lin"), ("M3-lin lr×2.0", "M3-lin"),
                        ("M3-cua lr×0.5", "M3-cua"), ("M3-cua lr×2.0", "M3-cua")],
    # F6 se completa en tiempo de ejecución con los clásicos y el mejor cuántico por validación
}

COLS = ["label", "lep1_pt", "lep1_eta", "lep1_phi", "lep2_pt", "lep2_eta", "lep2_phi", "met", "met_phi", "met_rel",
        "axial_met", "m_r", "m_tr_2", "r", "mt2", "s_r", "m_delta_r", "dphi_r_b", "cos_theta_r1"]

DF = Y = IDX = None


def load_data():
    global DF, Y, IDX
    DF = pd.read_csv(CSV, header=None, names=COLS, nrows=26_000)
    Y = DF["label"].to_numpy(int)
    IDX = split_literal_four(Y)
    z = np.load(SPLIT_NPZ)
    assert all(np.array_equal(z[k], IDX[k]) for k in IDX), "split_comun.npz no coincide con four_models"
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(IDX[a]) & set(IDX[b])


def split_literal_four(labels, seed=2021, tr=2500, va=500, te=2500):  # four_models, celda 19
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for c in (0, 1):
        idx = np.where(labels == c)[0]; rng.shuffle(idx); sel = idx[:tr + va + te]
        out["train"].extend(sel[:tr]); out["val"].extend(sel[tr:tr + va]); out["test"].extend(sel[tr + va:])
    for k in out:
        out[k] = np.asarray(out[k], int); rng.shuffle(out[k])
    return out


# ----------------------------------------------------------------------------- motor (= notebook, celda 17)
def prepare(hp):
    X = DF[list(hp.features)].to_numpy(float)
    tr, va, te = IDX["train"], IDX["val"], IDX["test"]
    if hp.split_protocol == "train_test":
        tr = np.concatenate([tr, va])
    h = hp.angle_half_range
    if hp.scaler == "minmax":
        t = MinMaxScaler(feature_range=(-h, h)).fit(X[tr]); f = t.transform
    elif hp.scaler == "quantile":
        t = QuantileTransformer(n_quantiles=min(1000, len(tr)), output_distribution="uniform",
                                random_state=2021).fit(X[tr])
        f = lambda A: (2.0 * t.transform(A) - 1.0) * h  # noqa: E731
    else:
        f = lambda A: A  # noqa: E731
    # En M4 la validación NO se usa para entrenar ni para parar; se puntúa solo para poder reportarla
    # (está dentro de los datos de ajuste, así que su AUC de validación es optimista y no se usa para elegir).
    return dict(Xtr=f(X[tr]), ytr=Y[tr], Xte=f(X[te]), yte=Y[te], Xva=f(X[va]), yva=Y[va],
                use_val=hp.split_protocol != "train_test")


def init_params(hp, circ, seed):
    rng = np.random.default_rng(seed)
    n = circ.n_params
    if hp.init == "normal":
        p = hp.init_scale * rng.normal(size=n)
    elif hp.init == "uniform01":
        p = rng.uniform(0.0, 1.0, size=n)
    elif hp.init == "uniform2pi":
        p = rng.uniform(0.0, 2 * np.pi, size=n)
        p[circ.w_indices] = 1.0 + 0.1 * rng.normal(size=len(circ.w_indices))
    else:
        raise ValueError(hp.init)
    if circ.readout_slice is not None:
        p[circ.readout_slice] = np.asarray(hp.readout_init, float)
    return p


def train(hp, seed, data):
    Xtr, ytr, Xva, yva, use_val = data["Xtr"], data["ytr"], data["Xva"], data["yva"], data["use_val"]
    rng = np.random.default_rng(seed)
    circ = build_circuit("muten", len(hp.features), hp.n_layers, 1, hp.readout)
    model = DRUModel(circ, loss=hp.loss, alpha=hp.hybrid_alpha, l2=0.0)
    p = init_params(hp, circ, seed)
    l2mask = np.ones(circ.n_params)
    if circ.readout_slice is not None and not hp.l2_on_readout:
        l2mask[circ.readout_slice] = 0.0
    n_reg = l2mask.sum()

    def loss_grad(p, Xb, yb):
        v, g = model.loss_and_grad(p, Xb, yb)
        if hp.l2:
            v += hp.l2 * np.sum((p * l2mask) ** 2) / n_reg
            g = g + 2.0 * hp.l2 * p * l2mask / n_reg
        return v, g

    def batches():
        o = rng.permutation(len(Xtr))
        for s in range(0, len(o), hp.batch_size):
            yield Xtr[o[s:s + hp.batch_size]], ytr[o[s:s + hp.batch_size]]

    for layer in range(hp.n_layers):
        opt, m = Adam(circ.n_params, lr=hp.layerwise_lr), layer_mask(circ, layer)
        for _ in range(hp.layerwise_epochs):
            for Xb, yb in batches():
                p = opt.step(p, loss_grad(p, Xb, yb)[1], m)

    best = dict(auc=-np.inf, p=p.copy(), epoch=0)
    hist, ep, stalled = [], 0, 0
    opt = Adam(circ.n_params)
    for s_i, (n_ep, lr) in enumerate(hp.stages):
        if hp.adam_reset == "stage" or (hp.adam_reset == "never" and s_i == 0):
            opt = Adam(circ.n_params)
        opt.lr = lr
        if hp.patience_per_stage:
            stalled = 0
        for _ in range(n_ep):
            ep += 1
            if hp.adam_reset == "epoch":
                opt = Adam(circ.n_params, lr=lr)
            losses = []
            for Xb, yb in batches():
                v, g = loss_grad(p, Xb, yb); p = opt.step(p, g); losses.append(v)
            row = dict(epoch=ep, stage=s_i, lr=lr, loss=float(np.mean(losses)))
            stop_stage = False
            if use_val:
                auc = roc_auc_score(yva, model.predict_proba(p, Xva)); row["val_auc"] = auc
                if auc > best["auc"] + hp.min_delta:
                    best, stalled = dict(auc=auc, p=p.copy(), epoch=ep), 0
                else:
                    stalled += 1
                    stop_stage = stalled >= hp.patience
            hist.append(row)
            if stop_stage:      # semántica de four_models: corta SOLO esta fase
                break
    if not use_val:             # M4: sin validación se queda con la última época
        best = dict(auc=np.nan, p=p.copy(), epoch=ep)
    return model, best, hist


def job(name, hp, seed):
    f = SCORES / f"{hp_key(hp)}_{seed}.npz"
    if f.exists():
        return
    t0 = time.perf_counter()
    data = prepare(hp)
    model, best, hist = train(hp, seed, data)
    meta = dict(config=name, seed=seed, best_epoch=best["epoch"], epochs_run=len(hist),
                n_params=model.circuit.n_params, seconds=time.perf_counter() - t0,
                stages_run=sorted({h["stage"] for h in hist}))
    np.savez(f, val=model.predict_proba(best["p"], data["Xva"]), test=model.predict_proba(best["p"], data["Xte"]),
             loss_hist=np.array([h["loss"] for h in hist]),
             val_hist=np.array([h.get("val_auc", np.nan) for h in hist]), meta=json.dumps(meta))


# ----------------------------------------------------------------------------- clásicos (tuning en validación)
def classical(out_rows):
    X = DF[list(HP().features)].to_numpy(float)
    Xtr, Xva, Xte = X[IDX["train"]], X[IDX["val"]], X[IDX["test"]]
    ytr, yva = Y[IDX["train"]], Y[IDX["val"]]

    def score(m, A):
        return m.predict_proba(A)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(A)

    grids = {
        "LogReg": [({"C": C}, lambda C=C: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)))
                   for C in (0.01, 0.1, 1.0, 10.0, 100.0)],
        "SVM lineal": [({"C": C}, lambda C=C: make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=50_000)))
                       for C in (0.01, 0.1, 1.0, 10.0)],
        "RF": [({"max_depth": d, "min_samples_leaf": l},
                lambda d=d, l=l: RandomForestClassifier(n_estimators=400, max_depth=d, min_samples_leaf=l,
                                                        random_state=2021, n_jobs=2))
               for d in (6, 8, 12, None) for l in (1, 10)],
        "MLP-73": [({"alpha": a, "lr": lr},
                    lambda a=a, lr=lr, s=101: make_pipeline(StandardScaler(), MLPClassifier(
                        hidden_layer_sizes=(8,), alpha=a, learning_rate_init=lr, max_iter=1200,
                        early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=s)))
                   for a in (1e-5, 1e-4, 1e-3) for lr in (5e-4, 1e-3, 3e-3)],
    }
    chosen = {}
    for fam, cands in grids.items():
        best = None
        for cfg, mk in cands:
            m = mk().fit(Xtr, ytr)
            v = roc_auc_score(yva, score(m, Xva))
            out_rows.append(dict(familia=fam, config=json.dumps(cfg), val_auc=v))
            if best is None or v > best[0]:
                best = (v, cfg)
        chosen[fam] = best[1]
    # ajuste final con train (sin validación), como en el Entregable 1
    res = {}
    for fam, cfg in chosen.items():
        if fam == "MLP-73":
            vals, tests = [], []
            for s in SEEDS:
                m = make_pipeline(StandardScaler(), MLPClassifier(
                    hidden_layer_sizes=(8,), alpha=cfg["alpha"], learning_rate_init=cfg["lr"], max_iter=1200,
                    early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=s)).fit(Xtr, ytr)
                vals.append(score(m, Xva)); tests.append(score(m, Xte))
            res[fam] = dict(val=np.array(vals), test=np.array(tests), cfg=cfg, n_params=73)
        else:
            mk = dict((json.dumps(c), f) for c, f in grids[fam])[json.dumps(cfg)]
            m = mk().fit(Xtr, ytr)
            res[fam] = dict(val=score(m, Xva)[None], test=score(m, Xte)[None], cfg=cfg,
                            n_params={"LogReg": 8, "SVM lineal": 8}.get(fam, np.nan))
    return res


# ----------------------------------------------------------------------------- estadística (= notebook, celda 18)
def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong(y, s1, s2):
    y = np.asarray(y); pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, V10, V01 = [], [], []
    for s in (np.asarray(s1), np.asarray(s2)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(np.r_[s[pos], s[neg]])
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        V10.append((tz[:m] - tx) / n); V01.append(1 - (tz[m:] - ty) / m)
    S = np.cov(np.vstack(V10)) / m + np.cov(np.vstack(V01)) / n
    d = aucs[0] - aucs[1]; var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    return d, (2 * stats.norm.sf(abs(d) / np.sqrt(var)) if var > 0 else 1.0)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def auc_ci(y, s, B=2000, seed=2026):
    rng = np.random.default_rng(seed); c0, c1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    a = [roc_auc_score(y[i], s[i]) for i in
         (np.r_[rng.choice(c0, len(c0)), rng.choice(c1, len(c1))] for _ in range(B))]
    return np.percentile(a, [2.5, 97.5])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()

    if (HERE / "resultados_test.csv").exists():
        sys.exit("resultados_test.csv ya existe: la corrida final no se repite (el test se toca una vez).")
    SCORES.mkdir(exist_ok=True)
    load_data()

    import sklearn, scipy, joblib  # noqa: E401
    config = dict(
        fecha=time.strftime("%Y-%m-%d %H:%M:%S"), protocolo=__doc__.strip().splitlines()[0],
        semillas_modelo=SEEDS, split=dict(archivo=str(SPLIT_NPZ.relative_to(PROJECT)), seed=2021,
                                          tamaños={k: int(len(v)) for k, v in IDX.items()},
                                          senal={k: int(Y[v].sum()) for k, v in IDX.items()}),
        early_stopping="four_models: la paciencia corta solo la fase en curso",
        configuraciones={k: {**asdict(v), "hp_key": hp_key(v)} for k, v in CONFIGS.items()},
        alias=ALIAS, familias=FAMILIAS,
        familia_F6="mejor cuántico por AUC de validación (entre M1, M2, M3-*, híbridas, Z BCE, rango×init, capas, "
                   "pérdida×lr; M4 excluido porque no tiene validación) y M2 frente a cada clásico",
        clasicos=dict(LogReg="C ∈ {0.01,0.1,1,10,100}", SVM_lineal="C ∈ {0.01,0.1,1,10}",
                      RF="400 árboles, depth ∈ {6,8,12,None}, leaf ∈ {1,10}",
                      MLP_73="8 ocultas; alpha ∈ {1e-5,1e-4,1e-3} × lr ∈ {5e-4,1e-3,3e-3}; 5 semillas",
                      seleccion="mejor AUC de validación; ajuste final con train", escalado="StandardScaler"),
        estadistica=dict(ic="bootstrap estratificado 2000, percentiles 2.5-97.5, seed 2026",
                         comparacion="DeLong (Sun y Xu) sobre el ensemble de test", multiples="Holm por familia"),
        sha256=dict(dru_core=sha256(REPLICA / "dru_core.py"), split_comun=sha256(SPLIT_NPZ),
                    script=sha256(__file__), csv_60000=sha256(CSV)),
        versiones=dict(python=sys.version.split()[0], plataforma=platform.platform(), numpy=np.__version__,
                       pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
                       joblib=joblib.__version__),
    )
    (HERE / "config_corrida_final.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
    print(f"{len(CONFIGS)} configuraciones × {len(SEEDS)} semillas = {len(CONFIGS) * len(SEEDS)} corridas", flush=True)

    from joblib import Parallel, delayed
    t0 = time.time()
    Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(job)(n, h, s) for n, h in CONFIGS.items() for s in SEEDS)
    print(f"cuánticos listos en {time.time() - t0:.0f} s", flush=True)

    # ---- cotejo con la caché de validación del notebook (mismo hp_key) para M2 y M3-lin
    cot = []
    for name in ("M2", "M3-lin", "M1"):
        for s in SEEDS:
            f = NB_CACHE / f"{hp_key(CONFIGS[name])}_{s}.npz"
            if f.exists():
                a = np.load(f)["val_scores"]; b = np.load(SCORES / f"{hp_key(CONFIGS[name])}_{s}.npz")["val"]
                cot.append(dict(config=name, seed=s, max_abs_diff_val=float(np.max(np.abs(a - b))),
                                auc_nb=roc_auc_score(Y[IDX["val"]], a), auc_aqui=roc_auc_score(Y[IDX["val"]], b)))
    pd.DataFrame(cot).to_csv(HERE / "cotejo_cache_notebook.csv", index=False)

    # ---- clásicos
    tun = []
    clas = classical(tun)
    pd.DataFrame(tun).to_csv(HERE / "clasicos_tuning_validacion.csv", index=False)

    # ---- tablas
    yva, yte = Y[IDX["val"]], Y[IDX["test"]]
    S = {}  # nombre -> dict(val=(k,n), test=(k,n), ...)
    for name, hp in CONFIGS.items():
        zs = [np.load(SCORES / f"{hp_key(hp)}_{s}.npz") for s in SEEDS]
        metas = [json.loads(str(z["meta"])) for z in zs]
        S[name] = dict(val=np.array([z["val"] for z in zs]), test=np.array([z["test"] for z in zs]),
                       n_params=metas[0]["n_params"], tipo="cuántico",
                       epochs=np.mean([m["epochs_run"] for m in metas]),
                       best_epoch=np.mean([m["best_epoch"] for m in metas]),
                       seconds=np.mean([m["seconds"] for m in metas]),
                       sin_fase2=sum(1 not in m["stages_run"] for m in metas))
    for fam, r in clas.items():
        S[fam] = dict(val=r["val"], test=r["test"], n_params=r["n_params"], tipo="clásico", cfg=r["cfg"])

    rows = []
    for name, d in S.items():
        ens_te, ens_va = d["test"].mean(0), d["val"].mean(0)
        lo, hi = auc_ci(yte, ens_te)
        per_te = [roc_auc_score(yte, t) for t in d["test"]]
        per_va = [roc_auc_score(yva, v) for v in d["val"]]
        rows.append(dict(modelo=name, tipo=d["tipo"], n_params=d["n_params"],
                         test_auc_ens=roc_auc_score(yte, ens_te), ic95_lo=lo, ic95_hi=hi,
                         test_auc_media=np.mean(per_te), test_auc_sd=np.std(per_te, ddof=1) if len(per_te) > 1 else np.nan,
                         val_auc_ens=roc_auc_score(yva, ens_va), val_auc_media=np.mean(per_va),
                         val_auc_sd=np.std(per_va, ddof=1) if len(per_va) > 1 else np.nan,
                         n_semillas=len(d["test"]), epocas=d.get("epochs"), mejor_epoca=d.get("best_epoch"),
                         semillas_sin_fase2=d.get("sin_fase2"), s_por_corrida=d.get("seconds"),
                         config_clasico=json.dumps(d.get("cfg")) if d.get("cfg") else ""))
    res = pd.DataFrame(rows)
    res.loc[res.modelo.str.startswith("M4"), "val_auc_ens"] = np.nan   # validación dentro del ajuste: no se reporta
    res.loc[res.modelo.str.startswith("M4"), ["val_auc_media", "val_auc_sd"]] = np.nan

    # F6: mejor cuántico por validación
    elegibles = res[(res.tipo == "cuántico") & ~res.modelo.str.startswith("M4")]
    best_q = elegibles.sort_values("val_auc_ens", ascending=False).modelo.iloc[0]
    fams = dict(FAMILIAS)
    fams["F6 cuántico vs clásico"] = ([("M2", c) for c in clas] +
                                      ([(best_q, c) for c in clas] if best_q != "M2" else []))
    comp = []
    for fam, pairs in fams.items():
        rr = []
        for a, b in pairs:
            d, p = delong(yte, S[a]["test"].mean(0), S[b]["test"].mean(0))
            pa = [roc_auc_score(yte, t) for t in S[a]["test"]]; pb = [roc_auc_score(yte, t) for t in S[b]["test"]]
            dpar = (np.mean(np.array(pa) - np.array(pb)) if len(pa) == len(pb) else np.nan)
            rr.append(dict(familia=fam, modelo=a, referencia=b, delta_auc_ens=d, delta_pareado_semillas=dpar,
                           p_delong=p))
        for r, ph in zip(rr, holm([r["p_delong"] for r in rr])):
            r["p_holm"] = ph
        comp += rr
    comp = pd.DataFrame(comp)
    res.to_csv(HERE / "resultados_test.csv", index=False)
    comp.to_csv(HERE / "comparaciones_test.csv", index=False)
    (HERE / "mejor_cuantico_por_validacion.txt").write_text(best_q + "\n", encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(res[["modelo", "tipo", "n_params", "test_auc_ens", "ic95_lo", "ic95_hi", "test_auc_media", "test_auc_sd",
               "val_auc_ens", "semillas_sin_fase2"]].round(4).to_string(index=False))
    print(comp.round(4).to_string(index=False))
    print("mejor cuántico por validación:", best_q)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
