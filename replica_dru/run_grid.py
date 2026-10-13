"""
Experimento principal: comparacion de funciones de perdida y escalados para el
clasificador DRU sobre SUSY (objetivos especificos 2 y 3 de la propuesta
firmada).

Uso:
    python run_grid.py --preset quick     # ~2 min, verifica el flujo
    python run_grid.py --preset medium    # ~15 min, resultados presentables
    python run_grid.py --preset paper     # presupuesto del Entregable/informe

Salidas (en ./resultados/):
    runs_<preset>.csv         una fila por (metodo, semilla)
    history_<preset>.csv      curva por epoca: loss, val AUC, ||g||, Var[g]
    benchmark_<preset>.csv    AUC de test por metodo (ensemble de semillas)
    bootstrap_<preset>.csv    diferencias pareadas de AUC con IC 95%
    figuras/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from susy_data import FEATURES_7, load_susy, make_split
from train import TrainConfig, ensemble_scores, train_one

OUT = Path(__file__).resolve().parent / "resultados"
FIG = OUT / "figuras"

PRESETS = {
    "quick":  dict(train=400,  val=200,  test=800,  epochs=6,  seeds=(101,),           boot=500),
    "medium": dict(train=2000, val=1000, test=4000, epochs=15, seeds=(101, 202, 303),  boot=2000),
    "paper":  dict(train=4000, val=1000, test=4000, epochs=25, seeds=(101, 202, 303, 404, 505), boot=3000),
}

# --------------------------------------------------------------------------
# Definicion del grid
# --------------------------------------------------------------------------
# Bloque A (nucleo de la propuesta): misma arquitectura y misma lectura Z,
# solo cambia la funcion de perdida -> el efecto observado ES el de la perdida.
# Bloque B: ablacion de escalado (MinMax vs Quantile) con la perdida fijada.
# Bloque C: ablacion de arquitectura y de lectura (contexto, no la hipotesis).
GRID = [
    # id            arch             loss          readout  scaler      layers  d
    ("A-fid",      "muten",          "fidelity",   "z",     "quantile", 5, "f7"),
    ("A-infid",    "muten",          "infidelity", "z",     "quantile", 5, "f7"),
    ("A-cerezo",   "muten",          "cerezo",     "z",     "quantile", 5, "f7"),
    ("A-bce",      "muten",          "bce",        "z",     "quantile", 5, "f7"),
    ("A-mse",      "muten",          "mse",        "z",     "quantile", 5, "f7"),
    ("A-hybrid",   "muten",          "hybrid",     "z",     "quantile", 5, "f7"),
    ("B-bce-mm",   "muten",          "bce",        "z",     "minmax",   5, "f7"),
    ("B-fid-mm",   "muten",          "fidelity",   "z",     "minmax",   5, "f7"),
    ("C-xyz-bce",  "muten",          "bce",        "xyz",   "quantile", 5, "f7"),
    ("C-ps-fid",   "perez_salinas",  "fidelity",   "z",     "quantile", 5, "f9"),
    ("C-psw-fid",  "perez_salinas_w", "fidelity",  "z",     "quantile", 5, "f9"),
    # Bloque D: interaccion escalado x ansatz. El encoding de Perez-Salinas NO
    # tiene pesos entrenables, asi que no puede reescalar la entrada; con
    # Quantile (uniforme en todo [-pi,pi]) el estado se aleatoriza y el modelo
    # no aprende (AUC ~ 0.50), mientras que con MinMax si aprende. El ansatz de
    # Muten, con w entrenable, absorbe el escalado y es casi insensible.
    ("D-ps-fid-mm",  "perez_salinas",   "fidelity", "z", "minmax", 5, "f9"),
    ("D-psw-fid-mm", "perez_salinas_w", "fidelity", "z", "minmax", 5, "f9"),
]


def build_splits(df, cfg):
    """Una particion por (escalado, padding). El test es el MISMO en todas."""
    splits = {}
    for scaler in ("quantile", "minmax"):
        for tag, pad in (("f7", None), ("f9", 3)):
            splits[(scaler, tag)] = make_split(
                df, features=FEATURES_7, scaler=scaler,
                train_per_class=cfg["train"], val_per_class=cfg["val"],
                test_per_class=cfg["test"], pad_to_multiple_of=pad,
            )
    return splits


def paired_bootstrap(y_true, score_map, n_replicates, seed=2026):
    rng = np.random.default_rng(seed)
    c0 = np.flatnonzero(y_true == 0)
    c1 = np.flatnonzero(y_true == 1)
    rows = []
    for _ in range(n_replicates):
        idx = np.concatenate([rng.choice(c0, len(c0), True), rng.choice(c1, len(c1), True)])
        ys = y_true[idx]
        rows.append({k: roc_auc_score(ys, v[idx]) for k, v in score_map.items()})
    return pd.DataFrame(rows)


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return float(thr[np.argmax(tpr - fpr)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="medium", choices=list(PRESETS))
    ap.add_argument("--rows", type=int, default=60_000)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = PRESETS[args.preset]
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    print(f"== preset {args.preset} == train/clase={cfg['train']} epochs={cfg['epochs']} "
          f"semillas={cfg['seeds']}\n")
    df = load_susy(args.rows)
    splits = build_splits(df, cfg)
    ref_split = splits[("quantile", "f7")]
    y_test = ref_split.y_test
    for key, sp in splits.items():
        assert np.array_equal(sp.y_test, y_test), "el test debe ser identico en todas"
    print(ref_split.summary(), "\n")

    tcfg = TrainConfig(max_epochs=cfg["epochs"], patience=5, layerwise_epochs=1,
                       batch_size=128, lr=0.05, l2=1e-5)

    run_rows, hist_rows, runs_by_method = [], [], {}
    for mid, arch, loss, readout, scaler, layers, dtag in GRID:
        sp = splits[(scaler, dtag)]
        # Perez-Salinas usa la convencion de inicializacion uniforme [0, 2pi]
        cfg_i = TrainConfig(**{**tcfg.__dict__,
                               "uniform_2pi_init": arch.startswith("perez"),
                               "init_scale": 0.05})
        members = []
        for seed in cfg["seeds"]:
            print(f"[{mid}] {arch} loss={loss} readout={readout} scaler={scaler} seed={seed}")
            r = train_one(arch, sp, loss=loss, n_layers=layers, readout=readout,
                          seed=seed, cfg=cfg_i, method_id=mid, verbose=not args.quiet)
            members.append(r)
            for h in r.history:
                hist_rows.append({"method_id": mid, "seed": seed, **h})
            from train import evaluate
            _, test_auc = evaluate(r.config, r.params, sp)
            run_rows.append(dict(method_id=mid, arch=arch, loss=loss, readout=readout,
                                 scaler=scaler, n_layers=layers, seed=seed,
                                 n_params=r.n_params, val_auc=r.best_val_auc,
                                 test_auc=test_auc, best_epoch=r.best_epoch,
                                 seconds=round(r.seconds, 1),
                                 final_grad_norm=r.history[-1]["grad_norm"]))
            print(f"    -> val_auc={r.best_val_auc:.4f} test_auc={test_auc:.4f} "
                  f"({r.seconds:.0f}s)\n")
        runs_by_method[mid] = (members, sp)

    runs_df = pd.DataFrame(run_rows)
    hist_df = pd.DataFrame(hist_rows)
    runs_df.to_csv(OUT / f"runs_{args.preset}.csv", index=False)
    hist_df.to_csv(OUT / f"history_{args.preset}.csv", index=False)

    # ---- ensembles cuanticos + baselines clasicos ----
    score_map, bench_rows = {}, []
    for mid, (members, sp) in runs_by_method.items():
        s = ensemble_scores(members, sp, "test")
        score_map[mid] = s
        sub = runs_df[runs_df.method_id == mid]
        bench_rows.append(dict(model=mid, family="quantum",
                               loss=sub.loss.iloc[0], scaler=sub.scaler.iloc[0],
                               readout=sub.readout.iloc[0], n_params=int(sub.n_params.iloc[0]),
                               test_auc_ensemble=roc_auc_score(y_test, s),
                               test_auc_mean=sub.test_auc.mean(),
                               test_auc_std=sub.test_auc.std(ddof=1) if len(sub) > 1 else np.nan))

    # Baselines clasicos sobre las MISMAS particiones (sin escalado a angulos;
    # cada pipeline clasico lleva su propio StandardScaler).
    frames = make_split(df, features=FEATURES_7, scaler="identity",
                        train_per_class=cfg["train"], val_per_class=cfg["val"],
                        test_per_class=cfg["test"])
    assert np.array_equal(frames.y_test, y_test), "los baselines deben usar el mismo test"
    for name, model in [
        ("LogReg", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=4000))),
        ("RandomForest", RandomForestClassifier(n_estimators=400, max_depth=8,
                                                min_samples_leaf=10, random_state=2021, n_jobs=-1)),
    ]:
        m = clone(model).fit(frames.X_train, frames.y_train)
        s = m.predict_proba(frames.X_test)[:, 1]
        score_map[name] = s
        bench_rows.append(dict(model=name, family="classical", loss="-", scaler="standard",
                               readout="-", n_params=np.nan,
                               test_auc_ensemble=roc_auc_score(y_test, s),
                               test_auc_mean=roc_auc_score(y_test, s), test_auc_std=np.nan))

    mlp_scores = []
    for seed in cfg["seeds"]:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(8,), max_iter=1200,
                                        early_stopping=True, n_iter_no_change=30,
                                        random_state=seed))
        m.fit(frames.X_train, frames.y_train)
        mlp_scores.append(m.predict_proba(frames.X_test)[:, 1])
    s = np.mean(np.stack(mlp_scores), axis=0)
    score_map["MLP-73"] = s
    bench_rows.append(dict(model="MLP-73", family="classical", loss="-", scaler="standard",
                           readout="-", n_params=73,
                           test_auc_ensemble=roc_auc_score(y_test, s),
                           test_auc_mean=float(np.mean([roc_auc_score(y_test, v) for v in mlp_scores])),
                           test_auc_std=float(np.std([roc_auc_score(y_test, v) for v in mlp_scores], ddof=1))
                           if len(mlp_scores) > 1 else np.nan))

    bench = pd.DataFrame(bench_rows).sort_values("test_auc_ensemble", ascending=False)
    bench.to_csv(OUT / f"benchmark_{args.preset}.csv", index=False)

    # ---- diferencias pareadas ----
    boot = paired_bootstrap(y_test, score_map, cfg["boot"])
    comparisons = [
        ("A-bce", "A-fid"), ("A-bce", "A-cerezo"), ("A-fid", "A-cerezo"),
        ("A-hybrid", "A-fid"), ("A-bce", "B-bce-mm"), ("A-fid", "B-fid-mm"),
        ("C-xyz-bce", "A-bce"), ("A-bce", "C-ps-fid"),
        ("C-xyz-bce", "MLP-73"), ("C-xyz-bce", "RandomForest"),
        ("D-ps-fid-mm", "C-ps-fid"), ("D-psw-fid-mm", "C-psw-fid"),
    ]
    rows = []
    for a, b in comparisons:
        d = boot[a] - boot[b]
        rows.append(dict(comparison=f"{a} - {b}",
                         point=roc_auc_score(y_test, score_map[a]) - roc_auc_score(y_test, score_map[b]),
                         mean=d.mean(), ci_low=d.quantile(0.025), ci_high=d.quantile(0.975),
                         concluyente=bool(d.quantile(0.025) > 0 or d.quantile(0.975) < 0)))
    boot_df = pd.DataFrame(rows)
    boot_df.to_csv(OUT / f"bootstrap_{args.preset}.csv", index=False)

    _figures(args.preset, bench, hist_df, y_test, score_map)

    print("\n================ BENCHMARK (AUC test) ================")
    print(bench.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n================ DIFERENCIAS PAREADAS (IC 95%) ================")
    print(boot_df.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nSalidas en {OUT}")
    (OUT / f"config_{args.preset}.json").write_text(json.dumps(
        dict(preset=args.preset, rows=args.rows, cfg=cfg, grid=GRID,
             sha256=ref_split.meta.get("sha256", "")), indent=2), encoding="utf-8")


def _figures(preset, bench, hist, y_test, score_map):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. benchmark
    b = bench.sort_values("test_auc_ensemble")
    colors = ["#E15759" if f == "quantum" else "#4E79A7" for f in b.family]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(b.model, b.test_auc_ensemble, color=colors)
    for i, v in enumerate(b.test_auc_ensemble):
        ax.text(v + 0.0008, i, f"{v:.4f}", va="center", fontsize=8)
    ax.set_xlabel("AUC en test fijo")
    ax.set_xlim(min(b.test_auc_ensemble) - 0.02, max(b.test_auc_ensemble) + 0.02)
    ax.set_title("SUSY / DRU: perdidas, escalados y baselines clasicos")
    fig.tight_layout()
    fig.savefig(FIG / f"benchmark_{preset}.png", dpi=150)
    plt.close(fig)

    # 2. curvas por perdida (bloque A)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    core = [m for m in hist.method_id.unique() if m.startswith("A-")]
    # Estilos distintos porque hay pares EXACTAMENTE superpuestos
    # (A-fid == A-mse y A-infid == A-cerezo a 1 qubit con lectura Z):
    # con linea continua uno tapa al otro y parece que falta una curva.
    styles = {"A-fid": ("-", 2.4), "A-mse": ((0, (2, 3)), 1.6),
              "A-infid": ("-", 2.4), "A-cerezo": ((0, (2, 3)), 1.6),
              "A-bce": ("-", 1.8), "A-hybrid": ("-.", 1.8)}
    for mid in core:
        h = hist[hist.method_id == mid].groupby("epoch").mean(numeric_only=True)
        ls, lw = styles.get(mid, ("-", 1.8))
        kw = dict(marker="o", ms=3, ls=ls, lw=lw, label=mid)
        axes[0].plot(h.index, h.val_auc, **kw)
        axes[1].plot(h.index, h.train_loss, **kw)
        axes[2].semilogy(h.index, h.grad_norm, **kw)
    axes[0].set(xlabel="epoca", ylabel="AUC validacion", title="Entrenabilidad por perdida")
    axes[1].set(xlabel="epoca", ylabel="loss de entrenamiento", title="Convergencia")
    axes[2].set(xlabel="epoca", ylabel=r"$\|\nabla L\|$", title="Magnitud del gradiente")
    for a in axes:
        a.legend(fontsize=7)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / f"perdidas_{preset}.png", dpi=150)
    plt.close(fig)

    # 3. ROC
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name in ("C-xyz-bce", "A-bce", "A-fid", "MLP-73", "RandomForest"):
        if name not in score_map:
            continue
        fpr, tpr, _ = roc_curve(y_test, score_map[name])
        ax.plot(fpr, tpr, lw=2, label=f"{name} ({roc_auc_score(y_test, score_map[name]):.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.5)
    ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos",
           title="ROC en el test fijo de SUSY")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"roc_{preset}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
