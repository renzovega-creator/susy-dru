"""
Bucle de entrenamiento del clasificador DRU.

Combina lo mejor de las tres implementaciones de la literatura:
  - Perez-Salinas et al. (2020): perdidas de fidelidad sobre estados-etiqueta.
  - Skolik et al. (2021) / Muten et al. (2021): crecimiento *layerwise* (cada
    capa recibe una pasada antes del ajuste global) como mitigacion de
    plateaus.
  - Notebook del Entregable 1: seleccion de checkpoint por AUC de validacion,
    early stopping, decaimiento de lr a mitad de entrenamiento y ensemble de
    varias semillas.

Ademas instrumenta lo que la literatura de barren plateaus exige medir y que
ninguna de las tres implementaciones registra: norma y varianza del gradiente
epoca a epoca (Cerezo et al. 2021; Larocca et al. 2025).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

from dru_core import Adam, DRUModel, build_circuit, layer_mask


@dataclass
class TrainConfig:
    lr: float = 0.05
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5
    layerwise_epochs: int = 1        # 0 desactiva el crecimiento layerwise
    lr_decay_at_half: float = 0.5
    l2: float = 1e-5
    min_auc_improvement: float = 1e-4
    init_scale: float = 0.05
    uniform_2pi_init: bool = False   # True = convencion Perez-Salinas
    grad_probe_batches: int = 4      # lotes usados para estimar Var[grad]


@dataclass
class RunResult:
    method_id: str
    seed: int
    params: np.ndarray
    history: list[dict]
    best_val_auc: float
    best_epoch: int
    seconds: float
    n_params: int
    config: dict = field(default_factory=dict)


def _minibatches(X, y, rng, batch_size):
    order = rng.permutation(len(X))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        yield X[idx], y[idx]


def _grad_stats(model, params, X, y, rng, n_batches, batch_size):
    """||grad|| media y Var[grad] entre parametros, sobre lotes independientes."""
    norms, grads = [], []
    for _ in range(n_batches):
        idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
        _, g = model.loss_and_grad(params, X[idx], y[idx])
        norms.append(np.linalg.norm(g))
        grads.append(g)
    G = np.stack(grads)
    return float(np.mean(norms)), float(np.mean(np.var(G, axis=0))), float(np.var(G))


def train_one(
    arch: str,
    split,
    loss: str,
    n_layers: int = 5,
    n_wires: int = 1,
    readout: str = "z",
    entangle: bool = False,
    alpha: float = 0.5,
    seed: int = 101,
    cfg: TrainConfig | None = None,
    method_id: str = "",
    verbose: bool = True,
) -> RunResult:
    cfg = cfg or TrainConfig()
    circ = build_circuit(arch, split.n_features, n_layers, n_wires, readout, entangle)
    model = DRUModel(circ, loss=loss, alpha=alpha, l2=cfg.l2)
    params = model.init_params(seed, cfg.init_scale, cfg.uniform_2pi_init)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val

    # ---- fase 1: crecimiento layerwise (Skolik et al.) ----
    if cfg.layerwise_epochs > 0:
        for layer in range(n_layers):
            opt = Adam(circ.n_params, lr=2.0 * cfg.lr)
            mask = layer_mask(circ, layer)
            for _ in range(cfg.layerwise_epochs):
                for Xb, yb in _minibatches(Xtr, ytr, rng, cfg.batch_size):
                    _, g = model.loss_and_grad(params, Xb, yb)
                    params = opt.step(params, g, mask)

    # ---- fase 2: ajuste global con early stopping por AUC de validacion ----
    opt = Adam(circ.n_params, lr=cfg.lr)
    history: list[dict] = []
    best = {"auc": -np.inf, "params": params.copy(), "epoch": 0}
    stalled = 0

    for epoch in range(1, cfg.max_epochs + 1):
        if epoch == cfg.max_epochs // 2 + 1:
            opt.lr *= cfg.lr_decay_at_half
        epoch_loss, n_batches = 0.0, 0
        for Xb, yb in _minibatches(Xtr, ytr, rng, cfg.batch_size):
            value, g = model.loss_and_grad(params, Xb, yb)
            params = opt.step(params, g)
            epoch_loss += value
            n_batches += 1

        p_val = model.predict_proba(params, Xva)
        val_auc = float(roc_auc_score(yva, p_val))
        val_loss, _ = model.loss_and_grad(params, Xva, yva)
        gnorm, gvar_param, gvar_all = _grad_stats(
            model, params, Xtr, ytr, rng, cfg.grad_probe_batches, cfg.batch_size
        )
        history.append(dict(
            epoch=epoch, train_loss=epoch_loss / max(n_batches, 1), val_loss=val_loss,
            val_auc=val_auc, grad_norm=gnorm, grad_var_param=gvar_param, grad_var=gvar_all,
            lr=opt.lr,
        ))
        if verbose:
            print(f"    ep {epoch:2d} | loss {epoch_loss / max(n_batches,1):.5f} | "
                  f"val_auc {val_auc:.4f} | |g| {gnorm:.3e} | Var[g] {gvar_all:.2e}")

        if val_auc > best["auc"] + cfg.min_auc_improvement:
            best = {"auc": val_auc, "params": params.copy(), "epoch": epoch}
            stalled = 0
        else:
            stalled += 1
            if stalled >= cfg.patience:
                break

    return RunResult(
        method_id=method_id or f"{arch}-{loss}-{readout}",
        seed=seed,
        params=best["params"],
        history=history,
        best_val_auc=best["auc"],
        best_epoch=best["epoch"],
        seconds=time.perf_counter() - started,
        n_params=circ.n_params,
        config=dict(arch=arch, loss=loss, n_layers=n_layers, n_wires=n_wires,
                    readout=readout, entangle=entangle, alpha=alpha,
                    scaler=split.scaler_name, n_features=split.n_features),
    )


def evaluate(model_cfg, params, split, threshold=None):
    """Metricas de test para un conjunto de parametros ya entrenado."""
    circ = build_circuit(
        model_cfg["arch"], split.n_features, model_cfg["n_layers"],
        model_cfg["n_wires"], model_cfg["readout"], model_cfg["entangle"],
    )
    model = DRUModel(circ, loss=model_cfg["loss"], alpha=model_cfg.get("alpha", 0.5))
    scores = model.predict_proba(params, split.X_test)
    return scores, float(roc_auc_score(split.y_test, scores))


def ensemble_scores(runs, split, which: str = "test") -> np.ndarray:
    """Promedio de probabilidades sobre semillas (ensemble del Entregable 1)."""
    X = getattr(split, f"X_{which}")
    members = []
    for r in runs:
        circ = build_circuit(
            r.config["arch"], split.n_features, r.config["n_layers"],
            r.config["n_wires"], r.config["readout"], r.config["entangle"],
        )
        model = DRUModel(circ, loss=r.config["loss"], alpha=r.config["alpha"])
        members.append(model.predict_proba(r.params, X))
    return np.mean(np.stack(members), axis=0)
