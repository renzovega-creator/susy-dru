"""Verificacion del nucleo: gradiente adjoint vs. diferencias finitas,
identidades analiticas entre perdidas y sanidad del entrelazamiento.

Ejecutar:  python test_gradients.py
"""

from __future__ import annotations

import itertools

import numpy as np

from dru_core import DRUModel, build_circuit

RNG = np.random.default_rng(7)


def finite_diff(model, params, X, y, h=1e-6):
    g = np.zeros_like(params)
    for i in range(len(params)):
        pp, pm = params.copy(), params.copy()
        pp[i] += h
        pm[i] -= h
        g[i] = (model.loss_and_grad(pp, X, y)[0] - model.loss_and_grad(pm, X, y)[0]) / (2 * h)
    return g


def check(arch, n_features, n_layers, n_wires, readout, loss, entangle=False):
    circ = build_circuit(arch, n_features, n_layers, n_wires, readout, entangle)
    model = DRUModel(circ, loss=loss, alpha=0.35, l2=1e-4)
    params = model.init_params(seed=3, scale=0.7)
    X = RNG.uniform(-np.pi, np.pi, size=(16, n_features))
    y = RNG.integers(0, 2, size=16).astype(float)

    _, g_adj = model.loss_and_grad(params, X, y)
    g_fd = finite_diff(model, params, X, y)
    denom = max(np.linalg.norm(g_fd), 1e-12)
    err = np.linalg.norm(g_adj - g_fd) / denom
    tag = f"{arch:16s} d={n_features:2d} L={n_layers} n={n_wires} {readout:6s} {loss:10s}"
    status = "OK " if err < 1e-5 else "FALLA"
    print(f"  [{status}] {tag}  err_rel={err:.2e}  |g|={np.linalg.norm(g_adj):.4f}")
    return err < 1e-5


def main():
    print("1) Gradiente adjoint vs. diferencias finitas")
    ok = True
    for arch, d in (("perez_salinas", 9), ("perez_salinas_w", 9), ("muten", 7)):
        for loss in ("fidelity", "infidelity", "cerezo", "bce", "mse", "hybrid"):
            ok &= check(arch, d, 3, 1, "z", loss)
    for readout in ("xyz", "z"):
        ok &= check("muten", 7, 4, 1, readout, "bce")
    print("   -- multi-qubit con entrelazamiento --")
    for readout, loss in itertools.product(("z", "parity"), ("cerezo", "fidelity", "bce")):
        ok &= check("muten", 7, 2, 3, readout, loss, entangle=True)

    print("\n2) Identidad n=1: cerezo == infidelity (proyector global = local)")
    circ = build_circuit("muten", 7, 4, 1, "z")
    p = DRUModel(circ).init_params(11, scale=0.6)
    X = RNG.uniform(-np.pi, np.pi, size=(64, 7))
    y = RNG.integers(0, 2, size=64).astype(float)
    lc, gc = DRUModel(circ, loss="cerezo").loss_and_grad(p, X, y)
    li, gi = DRUModel(circ, loss="infidelity").loss_and_grad(p, X, y)
    same = abs(lc - li) < 1e-14 and np.allclose(gc, gi, atol=1e-14)
    print(f"  [{'OK ' if same else 'FALLA'}] L_cerezo={lc:.12f}  L_infid={li:.12f}")
    ok &= same

    print("\n2b) Identidad n=1 + lectura Z: fidelidad cuadratica == MSE")
    # F_y = (1 + s_y <Z>)/2 y p = (1 - <Z>)/2  =>  1 - F_y = |p - y|
    # luego (1 - F_y)^2 = (p - y)^2 exactamente.
    lf, gf = DRUModel(circ, loss="fidelity").loss_and_grad(p, X, y)
    lm, gm = DRUModel(circ, loss="mse").loss_and_grad(p, X, y)
    same2 = abs(lf - lm) < 1e-14 and np.allclose(gf, gm, atol=1e-14)
    print(f"  [{'OK ' if same2 else 'FALLA'}] L_fidelidad={lf:.12f}  L_mse={lm:.12f}")
    ok &= same2
    print("      -> a 1 qubit con lectura Z solo hay TRES objetivos distintos:")
    print("         lineal (=Cerezo=infidelidad), cuadratico (=fidelidad=MSE) y log (=BCE).")

    print("\n3) El primer RZ sobre |0> no puede tener gradiente (fase global)")
    circ = build_circuit("muten", 7, 3, 1, "z")
    p0 = DRUModel(circ).init_params(5, scale=np.pi)
    Xs = RNG.uniform(-np.pi, np.pi, size=(32, 7))
    ys = RNG.integers(0, 2, size=32).astype(float)
    g0 = DRUModel(circ, loss="cerezo").loss_and_grad(p0, Xs, ys)[1]
    zero_ok = abs(g0[0]) < 1e-14 and abs(g0[1]) < 1e-14
    print(f"  [{'OK ' if zero_ok else 'FALLA'}] |dL/dw_0|={abs(g0[0]):.2e}  "
          f"|dL/db_0|={abs(g0[1]):.2e}  (esperado 0)")
    ok &= zero_ok

    print("\nRESULTADO:", "TODO OK" if ok else "HAY FALLAS")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
