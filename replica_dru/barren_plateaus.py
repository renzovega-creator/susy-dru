"""
Diagnostico de barren plateaus para el clasificador DRU sobre SUSY.
(Cronograma de la propuesta firmada: bloque de agosto 2026.)

Reproduce el protocolo estandar de McClean et al. (2018) y Cerezo et al. (2021):
inicializar los parametros al azar muchas veces y medir Var[dL/dtheta_k] para un
parametro fijo, como funcion de (a) la profundidad L y (b) el numero de qubits n,
comparando un coste LOCAL (<Z_0>) con uno GLOBAL (<Z_0 Z_1 ... Z_{n-1}>).

Lo que este experimento tiene que dejar claro en el informe
-----------------------------------------------------------
Con n = 1 qubit NO hay barren plateau exponencial: la cota de Cerezo et al.
decae como 2^{-n} y para n = 1 es O(1). Ademas el proyector "global" |0><0| de un
solo qubit ES un observable local, asi que las perdidas de fidelidad y la local
de Cerezo coinciden exactamente (ver test_gradients.py, prueba 2). Por tanto la
comparacion global-vs-local de la propuesta SOLO es una pregunta con contenido
si se hace a n >= 2. Este script mide las dos cosas: el barrido en L a n = 1
(donde lo que se observa es el decaimiento de gradientes propio del
re-uploading, Barthe & Perez-Salinas 2024) y el barrido en n (donde aparece la
separacion global/local de verdad).

Uso:
    python barren_plateaus.py --samples 200
Salidas: resultados/barren_plateaus.csv y figuras/barren_plateaus.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dru_core import DRUModel, build_circuit
from susy_data import FEATURES_7, load_susy, make_split

OUT = Path(__file__).resolve().parent / "resultados"
FIG = OUT / "figuras"


def probe_index(circ) -> int:
    """Parametro sonda: el peso de la primera RY.

    La primerisima RZ actua sobre |0>, autoestado de Z, asi que su gradiente es
    identicamente cero (fase global) y medir la varianza ahi no dice nada.

    Aviso: incluso esta sonda puede ser estructuralmente nula en casos
    degenerados (p. ej. n=2, L=1 con anillo de CNOT: los dos CNOT transforman
    Z_0 en Z_1 en la imagen de Heisenberg, que no depende de los parametros del
    hilo 0). Por eso `scan` reporta tambien `var_grad_mean_params`, la varianza
    promediada sobre TODOS los parametros, que es inmune a este artefacto.
    """
    for k, g in enumerate(circ.gates):
        if g.pauli == "Y" and g.w_idx is not None:
            return g.w_idx
    return circ.w_indices[0]


def scan(X, y, n_samples, layers_list, wires_list, losses=("cerezo",), seed0=0):
    rows = []
    for n_wires in wires_list:
        readouts = ("z",) if n_wires == 1 else ("z", "parity")
        for readout in readouts:
            for n_layers in layers_list:
                circ = build_circuit("muten", X.shape[1], n_layers, n_wires,
                                     readout, entangle=n_wires > 1)
                k = probe_index(circ)
                for loss in losses:
                    model = DRUModel(circ, loss=loss)
                    grads, full_grads = [], []
                    rng = np.random.default_rng(1234 + n_layers + 97 * n_wires)
                    for s in range(n_samples):
                        # Inicializacion Haar-like: angulos uniformes en [0, 2pi)
                        p = np.random.default_rng(seed0 + s).uniform(
                            0, 2 * np.pi, size=circ.n_params)
                        idx = rng.choice(len(X), size=64, replace=False)
                        _, g = model.loss_and_grad(p, X[idx], y[idx])
                        grads.append(g[k])
                        full_grads.append(g)
                    grads = np.asarray(grads)          # (n_samples,) parametro sonda
                    full = np.asarray(full_grads)      # (n_samples, n_params)
                    rows.append(dict(
                        n_wires=n_wires, readout=readout, cost="local" if readout == "z" else "global",
                        n_layers=n_layers, loss=loss, n_params=circ.n_params,
                        var_grad=float(np.var(grads)), mean_abs=float(np.mean(np.abs(grads))),
                        var_grad_mean_params=float(np.mean(np.var(full, axis=0))),
                        samples=n_samples))
                    print(f"  n={n_wires} {rows[-1]['cost']:6s} L={n_layers:2d} {loss:8s} "
                          f"Var_sonda={rows[-1]['var_grad']:.4e}  "
                          f"Var_media={rows[-1]['var_grad_mean_params']:.4e}  "
                          f"E|g|={rows[-1]['mean_abs']:.4e}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--rows", type=int, default=60_000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    df = load_susy(args.rows)
    sp = make_split(df, features=FEATURES_7, scaler="quantile",
                    train_per_class=1000, val_per_class=200, test_per_class=200)
    X, y = sp.X_train, sp.y_train

    print("A) Barrido en profundidad (n = 1 qubit)")
    dep = scan(X, y, args.samples, [1, 2, 3, 5, 8, 12, 16, 24], [1],
               losses=("cerezo", "fidelity", "bce"))
    print("\nB) Barrido en numero de qubits a profundidad fija: coste local vs global")
    qub = scan(X, y, args.samples, [4], [1, 2, 3, 4, 5, 6], losses=("cerezo",))

    print("\nC) Interaccion profundidad x numero de qubits (cono de luz)")
    print("   La garantia polinomica de Cerezo et al. para el coste LOCAL exige")
    print("   profundidad O(log n): con L grande el cono de luz del observable")
    print("   local ya cubre todos los qubits y la separacion desaparece.")
    grid = scan(X, y, args.samples, [1, 2, 4, 8], [2, 4, 6], losses=("cerezo",))
    grid["scan"] = "C"

    dep["scan"], qub["scan"] = "A", "B"
    table = pd.concat([dep, qub, grid], ignore_index=True)
    table.to_csv(OUT / "barren_plateaus.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for loss, sub in dep.groupby("loss"):
        axes[0].semilogy(sub.n_layers, sub.var_grad, marker="o", label=loss)
    axes[0].set(xlabel="numero de capas L", ylabel=r"Var$[\partial L/\partial w_k]$",
                title="n = 1 qubit: efecto de la profundidad")
    axes[0].legend()
    axes[0].grid(alpha=.3)

    for cost, sub in qub.groupby("cost"):
        axes[1].semilogy(sub.n_wires, sub.var_grad, marker="s",
                         label=f"coste {cost}")
    ref = qub[qub.cost == "global"]
    if len(ref):
        base = float(ref.var_grad.iloc[0])
        axes[1].semilogy(ref.n_wires, base * 4.0 ** -(ref.n_wires - ref.n_wires.iloc[0]),
                         "k--", alpha=.6, label=r"referencia $4^{-n}$")
    axes[1].set(xlabel="numero de qubits n", ylabel=r"Var$[\partial L/\partial w_k]$",
                title="L = 4: coste local vs global (Cerezo et al.)")
    axes[1].legend()
    axes[1].grid(alpha=.3)

    # Aqui se usa la varianza promediada sobre TODOS los parametros: la sonda
    # unica se anula estructuralmente en L=1 (ver docstring de probe_index) y
    # arruina la escala logaritmica.
    for (n_wires, cost), sub in grid.groupby(["n_wires", "cost"]):
        axes[2].semilogy(sub.n_layers, sub.var_grad_mean_params, marker="^",
                         ls="-" if cost == "local" else "--",
                         label=f"n={n_wires} {cost}")
    axes[2].set(xlabel="numero de capas L",
                ylabel=r"$\langle$Var$[\partial L/\partial\theta]\rangle_{\rm params}$",
                title="Cono de luz: la separacion vive en L pequeno")
    axes[2].legend(fontsize=7, ncol=2)
    axes[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "barren_plateaus.png", dpi=150)
    print(f"\nSalidas en {OUT}")


if __name__ == "__main__":
    main()
