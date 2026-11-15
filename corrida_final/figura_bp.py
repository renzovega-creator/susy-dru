"""Figura de barren plateaus en español, a partir de barren_plateaus_500/barren_plateaus.csv (500 inicializaciones).

Mismos tres paneles que replica_dru/barren_plateaus.py (A: profundidad con n = 1; B: n con L = 4;
C: cono de luz), con etiquetas en español y la paleta de las demás figuras. No recalcula nada.
Uso: python figura_bp.py  ->  figuras/barren_plateaus.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "barren_plateaus_500" / "barren_plateaus.csv")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False, "savefig.bbox": "tight"})

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
A = D[D.scan == "A"]
for loss, col, lab in (("cerezo", C1, "Cerezo local = fidelidad lineal"), ("fidelity", C3, "fidelidad cuadrática"),
                       ("bce", C2, "BCE")):
    s = A[A.loss == loss].sort_values("n_layers")
    ax[0].semilogy(s.n_layers, s.var_grad, "o-", color=col, lw=2, ms=6, label=lab)
ax[0].set(xlabel="número de capas $L$", ylabel=r"$\mathrm{Var}[\partial \mathcal{L}/\partial w]$",
          title="(a) un qubit: efecto de la profundidad")
ax[0].legend(fontsize=8)

B = D[D.scan == "B"]
for cost, col, mk, lab in (("global", C2, "s", "costo global"), ("local", C1, "o", "costo local")):
    s = B[B.cost == cost].sort_values("n_wires")
    ax[1].semilogy(s.n_wires, s.var_grad, mk + "-", color=col, lw=2, ms=6, label=lab)
ref = B[B.cost == "global"].sort_values("n_wires")
base = float(ref.var_grad.iloc[0]); n0 = int(ref.n_wires.iloc[0])
ax[1].semilogy(ref.n_wires, base * 4.0 ** -(ref.n_wires - n0), "--", color=INK2, lw=1.2, label=r"referencia $4^{-n}$")
ax[1].set(xlabel="número de qubits $n$", ylabel=r"$\mathrm{Var}[\partial \mathcal{L}/\partial w]$",
          title="(b) $L=4$: costo local y global")
ax[1].legend(fontsize=8)

Cc = D[D.scan == "C"]
cols = {2: C1, 4: C2, 6: C3}
for (n, cost), s in Cc.groupby(["n_wires", "cost"]):
    s = s.sort_values("n_layers")
    ax[2].semilogy(s.n_layers, s.var_grad_mean_params, ("o" if cost == "local" else "s"),
                   ls="-" if cost == "local" else "--", color=cols[n], lw=2, ms=6, label=f"$n={n}$, costo {cost}")
ax[2].set(xlabel="número de capas $L$", ylabel=r"$\langle\mathrm{Var}[\partial \mathcal{L}/\partial\theta]\rangle_\theta$",
          title="(c) separación local/global según $L$")
ax[2].legend(fontsize=7.5, ncol=2)
fig.tight_layout()
(HERE / "figuras").mkdir(exist_ok=True)
fig.savefig(HERE / "figuras" / "barren_plateaus.png", dpi=200)
print("ok")
