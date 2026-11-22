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
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
D = pd.read_csv(HERE / "barren_plateaus_500" / "barren_plateaus.csv")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 16, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "legend.frameon": False, "axes.titlesize": 15, "savefig.bbox": "tight"})

fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
A = D[D.scan == "A"]
for loss, col, mk, lab in (("cerezo", C1, "o", "pérdida lineal = costo local"),
                           ("fidelity", C3, "^", "pérdida cuadrática"), ("bce", INK2, "s", "BCE")):
    s = A[A.loss == loss].sort_values("n_layers")
    ax[0].semilogy(s.n_layers, s.var_grad, mk + "-", color=col, lw=2, ms=7, label=lab)
ax[0].set(xlabel="número de capas $L$", ylabel=r"$\mathrm{Var}[\partial \mathcal{L}/\partial w]$",
          title="(a) un qubit: profundidad")
ax[0].legend(fontsize=12, loc="center right")

B = D[D.scan == "B"]
for cost, col, mk, lab in (("global", C2, "s", "costo global"), ("local", C1, "o", "costo local")):
    s = B[B.cost == cost].sort_values("n_wires")
    ax[1].semilogy(s.n_wires, s.var_grad, mk + "-", color=col, lw=2, ms=6, label=lab)
ref = B[B.cost == "global"].sort_values("n_wires")
base = float(ref.var_grad.iloc[0]); n0 = int(ref.n_wires.iloc[0])
ax[1].semilogy(ref.n_wires, base * 4.0 ** -(ref.n_wires - n0), "--", color=INK2, lw=1.2, label=r"referencia $4^{-n}$")
ax[1].set(xlabel="número de qubits $n$", ylabel=r"$\mathrm{Var}[\partial \mathcal{L}/\partial w]$",
          title="(b) $L=4$: número de qubits")
ax[1].legend(fontsize=13)

Cc = D[D.scan == "C"]
cols = {2: INK, 4: INK2, 6: "#9a9893"}   # grises: el color queda para local/global
for (n, cost), s in Cc.groupby(["n_wires", "cost"]):
    s = s.sort_values("n_layers")
    ax[2].semilogy(s.n_layers, s.var_grad_mean_params, ("o" if cost == "local" else "s"),
                   ls="-" if cost == "local" else "--", color=cols[n], lw=2, ms=6)
    if cost == "local":
        ax[2].annotate(f"$n={n}$", (s.n_layers.iloc[-1], s.var_grad_mean_params.iloc[-1]), xytext=(8, 0),
                       textcoords="offset points", va="center", color=cols[n], fontsize=14)
ax[2].set(xlabel="número de capas $L$", ylabel=r"$\langle\mathrm{Var}[\partial \mathcal{L}/\partial\theta]\rangle_\theta$",
          title="(c) varios qubits: profundidad")
ax[2].set_xlim(right=9.4); ax[2].set_xticks([1, 2, 4, 8])
ax[2].legend(handles=[Line2D([], [], color=INK2, ls="-", marker="o", lw=2, label="costo local"),
                      Line2D([], [], color=INK2, ls="--", marker="s", lw=2, label="costo global")],
             fontsize=13, loc="lower right")
fig.tight_layout()
(HERE / "figuras").mkdir(exist_ok=True)
fig.savefig(HERE / "figuras" / "barren_plateaus.png", dpi=200)
print("ok")
