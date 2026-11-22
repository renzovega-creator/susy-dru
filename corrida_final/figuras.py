"""Figuras de la corrida final (lee resultados_test.csv; no entrena ni toca el test de nuevo).

Uso: python figuras.py   ->  figuras/*.png (dpi 200, listas para \\includegraphics)
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figuras"; FIG.mkdir(exist_ok=True)
R = pd.read_csv(HERE / "resultados_test.csv").set_index("modelo")

# paleta categórica validada (dataviz, orden fijo): azul, naranja, aqua, amarillo
C_TEST, C_VAL, C_3, C_4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
                     "figure.facecolor": "white", "savefig.bbox": "tight"})


def err(names):
    r = R.loc[names]
    return r.test_auc_ens, [r.test_auc_ens - r.ic95_lo, r.ic95_hi - r.test_auc_ens]


def dots(ax, names, labels, val=True, horizontal=False):
    y, e = err(names)
    pos = range(len(names))
    if horizontal:
        ax.errorbar(y, pos, xerr=e, fmt="o", ms=8, color=C_TEST, ecolor=C_TEST, elinewidth=2, capsize=0,
                    label="prueba · ensemble de 5 semillas (IC 95 %)")
        if val:
            ax.plot(R.loc[names].val_auc_ens, pos, "D", ms=7, mfc="white", mec=INK2, mew=2,
                    label="validación · ensemble")
        ax.set_yticks(list(pos), labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    else:
        ax.errorbar(pos, y, yerr=e, fmt="o", ms=8, color=C_TEST, ecolor=C_TEST, elinewidth=2, capsize=0,
                    label="prueba · ensemble de 5 semillas (IC 95 %)")
        if val:
            ax.plot(pos, R.loc[names].val_auc_ens, "D", ms=7, mfc="white", mec=INK2, mew=2,
                    label="validación · ensemble")
        ax.set_xticks(list(pos), labels); ax.grid(axis="x", visible=False)


# 1 · núcleo firmado y clásicos (forest plot)
q = ["M1", "M2", "M3-lin", "M3-cua", "M4-lin", "M4-cua"]
c = [m for m in ("RF", "MLP-73", "LogReg", "SVM lineal") if m in R.index]
lab = {"M1": "M1 · MinMax + XYZ + BCE", "M2": "M2 · Quantile + XYZ + BCE",
       "M3-lin": "M3 · pérdida lineal (Z)", "M3-cua": "M3 · pérdida cuadrática (Z)",
       "M4-lin": "M4 · lineal, entren. + valid.", "M4-cua": "M4 · cuadrática, entren. + valid.",
       "RF": "Random forest (ajustado)", "MLP-73": "MLP-73 (ajustado)", "LogReg": "Regresión logística",
       "SVM lineal": "SVM lineal"}
fig, ax = plt.subplots(figsize=(7.2, 4.6))
names = q + c
y, e = err(names)
cols = [C_TEST] * len(q) + [C_VAL] * len(c)
for i, (n, col) in enumerate(zip(names, cols)):
    ax.errorbar(y[n], i, xerr=[[e[0][n]], [e[1][n]]], fmt="o" if col == C_TEST else "s", ms=8, color=col,
                elinewidth=2, capsize=0)
    ax.text(R.loc[n].ic95_hi + 0.0015, i, f"{y[n]:.4f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(names)), [lab[n] for n in names]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.axhline(len(q) - 0.5, color=INK2, lw=0.8, ls=":")
ax.set_xlabel("AUC de prueba (ensemble; barras: IC 95 % bootstrap)")
ax.plot([], [], "o", color=C_TEST, label="cuántico (1 qubit, L = 5)")
ax.plot([], [], "s", color=C_VAL, label="clásico")
ax.legend(loc="upper left", fontsize=8.5)
ax.set_title("Diseño firmado M1–M4 frente a los clásicos (5000 eventos de prueba)", fontsize=10,
             loc="left", color=INK)
fig.savefig(FIG / "nucleo_y_clasicos.png", dpi=200); plt.close(fig)

# 2 · pérdidas del objetivo 2
pn = ["M3-lin", "Z híbrida α=0.75", "Z híbrida α=0.5", "Z híbrida α=0.25", "M3-cua", "Z BCE", "M2"]
pl = ["lineal\n(α = 1)", "híbrida\nα = 0.75", "híbrida\nα = 0.5", "híbrida\nα = 0.25", "cuadrática\n(α = 0)",
      "BCE\n(lectura Z)", "BCE\n(lectura XYZ)"]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
dots(ax, pn, pl)
ax.axvline(5.5, color=INK2, lw=0.8, ls=":")
ax.set_ylabel("AUC-ROC"); ax.legend(loc="lower right", fontsize=8.5)
ax.set_title("Funciones de pérdida (Quantile, L = 5)", fontsize=10, loc="left")
fig.savefig(FIG / "perdidas.png", dpi=200); plt.close(fig)

# 3 · rango angular × inicialización
fig, ax = plt.subplots(figsize=(6.4, 3.8))
hs = [("π/4", math.pi / 4), ("π/2", math.pi / 2), ("π", math.pi)]
for il, col, mk, dx in (("0.05·N", C_TEST, "o", -0.03), ("U[0,1)", C_3, "s", 0.03)):
    names = ["M2" if (h == "π" and il == "0.05·N") else f"h={h} {il}" for h, _ in hs]
    y, e = err(names)
    xs = [i + dx for i in range(3)]
    ax.errorbar(xs, y, yerr=e, fmt=mk + "-", ms=8, lw=2, color=col, elinewidth=2, capsize=0,
                label=f"inicialización {'0.05·N(0,1)' if il == '0.05·N' else 'U[0,1)'}")
ax.set_xticks(range(3), [f"h = {h}" for h, _ in hs]); ax.grid(axis="x", visible=False)
ax.set_ylabel("AUC de prueba (IC 95 %)"); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("Rango angular [−h, h] e inicialización (base M2)", fontsize=10, loc="left")
fig.savefig(FIG / "rango_inicializacion.png", dpi=200); plt.close(fig)

# 4 · capas
fig, ax = plt.subplots(figsize=(6.0, 3.6))
Ls = [1, 2, 3, 5, 8]
dots(ax, [("M2" if L == 5 else f"L={L}") for L in Ls], [str(L) for L in Ls])
ax.set_xlabel("número de capas L"); ax.set_ylabel("AUC-ROC"); ax.legend(fontsize=8.5, loc="lower right")
ax.set_title("Profundidad del circuito (base M2)", fontsize=10, loc="left")
fig.savefig(FIG / "capas.png", dpi=200); plt.close(fig)

# 5 · esquema del circuito (una capa)
fig, ax = plt.subplots(figsize=(9.5, 1.9)); ax.axis("off"); ax.grid(False)
ax.plot([0.2, 8.1], [0, 0], color=INK, lw=1.2, zorder=0)
ax.text(0.0, 0, r"$|0\rangle$", ha="right", va="center", fontsize=13)
gates = [("RZ", "lep1_pt"), ("RY", "lep2_pt"), ("RZ", "met"), ("RZ", "m_tr_2"), ("RY", "m_delta_r"),
         ("RZ", "lep1_eta"), ("RY", "lep2_eta")]
for i, (g, v) in enumerate(gates):
    x = 0.8 + i * 1.05
    ax.add_patch(plt.Rectangle((x - 0.42, -0.32), 0.84, 0.64, fc="white", ec=C_TEST, lw=1.6))
    ax.text(x, 0.04, g, ha="center", va="center", fontsize=10, color=INK)
    ax.text(x, -0.55, v, ha="center", va="top", fontsize=7.5, color=INK2)
for (a, b, t) in ((0, 2, "Rot(φ₁, φ₂, φ₃)"), (3, 5, "Rot(φ₄, φ₅, φ₆)"), (6, 6, "RY(φ₇)")):
    xa, xb = 0.8 + a * 1.05 - 0.45, 0.8 + b * 1.05 + 0.45
    ax.add_patch(plt.Rectangle((xa, -0.42), xb - xa, 0.84, fc="none", ec=INK2, lw=0.9, ls="--"))
    ax.text((xa + xb) / 2, 0.55, t, ha="center", va="bottom", fontsize=9, color=INK)
ax.text(8.3, 0, r"$\times\,5$ capas", ha="left", va="center", fontsize=11)
ax.text(4.0, -1.15, r"$\varphi_{l j} = w_{l j}\,x_j + b_{l j}$   (70 parámetros)   ·   lectura: "
        r"$p=\sigma(a\langle X\rangle+b\langle Y\rangle+c\langle Z\rangle+d)$ (M1, M2) o "
        r"$p=(1-\langle Z\rangle)/2$ (M3, M4)", ha="center", va="top", fontsize=8.5, color=INK2)
ax.set_xlim(-0.4, 9.7); ax.set_ylim(-1.5, 1.0)
fig.savefig(FIG / "circuito_capa.png", dpi=200); plt.close(fig)
print("figuras en", FIG)
