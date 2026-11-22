"""Genera las tablas LaTeX del informe a partir de los CSV de la corrida final (regla 3 del README del informe).

Uso: python tablas_latex.py <carpeta_destino>   (por defecto ./tablas_latex)
Lee resultados_test.csv, comparaciones_test.csv, clasicos_tuning_validacion.csv y
barren_plateaus_500/barren_plateaus.csv. No entrena nada ni recalcula estadística.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tablas_latex"
OUT.mkdir(parents=True, exist_ok=True)

R = pd.read_csv(HERE / "resultados_test.csv").set_index("modelo")
C = pd.read_csv(HERE / "comparaciones_test.csv")
BP = pd.read_csv(HERE / "barren_plateaus_500" / "barren_plateaus.csv")

CAB = ("% Generada por conciliacion_codigos/corrida_final/tablas_latex.py a partir de {src}.\n"
       "% No editar a mano: volver a correr el script.\n")


def f4(x):
    return "---" if pd.isna(x) else f"{x:.4f}"


def pfmt(p):
    if pd.isna(p):
        return "---"
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def dfmt(d):
    return "---" if pd.isna(d) else f"${d:+.4f}$".replace("+", "+").replace("-", "-")


def comp(modelo, ref, familia_prefijo):
    r = C[(C.modelo == modelo) & (C.referencia == ref) & C.familia.str.startswith(familia_prefijo)]
    if r.empty:
        return None
    return r.iloc[0]


def ic(m):
    return f"[{R.loc[m].ic95_lo:.4f}, {R.loc[m].ic95_hi:.4f}]"


def semillas(m):
    r = R.loc[m]
    if pd.isna(r.test_auc_sd):
        return "---"
    return f"{r.test_auc_media:.4f} $\\pm$ {r.test_auc_sd:.4f}"


def write(name, body, src):
    (OUT / name).write_text(CAB.format(src=src) + body, encoding="utf-8")


# ---------------------------------------------------------------- 1. núcleo M1-M4 + clásicos
filas = [("M1", r"M1 (MinMax, XYZ, BCE)"), ("M2", r"M2 (Quantile, XYZ, BCE)"),
         ("M3-lin", r"M3, fidelidad lineal"),
         ("M3-cua", r"M3, fidelidad cuadrática"),
         ("M4-lin", r"M4, fidelidad lineal"),
         ("M4-cua", r"M4, fidelidad cuadrática")]
clas = [("RF", r"\emph{Random forest}"), ("MLP-73", "MLP-73"), ("LogReg", "Regresión logística"),
        ("SVM lineal", "SVM lineal")]
lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
         r"Modelo & Parám. & $\AUC$ & IC 95\,\% & Por semilla & $\Delta$ vs M2 ($p_\mathrm{Holm}$) \\",
         r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Cuánticos (1 qubit, $L=5$, 5 semillas)}} \\"]
for m, lab in filas:
    c = comp(m, "M2", "F1 núcleo")
    d = "(referencia)" if c is None else f"{dfmt(c.delta_auc_ens)} ({pfmt(c.p_holm)})"
    lines.append(f"{lab} & {int(R.loc[m].n_params)} & {f4(R.loc[m].test_auc_ens)} & {ic(m)} & {semillas(m)} & {d} \\\\")
lines += [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{Clásicos (ajuste de hiperparámetros con validación)}} \\"]
for m, lab in clas:
    c = comp("M2", m, "F6")
    d = "---" if c is None else f"{dfmt(-c.delta_auc_ens)} ({pfmt(c.p_holm)})"
    npar = "---" if pd.isna(R.loc[m].n_params) else str(int(R.loc[m].n_params))
    lines.append(f"{lab} & {npar} & {f4(R.loc[m].test_auc_ens)} & {ic(m)} & {semillas(m)} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_nucleo.tex", "\n".join(lines), "resultados_test.csv y comparaciones_test.csv (familias F1 y F6)")

# ---------------------------------------------------------------- 2. pérdidas (F2)
perd = [("M3-lin", r"lineal, $\alpha=1$"), ("Z híbrida α=0.75", r"híbrida, $\alpha=0.75$"),
        ("Z híbrida α=0.5", r"híbrida, $\alpha=0.5$"), ("Z híbrida α=0.25", r"híbrida, $\alpha=0.25$"),
        ("M3-cua", r"cuadrática, $\alpha=0$"), ("Z BCE", r"BCE"), ("M2", r"BCE (lectura XYZ, = M2)")]
lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r"Pérdida & $\AUC$ & IC 95\,\% & Por semilla & $\Delta$ vs lineal ($p_\mathrm{Holm}$) \\",
         r"\midrule"]
for m, lab in perd:
    c = comp(m, "M3-lin", "F2")
    d = "(referencia)" if c is None else f"{dfmt(c.delta_auc_ens)} ({pfmt(c.p_holm)})"
    lines.append(f"{lab} & {f4(R.loc[m].test_auc_ens)} & {ic(m)} & {semillas(m)} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_perdidas.tex", "\n".join(lines), "resultados_test.csv y comparaciones_test.csv (familia F2)")

# ---------------------------------------------------------------- 3. rango x inicialización (F3)
lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule",
         r"$h$ & Inicialización & $\AUC$ & IC 95\,\% & Por semilla & $\Delta$ vs M2 ($p_\mathrm{Holm}$) \\",
         r"\midrule"]
for hlab, hname in ((r"$\pi/4$", "π/4"), (r"$\pi/2$", "π/2"), (r"$\pi$", "π")):
    for ilab, iname in ((r"$0.05\,\mathcal{N}(0,1)$", "0.05·N"), (r"$U[0,1)$", "U[0,1)")):
        m = "M2" if (hname == "π" and iname == "0.05·N") else f"h={hname} {iname}"
        c = comp(m, "M2", "F3")
        d = "(referencia)" if m == "M2" else f"{dfmt(c.delta_auc_ens)} ({pfmt(c.p_holm)})"
        lines.append(f"{hlab} & {ilab} & {f4(R.loc[m].test_auc_ens)} & {ic(m)} & {semillas(m)} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_rango.tex", "\n".join(lines), "resultados_test.csv y comparaciones_test.csv (familia F3)")

# ---------------------------------------------------------------- 4. capas (F4)
lines = [r"\begin{tabular}{@{}cccccc@{}}", r"\toprule",
         r"$L$ & Parám. & $\AUC$ & IC 95\,\% & Por semilla & $\Delta$ vs $L=5$ ($p_\mathrm{Holm}$) \\",
         r"\midrule"]
for L in (1, 2, 3, 5, 8):
    m = "M2" if L == 5 else f"L={L}"
    c = comp(m, "M2", "F4")
    d = "(referencia)" if L == 5 else f"{dfmt(c.delta_auc_ens)} ({pfmt(c.p_holm)})"
    lines.append(f"{L} & {int(R.loc[m].n_params)} & {f4(R.loc[m].test_auc_ens)} & {ic(m)} & {semillas(m)} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_capas.tex", "\n".join(lines), "resultados_test.csv y comparaciones_test.csv (familia F4)")

# ---------------------------------------------------------------- 5. barren plateaus, escaneo C
Cc = BP[BP.scan == "C"].pivot_table(index=["n_wires", "n_layers"], columns="cost", values="var_grad_mean_params")
Cc["r"] = Cc["local"] / Cc["global"]
Ls = sorted(BP[BP.scan == "C"].n_layers.unique())
lines = [r"\begin{tabular}{@{}c" + "c" * len(Ls) + "@{}}", r"\toprule",
         r" & \multicolumn{" + str(len(Ls)) + r"}{c}{$\langle\Var_\mathrm{local}\rangle / \langle\Var_\mathrm{global}\rangle$} \\",
         r"\cmidrule(l){2-" + str(len(Ls) + 1) + "}",
         "$n$ & " + " & ".join(f"$L={L}$" for L in Ls) + r" \\", r"\midrule"]
for n in sorted(BP[BP.scan == "C"].n_wires.unique()):
    lines.append(f"{n} & " + " & ".join(f"{Cc.loc[(n, L), 'r']:.2f}" for L in Ls) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_bp.tex", "\n".join(lines), "barren_plateaus_500/barren_plateaus.csv (escaneo C, 500 inicializaciones)")

# ---------------------------------------------------------------- 6. tuning de los clásicos
T = pd.read_csv(HERE / "clasicos_tuning_validacion.csv")
best = T.loc[T.groupby("familia").val_auc.idxmax()]
nombre = {"LogReg": "Regresión logística", "SVM lineal": "SVM lineal", "RF": r"\emph{Random forest}", "MLP-73": "MLP-73"}
grilla = {"LogReg": r"$C\in\{0.01, 0.1, 1, 10, 100\}$", "SVM lineal": r"$C\in\{0.01, 0.1, 1, 10\}$",
          "RF": r"400 árboles; profundidad $\in\{6, 8, 12, \text{sin límite}\}$; hoja mín. $\in\{1, 10\}$",
          "MLP-73": r"8 neuronas ocultas; $\alpha\in\{10^{-5}, 10^{-4}, 10^{-3}\}$; tasa $\in\{5\cdot10^{-4}, 10^{-3}, 3\cdot10^{-3}\}$"}
import json  # noqa: E402
lines = [r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{0.47\linewidth}>{\raggedright\arraybackslash}p{0.25\linewidth}@{}}",
         r"\toprule", r"Modelo & \emph{Grid} & Elegido (AUC de validación) \\", r"\midrule"]
for fam in ("LogReg", "SVM lineal", "RF", "MLP-73"):
    b = best[best.familia == fam].iloc[0]
    cfg = json.loads(b.config)
    txt = {"LogReg": lambda c: f"$C={c['C']:g}$", "SVM lineal": lambda c: f"$C={c['C']:g}$",
           "RF": lambda c: f"prof.\\ {c['max_depth'] if c['max_depth'] is not None else 'sin límite'}, hoja {c['min_samples_leaf']}",
           "MLP-73": lambda c: f"$\\alpha=10^{{{int(round(__import__('math').log10(c['alpha'])))}}}$, tasa $10^{{{int(round(__import__('math').log10(c['lr'])))}}}$"}[fam](cfg)
    lines.append(f"{nombre[fam]} & {grilla[fam]} & {txt} ({b.val_auc:.4f}) \\\\")
lines += [r"\bottomrule", r"\end{tabular}", ""]
write("tab_clasicos_tuning.tex", "\n".join(lines), "clasicos_tuning_validacion.csv")
print("tablas en", OUT)
