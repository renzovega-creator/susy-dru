"""Convierte los notebooks del equipo en scripts que corren sobre el split común y vuelcan sus scores.

Los notebooks NO están en este repositorio (entran solo con el visto bueno de sus autores).
Uso:  SUSY_NOTEBOOKS=/ruta/a/carpeta/con/los/3/notebooks  python build_subs.py
      -> subs_unif_*.json y run_four.py, run_angello.py, run_sebastian.py (ignorados por git)

Solo cambia: de dónde salen los datos/índices, las semillas, y añade el volcado de scores.
Modelo, preprocesamiento, entrenamiento y selección quedan como en cada notebook.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
COP = Path(os.environ["SUSY_NOTEBOOKS"]).resolve()
SPLIT, OUT = str(HERE / "split_comun.npz"), str(HERE / "scores")
CACHE = str(HERE.parents[1] / "replica_dru" / "data" / "susy_head_60000.csv")


def cell(nb, i):
    return "".join(json.load(open(COP / nb, encoding="utf-8"))["cells"][i - 1]["source"])


# Parche de compatibilidad: con PennyLane 0.42 + NumPy 2.5, tensor_pennylane * ArrayBox devuelve
# NotImplemented. Se invierte el orden de los factores (matemáticamente idéntico).
COMPAT = {
    "angello": [[19, a, b] for a, b in [
        ("targets * np.log(probabilities + epsilon)", "np.log(probabilities + epsilon) * targets"),
        ("(1.0 - targets) * np.log(1.0 - probabilities + epsilon)", "np.log(1.0 - probabilities + epsilon) * (1.0 - targets)"),
        ("targets * probabilities\n", "probabilities * targets\n"),
        ("(1.0 - targets) * (1.0 - probabilities)", "(1.0 - probabilities) * (1.0 - targets)")]],
    "four": [[28, a, b] for a, b in [
        ("y_batch * np.log(probabilities + epsilon)", "np.log(probabilities + epsilon) * y_batch"),
        ("(1.0 - y_batch) * np.log(1.0 - probabilities + epsilon)", "np.log(1.0 - probabilities + epsilon) * (1.0 - y_batch)"),
        ("y_batch * signal_probability\n", "signal_probability * y_batch\n"),
        ("(1.0 - y_batch) * (1.0 - signal_probability)", "(1.0 - signal_probability) * (1.0 - y_batch)")]],
}


def compat(name):
    return COMPAT[name]


# ---------------- four_models: una semilla por proceso (env UNIF_SEED) ----------------
F = "final_quantum_susy_1q_four_models.ipynb"
four = {
    "skip_cells": [6] + list(range(37, 59)),
    "subs": compat("four") + [
        [10, "MODEL_SEEDS = [101, 202, 303]", "import os\nMODEL_SEEDS = [int(os.environ['UNIF_SEED'])]  # UNIF"],
        [13, 'Path("SUSY.csv"),', f'Path("{CACHE}"),  # UNIF: cache local'],
        [36, 'quantum_results.sort_values(["method_id", "val_auc"], ascending=[True, False])',
         f"""_out = Path("{OUT}"); _out.mkdir(exist_ok=True)
_S = onp.load("{SPLIT}")
for _k in ("train", "val", "test"):
    assert onp.array_equal(split_indices[_k], _S[_k]), "split distinto: " + _k
print("UNIF: split de four_models == split_comun (train/val/test idénticos)")
for _r in quantum_runs:
    onp.savez(_out / f"four__{{_r['method_id']}}__seed{{_r['seed']}}.npz",
              val=onp.asarray([] if _r["val_scores"] is None else _r["val_scores"], dtype=float),
              test=onp.asarray(_r["test_scores"], dtype=float), n_params=_r["n_parameters"])
if MODEL_SEEDS[0] == 101:
    for _n, _a in classical_artifacts.items():
        onp.savez(_out / f"four__{{_n.replace(' ', '_')}}__seed0.npz", val=onp.asarray(_a["val_scores"], dtype=float),
                  test=onp.asarray(_a["test_scores"], dtype=float))
quantum_results.sort_values(["method_id", "val_auc"], ascending=[True, False])"""],
    ],
}

# ---------------- Entregable 1 (Angello): modo paper, Q0-Q4 x 101/202/303 ----------------
A = "Entregable1_propuesta(1).ipynb"
c9 = cell(A, 9)
angello = {
    "skip_cells": [5, 17, 23] + list(range(30, 47)),
    "subs": compat("angello") + [
        [9, "rows = download_susy_head(SUSY_URL, N_ROWS)",
         f'rows = open("{CACHE}", encoding="utf-8").read().splitlines()[:N_ROWS]  # UNIF: cache local'],
        [10, "split_rows = []",
         f'_S = onp.load("{SPLIT}")  # UNIF: índices del split común\n'
         'train_indices, validation_indices, test_indices = _S["train"], _S["val"], _S["test"]\n'
         'assert onp.array_equal(labels[test_indices], _S["y_test"])\nsplit_rows = []'],
        [29, "quantum_individual_results = pd.DataFrame(quantum_individual_rows)",
         f"""_out = Path("{OUT}"); _out.mkdir(exist_ok=True)
for _mid, _art in quantum_artifacts.items():
    for _s, _v, _t in zip(MODEL_SEEDS, _art["validation_member_scores"], _art["test_member_scores"]):
        onp.savez(_out / f"angello__{{_mid}}__seed{{_s}}.npz", val=_v, test=_t,
                  n_params=effective_parameter_count(_art["method"]))
for _n, _m in [("LogReg", logistic_model), ("RF", random_forest_model)]:
    onp.savez(_out / f"angello__{{_n}}__seed0.npz", val=_m.predict_proba(X_validation_classical)[:, 1],
              test=_m.predict_proba(X_test_classical)[:, 1])
for _s, _m in zip(MODEL_SEEDS, mlp_models):
    onp.savez(_out / f"angello__MLP-73__seed{{_s}}.npz", val=_m.predict_proba(X_validation_classical)[:, 1],
              test=_m.predict_proba(X_test_classical)[:, 1])
print("UNIF: scores de Angello guardados")
quantum_individual_results = pd.DataFrame(quantum_individual_rows)"""],
    ],
}
assert "rows = download_susy_head(SUSY_URL, N_ROWS)" in c9

# ---------------- Sebastián: E0/E1/E2 (receta) + exploratorias (env UNIF_MODO) ----------------
D = "DRC_SUSY_protocolo_encoding_v2.ipynb"
c7 = cell(D, 7)
seb = {
    "skip_cells": [9, 15, 16, 17, 19, 23, 29, 30, 32, 33],
    "subs": [
        [3, 'subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or module_name])',
         'raise RuntimeError(f"Falta {module_name}: instalación neutralizada")'],
        [4, 'VARIANTS = ["E0", "E1", "E2"]     # E0 = encoding original',
         'import os\nVARIANTS = os.environ["UNIF_VARIANTS"].split(",")  # UNIF'],
        [4, "FEATURE_IDS = [0, 3, 6, 11, 15, 1]", "FEATURE_IDS = [0, 3, 6, 11, 15, 1, 4][:int(os.environ['UNIF_NFEAT'])]  # UNIF"],
        [4, "NORMALIZATION = None ", "NORMALIZATION = None if os.environ['UNIF_NORM'] == 'none' else os.environ['UNIF_NORM'] "],
        [4, "RUN_PENNYLANE_CHECK = True", "RUN_PENNYLANE_CHECK = False"],
        [4, 'RESULTS_DIR = Path("results_protocol")', "RESULTS_DIR = Path(os.environ['UNIF_RESDIR'])"],
        [4, "SEEDS = list(range(N_SEEDS))", "SEEDS = [101, 202, 303]  # UNIF: mismas semillas en los 4 códigos"],
        [7, c7, f"""# UNIF: datos del split común (mismas filas de SUSY que el resto de códigos), 18 variables crudas
_S = np.load("{SPLIT}")
_raw = np.loadtxt("{CACHE}", delimiter=",", max_rows=26000)
_X, _y = _raw[:, 1:], _raw[:, 0].astype(int)
X_pool_all = np.vstack([_X[_S["train"]], _X[_S["val"]]])
y_pool = np.r_[_y[_S["train"]], _y[_S["val"]]]
X_test_all, y_test = _X[_S["test"]], _y[_S["test"]]
assert np.array_equal(y_test, _S["y_test"])
N_TR = len(_S["train"])
X_pool = X_pool_all[:, FEATURE_IDS]
X_test = X_test_all[:, FEATURE_IDS]
print("pool:", X_pool.shape, np.bincount(y_pool), "| test:", X_test.shape, np.bincount(y_test))"""],
        [11, '"E2": "XYXYXY",', '"E2": "XYXYXY",\n    "E0_7": "ZYZZYZY",   # UNIF: Rot·Rot·RY con 7 variables (= ansatz de los otros códigos)\n    "E2_7": "XYXYXYX",   # UNIF: E2 extendido a 7 variables'],
        [21, "    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=seed)\n    train_idx, val_idx = next(splitter.split(X_pool, y_pool))",
         "    train_idx, val_idx = np.arange(N_TR), np.arange(N_TR, len(y_pool))  # UNIF: train/val fijos del split común"],
    ],
}
assert c7.count("urllib.request.urlretrieve") == 1

for name, nbname, cfg in [("four", F, four), ("angello", A, angello), ("sebastian", D, seb)]:
    sub_path = HERE / f"subs_unif_{name}.json"
    json.dump(cfg, open(sub_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    subprocess.run([sys.executable, str(HERE / "nb2py.py"), str(COP / nbname), str(HERE / f"run_{name}.py"),
                    str(sub_path)], check=True)
