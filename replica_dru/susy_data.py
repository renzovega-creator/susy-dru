"""
Carga y preparacion del dataset SUSY (UCI) para el proyecto SUSY_DRU.

Replica y unifica el manejo de datos de:
  - Baldi, Sadowski & Whiteson (2014)  -> definicion de las 18 variables y del
    split oficial (las ultimas 500.000 filas son el test del paper).
  - Terashi et al. (2021), repo QML_HEP -> subsample balanceado + min-max.
  - Muten, Tlimakhov & Tomut (2021), QHack -> subsample pequeno (5k) + 7 features.
  - Notebook `Entregable1_propuesta(1).ipynb` -> test fijo, transformadores
    ajustados SOLO en train, verificacion SHA-256 de integridad.

Sin dependencias fuera de numpy/pandas/scikit-learn.
"""

from __future__ import annotations

import hashlib
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler

SUSY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"

# Nombres oficiales (Baldi et al. 2014, Tabla 1 del suplemento).
COLUMN_NAMES = [
    "label",
    # --- 8 low-level ---
    "lep1_pt", "lep1_eta", "lep1_phi",
    "lep2_pt", "lep2_eta", "lep2_phi",
    "met", "met_phi",
    # --- 10 high-level ---
    "met_rel", "axial_met", "mr", "mtr2", "r", "mt2",
    "s_r", "m_delta_r", "dphi_r_b", "cos_theta_r1",
]

LOW_LEVEL = COLUMN_NAMES[1:9]
HIGH_LEVEL = COLUMN_NAMES[9:]
ALL_18 = COLUMN_NAMES[1:]

# Subconjunto de 7 variables usado en el avance del grupo (Entregable 1).
# Es el mismo de Muten et al. + lep2_eta.
FEATURES_7 = ["lep1_pt", "lep2_pt", "met", "mtr2", "m_delta_r", "lep1_eta", "lep2_eta"]
FEATURES_6 = FEATURES_7[:6]

DEFAULT_CACHE = Path(__file__).resolve().parent / "data"


# ---------------------------------------------------------------------------
# Descarga / cache
# ---------------------------------------------------------------------------
def download_susy_head(n_rows: int = 60_000, url: str = SUSY_URL) -> list[str]:
    """Descomprime al vuelo solo las primeras `n_rows` filas del SUSY.csv.gz."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    decompressor = zlib.decompressobj(31)  # 31 = cabecera gzip
    pending, rows = "", []
    with urlopen(request, timeout=600) as response:
        while len(rows) <= n_rows:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            pending += decompressor.decompress(chunk).decode("utf-8")
            parts = pending.split("\n")
            pending = parts.pop()
            rows.extend(parts)
    return rows[:n_rows]


def load_susy(
    n_rows: int = 60_000,
    cache_dir: Path | str = DEFAULT_CACHE,
    expected_sha256: str | None = None,
) -> pd.DataFrame:
    """Devuelve un DataFrame con las primeras `n_rows` filas de SUSY.

    Cachea el CSV en `cache_dir` para no volver a descargar. Si se pasa
    `expected_sha256`, verifica la integridad del texto descargado (mismo
    contrato que el notebook del Entregable 1).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"susy_head_{n_rows}.csv"

    if cache_file.exists():
        csv_text = cache_file.read_text(encoding="utf-8")
    else:
        rows = download_susy_head(n_rows)
        if len(rows) < n_rows:
            raise RuntimeError(f"Solo se descargaron {len(rows)} filas de {n_rows}.")
        csv_text = "\n".join(rows) + "\n"
        cache_file.write_text(csv_text, encoding="utf-8")

    observed = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(f"SHA-256 no coincide: {observed} != {expected_sha256}")

    from io import StringIO

    df = pd.read_csv(StringIO(csv_text), header=None, names=COLUMN_NAMES)
    df["label"] = df["label"].astype(int)
    df.attrs["sha256"] = observed
    df.attrs["n_rows"] = n_rows
    return df


# ---------------------------------------------------------------------------
# Particion balanceada con test fijo
# ---------------------------------------------------------------------------
def _balanced_indices(labels: np.ndarray, per_class: int, rng, excluded=None) -> np.ndarray:
    excluded = np.array([], dtype=int) if excluded is None else np.asarray(excluded, dtype=int)
    mask = np.ones(len(labels), dtype=bool)
    mask[excluded] = False
    picked = []
    for class_id in (0, 1):
        candidates = np.flatnonzero((labels == class_id) & mask)
        if len(candidates) < per_class:
            raise ValueError(f"Clase {class_id}: {len(candidates)} disponibles < {per_class}.")
        picked.append(rng.choice(candidates, size=per_class, replace=False))
    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


@dataclass
class SusySplit:
    """Particion balanceada, con test congelado antes de cualquier ajuste."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    features: list[str]
    scaler_name: str
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def summary(self) -> str:
        return (
            f"features={self.n_features} ({self.scaler_name}) | "
            f"train={len(self.y_train)} val={len(self.y_val)} test={len(self.y_test)} | "
            f"rango train=[{self.X_train.min():.3f}, {self.X_train.max():.3f}]"
        )


def make_split(
    df: pd.DataFrame,
    features: list[str] | None = None,
    scaler: str = "quantile",
    train_per_class: int = 4000,
    val_per_class: int = 1000,
    test_per_class: int = 4000,
    test_seed: int = 314159,
    dev_seed: int = 2021,
    angle_range: tuple[float, float] = (-np.pi, np.pi),
    pad_to_multiple_of: int | None = None,
) -> SusySplit:
    """Construye la particion (test primero, transformadores solo con train).

    `scaler` in {"minmax", "quantile", "standard", "identity"}.
    `pad_to_multiple_of=3` anade columnas de ceros hasta completar multiplo de 3
    (necesario para el ansatz de Perez-Salinas, que consume tripletes de Euler).
    """
    features = list(features) if features is not None else list(FEATURES_7)
    labels = df["label"].to_numpy(dtype=int)

    test_rng = np.random.default_rng(test_seed)
    test_idx = _balanced_indices(labels, test_per_class, test_rng)

    dev_rng = np.random.default_rng(dev_seed)
    train_idx = _balanced_indices(labels, train_per_class, dev_rng, excluded=test_idx)
    val_idx = _balanced_indices(
        labels, val_per_class, dev_rng, excluded=np.concatenate([test_idx, train_idx])
    )

    assert set(test_idx).isdisjoint(train_idx)
    assert set(test_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(val_idx)

    values = df[features].to_numpy(dtype=np.float64)
    Xtr_raw, Xva_raw, Xte_raw = values[train_idx], values[val_idx], values[test_idx]

    lo, hi = angle_range
    if scaler == "identity":
        Xtr, Xva, Xte = Xtr_raw.copy(), Xva_raw.copy(), Xte_raw.copy()
    elif scaler == "minmax":
        tf = MinMaxScaler(feature_range=(lo, hi))
        Xtr, Xva, Xte = tf.fit_transform(Xtr_raw), tf.transform(Xva_raw), tf.transform(Xte_raw)
    elif scaler == "standard":
        tf = StandardScaler()
        Xtr, Xva, Xte = tf.fit_transform(Xtr_raw), tf.transform(Xva_raw), tf.transform(Xte_raw)
    elif scaler == "quantile":
        tf = QuantileTransformer(
            n_quantiles=min(1000, len(Xtr_raw)),
            output_distribution="uniform",
            random_state=dev_seed,
        )
        to_angle = lambda v: lo + (hi - lo) * v
        Xtr = to_angle(tf.fit_transform(Xtr_raw))
        Xva = to_angle(tf.transform(Xva_raw))
        Xte = to_angle(tf.transform(Xte_raw))
    else:
        raise ValueError(f"scaler desconocido: {scaler}")

    if pad_to_multiple_of:
        m = pad_to_multiple_of
        pad = (-len(features)) % m
        if pad:
            zeros = lambda A: np.hstack([A, np.zeros((len(A), pad))])
            Xtr, Xva, Xte = zeros(Xtr), zeros(Xva), zeros(Xte)
            features = features + [f"_pad{i}" for i in range(pad)]

    return SusySplit(
        X_train=Xtr, y_train=labels[train_idx].astype(np.float64),
        X_val=Xva, y_val=labels[val_idx].astype(np.float64),
        X_test=Xte, y_test=labels[test_idx].astype(np.float64),
        features=features,
        scaler_name=scaler,
        meta=dict(
            test_seed=test_seed, dev_seed=dev_seed, angle_range=angle_range,
            n_rows_source=int(df.attrs.get("n_rows", len(df))),
            sha256=df.attrs.get("sha256", ""),
        ),
    )


if __name__ == "__main__":  # smoke test
    frame = load_susy(20_000)
    split = make_split(frame, train_per_class=500, val_per_class=200, test_per_class=500)
    print(split.summary())
    print("balance train:", split.y_train.mean(), "| test:", split.y_test.mean())
