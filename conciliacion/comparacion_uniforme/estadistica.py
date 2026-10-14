"""Estadística para comparar AUC sobre un mismo test.

- DeLong et al. (1988) para dos AUC correlacionados, con el algoritmo rápido de Sun y Xu (2014)
  basado en rangos medios.
- Corrección de Holm (1979) para comparaciones múltiples.
- IC del AUC por bootstrap estratificado (remuestreo con reemplazo dentro de cada clase).

Implementación propia del repositorio. Se verificó que da los mismos AUC, z, p e IC que la del
notebook DRC_SUSY_protocolo_encoding_v2.ipynb (celda 13), usada en la conciliación del 14/10/2026.
"""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def _componentes(y_true, scores):
    """AUC y componentes estructurales de DeLong para un clasificador."""
    y_true = np.asarray(y_true)
    pos, neg = scores[y_true == 1], scores[y_true == 0]
    m, n = len(pos), len(neg)
    r_pos = stats.rankdata(pos)                      # rangos medios dentro de los positivos
    r_neg = stats.rankdata(neg)                      # rangos medios dentro de los negativos
    r_all = stats.rankdata(np.concatenate([pos, neg]))
    auc = (r_all[:m].sum() - m * (m + 1) / 2) / (m * n)
    v_pos = (r_all[:m] - r_pos) / n                  # V10: fracción de negativos por debajo de cada positivo
    v_neg = 1.0 - (r_all[m:] - r_neg) / m            # V01: fracción de positivos por encima de cada negativo
    return auc, v_pos, v_neg


def delong_test(y_true, scores_a, scores_b):
    """Prueba de DeLong para H0: AUC_a = AUC_b. Devuelve (auc_a, auc_b, z, p bilateral)."""
    auc_a, pa, na = _componentes(y_true, np.asarray(scores_a, float))
    auc_b, pb, nb = _componentes(y_true, np.asarray(scores_b, float))
    m, n = len(pa), len(na)
    cov = np.cov(np.vstack([pa, pb])) / m + np.cov(np.vstack([na, nb])) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 1e-15:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    return auc_a, auc_b, z, 2 * stats.norm.sf(abs(z))


def holm_adjust(p_values):
    """p-valores ajustados por Holm (step-down), monótonos y acotados en 1."""
    p = np.asarray(p_values, float)
    orden = np.argsort(p)
    ajustados = np.empty_like(p)
    acumulado = 0.0
    for k, idx in enumerate(orden):
        acumulado = max(acumulado, (len(p) - k) * p[idx])
        ajustados[idx] = min(1.0, acumulado)
    return ajustados


def bootstrap_auc_ci(y_true, scores, n_boot=2000, level=0.95, seed=0):
    """IC del AUC por bootstrap estratificado: positivos y negativos se remuestrean por separado."""
    y_true = np.asarray(y_true)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    aucs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        aucs[i] = roc_auc_score(y_true[idx], scores[idx])
    lo, hi = np.quantile(aucs, [(1 - level) / 2, (1 + level) / 2])
    return lo, hi
