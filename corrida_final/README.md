# Corrida final (7/11/2026)

Evaluación única en test del diseño firmado (M1–M4) y de las familias del objetivo 2 (pérdidas, rango × inicialización,
capas, pérdida × lr), con baselines clásicos ajustados en validación. Es la corrida que alimenta los capítulos 3 y 4
del informe.

> **El test se puntuó una sola vez.** Las 23 configuraciones y las familias de comparación quedaron escritas en
> `config_corrida_final.json` antes de entrenar. `corrida_final.py` se niega a correr si ya existe
> `resultados_test.csv`. Todo lo que se elige (hiperparámetros de los clásicos, "mejor cuántico") se elige con validación.
>
> **Pendiente:** la auditoría independiente del motor (Angelo) no se hizo antes de lanzar esta corrida.

## Archivos

| Archivo | Qué es |
|---|---|
| `corrida_final.py` | Motor de entrenamiento + clásicos + estadística (DeLong, Holm, bootstrap) |
| `config_corrida_final.json` | Pre-registro: configuraciones con su `hp_key`, familias, semillas, split, SHA-256 y versiones |
| `requirements_corrida_final.txt` | Versiones exactas con las que se corrió |
| `resultados_test.csv` | 27 modelos: AUC test del ensemble, IC 95 %, media ± sd por semilla, AUC de validación, parámetros |
| `comparaciones_test.csv` | 36 comparaciones DeLong con p de Holm por familia (F1–F6) |
| `clasicos_tuning_validacion.csv` | Grilla de los clásicos y su AUC de validación |
| `cotejo_cache_notebook.csv` | M1, M2 y M3-lin frente a la caché de validación del notebook de hiperparámetros (\|Δ\| ≤ 9e-12) |
| `scores/<hp_key>_<semilla>.npz` | 115 corridas cuánticas: `val`, `test` (probabilidades), `loss_hist`, `val_hist`, `meta` |
| `figuras/`, `figuras.py`, `figura_bp.py` | Figuras del informe y los scripts que las generan desde los CSV |
| `tablas_latex.py` | Tablas LaTeX del informe desde los CSV (`python tablas_latex.py <carpeta_salida>`) |
| `barren_plateaus_500/`, `correr_barren_plateaus.py` | `replica_dru/barren_plateaus.py` con `--samples 500`, sin tocar `replica_dru/resultados/` |
| `corrida_final.log` | Salida completa de la corrida |

Para encontrar los puntajes de una configuración: `hp_key` está en `config_corrida_final.json`
(por ejemplo, M2 = `69e93f3adf2f`, así que su semilla 101 es `scores/69e93f3adf2f_101.npz`).

## Protocolo

- **Datos.** Primeras 26 000 filas de `SUSY.csv.gz` (UCI) → 5000/1000/5000 (2500/500/2500 por clase), `SPLIT_SEED = 2021`.
  El script regenera el split y lo compara con `../conciliacion/comparacion_uniforme/split_comun.npz` (assert).
  Escaladores ajustados solo con train (con train+val en M4).
- **Modelo.** 1 qubit, L = 5. Por capa: Rot(lep1_pt, lep2_pt, met) · Rot(m_tr_2, m_delta_r, lep1_eta) · RY(lep2_eta),
  con ángulo = w·x + b. 70 parámetros, más 4 de la lectura XYZ (p = sigmoide(a⟨X⟩ + b⟨Y⟩ + c⟨Z⟩ + d)).
  Simulador: `replica_dru/dru_core.py` (NumPy, gradiente adjoint), equivalente a PennyLane (`replica_dru/test_vs_pennylane.py`).
- **Entrenamiento.** Layerwise 1 época por capa (lr 0.05); fase 1: 12 épocas, lr 0.02; fase 2: 18 épocas, lr 0.01;
  Adam nuevo por fase, batch 128. Early stopping por AUC de validación (paciencia 7, Δ mín. 1e-4) que corta solo la
  fase en curso, con checkpoint en la mejor época; M4 se queda con la última. Inicialización 0.05·N(0,1); lectura en (0, 0, +1, 0).
- **Semillas.** 101, 202, 303, 404, 505. Ensemble = media de probabilidades.
- **Clásicos** (StandardScaler, tuning en validación, ajuste final con train): LogReg C ∈ {0.01…100}; SVM lineal C ∈ {0.01…10};
  RF 400 árboles, depth ∈ {6, 8, 12, None}, leaf ∈ {1, 10}; MLP-73 (8 ocultas), α × lr en grilla 3 × 3, 5 semillas.
- **Estadística.** IC 95 % por bootstrap estratificado (2000 réplicas); DeLong (Sun y Xu) sobre el ensemble de test,
  con Holm dentro de cada familia definida de antemano. También media ± sd por semilla y Δ pareado por semilla.
- **Entorno.** Python 3.12.3, NumPy 2.5.3, pandas 2.3.2, scikit-learn 1.7.2, SciPy 1.18.1 (WSL Ubuntu).
  4–7 s por corrida cuántica; 839 s en total con 5 procesos.

## Resultados (test, 5000 eventos)

| Modelo | AUC test (ensemble) | IC 95 % | AUC por semilla | Parám. |
|---|---|---|---|---|
| RF con tuning | **0.8559** | [0.8452, 0.8664] | | |
| M2 · Quantile + XYZ + BCE | **0.8549** | [0.8443, 0.8654] | 0.8483 ± 0.0034 | 74 |
| M1 · MinMax + XYZ + BCE | 0.8539 | [0.8435, 0.8644] | 0.8516 ± 0.0025 | 74 |
| M3 · fidelidad cuadrática (Z) | 0.8535 | [0.8432, 0.8637] | 0.8494 ± 0.0025 | 70 |
| M4 · cuadrática, train+val | 0.8533 | [0.8430, 0.8636] | 0.8485 ± 0.0033 | 70 |
| MLP-73 con tuning | 0.8532 | [0.8427, 0.8636] | 0.8463 ± 0.0108 | 73 |
| LogReg | 0.8382 | [0.8265, 0.8489] | | 8 |
| SVM lineal | 0.8375 | [0.8259, 0.8482] | | 8 |
| M3 · fidelidad lineal (Z) | 0.8360 | [0.8242, 0.8475] | 0.8321 ± 0.0017 | 70 |
| M4 · lineal, train+val | 0.8327 | [0.8207, 0.8442] | 0.8271 ± 0.0034 | 70 |

- M1 − M2 = −0.0009 (p = 0.65): con pesos entrenables el escalado no importa.
- Fidelidad cuadrática − lineal: +0.0175 en M3 y +0.0206 en M4 (p < 1e-4). Con la cuadrática, M3/M4 empatan con BCE.
- M2 − RF = −0.0010 (p = 1.0), M2 − MLP-73 = +0.0016 (p = 0.85): compatibles. M2 supera a LogReg y SVM (p < 1e-4).
- **Ningún cuántico supera al RF ni al MLP-73.**
- Pérdidas con lectura Z: el AUC sube de forma monótona al bajar α en la híbrida (0.8360 con α = 1 → 0.8535 con α = 0).
- Capas: 0.8055 (L = 1), 0.8446 (2), 0.8506 (3), 0.8549 (5), 0.8563 (8, p = 0.18 frente a L = 5).
- Init U[0,1) con h = π: el ensemble aguanta (0.8544) pero las semillas son inestables (sd 0.073). Hay que reportar ambas cosas.

Las tablas completas están en `resultados_test.csv` y `comparaciones_test.csv`.

## Diferencias con el script original

Este script vivía fuera del repo (`<proyecto>/conciliacion_codigos/corrida_final/`). Al traerlo solo cambió el bloque
`RUTAS` de `corrida_final.py` y la línea `PROJECT` de `correr_barren_plateaus.py`. Por eso el SHA-256 del script
registrado en el config (`1cff9a6b…`) es el del original. El resto de los SHA-256 siguen valiendo:
`dru_core.py` y `split_comun.npz` son los mismos archivos. El CSV registrado (`d4506f55…`) tenía fin de línea CRLF;
el que descarga `susy_data.py` en Linux o Colab tiene LF (`f10ddc24…`) y el contenido es idéntico.

Comprobado el 15/11/2026 en el repo: M2, semilla 101, reentrenado desde cero da exactamente los mismos puntajes de
validación y test que `scores/69e93f3adf2f_101.npz` (diferencia máxima 0.0).

## Cómo reproducir

```bash
# desde la raíz del repo
python -m pip install -r corrida_final/requirements_corrida_final.txt
python -c "import sys; sys.path.insert(0, 'replica_dru'); import susy_data; susy_data.load_susy(60000)"  # baja replica_dru/data/susy_head_60000.csv
```

Repetir la corrida completa en otra carpeta (la de aquí ya tiene resultados y no se sobrescribe):

```bash
mkdir corrida_final_rep && cp corrida_final/corrida_final.py corrida_final_rep/
cd corrida_final_rep && python corrida_final.py --n-jobs 4
```

Para una variante con PennyLane en Google Colab, ver `../colab/`.
