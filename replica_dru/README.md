# `replica_dru` — Clasificador Data Re-Uploading para SUSY

Réplica unificada de las implementaciones de la literatura del proyecto SUSY_DRU
(Pérez-Salinas 2020, Muten 2021, Skolik 2021, Terashi 2021, Baldi 2014, Cerezo 2021),
sobre un simulador de $n$ qubits en NumPy con gradiente **adjoint exacto**,
verificado bit a bit contra PennyLane 0.45.

Contexto y planificación: `../HOJA_DE_RUTA_SUSY_DRU.md`.

---

## Instalación

Solo hacen falta `numpy`, `pandas`, `scikit-learn` y `matplotlib` — que ya vienen
con Anaconda. PennyLane **solo** se necesita para el test de equivalencia.

```bash
python -m pip install numpy pandas scikit-learn matplotlib
```

En esta máquina hay además un entorno con PennyLane 0.45 en
`C:\Users\renzo\.venv_qml_susy`.

## Uso

```bash
python test_gradients.py                    # verificación del núcleo (~30 s)
python run_grid.py --preset quick           # flujo completo, ~2 min
python run_grid.py --preset medium          # resultados presentables, ~25 min
python run_grid.py --preset paper           # 5 semillas, presupuesto del informe
python barren_plateaus.py --samples 200     # diagnóstico de gradientes
```

Test de equivalencia con PennyLane:

```bash
C:/Users/renzo/.venv_qml_susy/Scripts/python.exe test_vs_pennylane.py
```

Salidas en `resultados/` (CSV) y `resultados/figuras/` (PNG).
El dataset se descarga una vez de UCI y queda cacheado en `data/`.

### En Google Colab

El paquete no necesita instalación: Colab ya trae numpy, pandas, scikit-learn y
matplotlib. Una vez que `replica_dru/` esté en el repositorio público del
proyecto (es un entregable comprometido para noviembre):

```python
!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>/replica_dru
!python run_grid.py --preset medium
```

Y para trabajar en celdas en lugar de por CLI:

```python
import sys; sys.path.insert(0, "/content/<repo>/replica_dru")
from susy_data import load_susy, make_split, FEATURES_7
from train import train_one, TrainConfig, evaluate
```

---

## Módulos

| Archivo | Qué hace |
|---|---|
| `susy_data.py` | Descarga por streaming de las primeras *N* filas de `SUSY.csv.gz`, verificación SHA-256, split balanceado con **test congelado**, escalados MinMax / Quantile / Standard / identidad, padding a múltiplo de 3 |
| `dru_core.py` | Simulador `(B, 2ⁿ)`, gradiente adjoint, 3 arquitecturas, 3 lecturas, 6 pérdidas, `Adam`, máscaras layerwise |
| `train.py` | Crecimiento layerwise + ajuste global con early stopping por AUC de validación; registra `‖∇L‖` y `Var[∇L]` por época |
| `run_grid.py` | Grid de pérdidas × escalados × arquitecturas, baselines clásicos, bootstrap pareado, figuras |
| `barren_plateaus.py` | `Var[∂L/∂θ]` en función de la profundidad y del número de qubits, coste local vs. global |
| `test_gradients.py` | Adjoint vs. diferencias finitas en 26 configuraciones + identidades analíticas |
| `test_vs_pennylane.py` | Equivalencia con `circuit_bloch` del Entregable 1 y con `qcircuit_susy` del SEED |

## API

```python
from susy_data import load_susy, make_split, FEATURES_7
from train import train_one, TrainConfig, evaluate

df = load_susy(60_000)
sp = make_split(df, features=FEATURES_7, scaler="quantile",
                train_per_class=2000, val_per_class=1000, test_per_class=4000)

r = train_one("muten", sp, loss="bce", n_layers=5, readout="z", seed=101,
              cfg=TrainConfig(max_epochs=15))
scores, test_auc = evaluate(r.config, r.params, sp)
```

### Arquitecturas (`arch`)

| Valor | Origen | Descripción |
|---|---|---|
| `"perez_salinas"` | Quantum **4**, 226 (2020) | Por capa: bloques `Rot(x₃ₖ, x₃ₖ₊₁, x₃ₖ₊₂)` **no entrenables** + una `Rot(θ)` entrenable. Requiere $d$ múltiplo de 3 (`pad_to_multiple_of=3`) |
| `"perez_salinas_w"` | Variante *with weights* del mismo paper | Igual, pero con pesos entrenables en el encoding |
| `"muten"` | IEEE QCE (2021) / QHack 2021 | Encoding y ansatz fusionados: ángulo $= w_{\ell j}x_j + b_{\ell j}$. Es el del `Entregable1_propuesta(1).ipynb` |

### Lecturas (`readout`)

| Valor | $p(\text{señal})$ | Parámetros extra |
|---|---|---|
| `"z"` | $(1 - \langle Z_0\rangle)/2$ | 0 |
| `"parity"` | $(1 - \langle Z_0 Z_1 \cdots Z_{n-1}\rangle)/2$ — coste **global** | 0 |
| `"xyz"` | $\sigma(a\langle X\rangle + b\langle Y\rangle + c\langle Z\rangle + d)$ | 4 (clásicos) |

### Pérdidas (`loss`)

| Valor | Expresión | Nota |
|---|---|---|
| `"fidelity"` | $\langle (1-F_y)^2\rangle$ | Pérez-Salinas |
| `"infidelity"` | $\langle 1-F_y\rangle$ | lineal |
| `"cerezo"` | $\langle (1 - s_y\langle Z_0\rangle)/2\rangle$ | coste local de Cerezo |
| `"bce"` | entropía cruzada binaria | opera sobre la lectura |
| `"mse"` | $\langle (p-y)^2\rangle$ | = *weighted fidelity* de Muten |
| `"hybrid"` | $\alpha\,L_{\text{lineal}} + (1-\alpha)L_{\text{cuadrática}}$ | ablación en $\alpha$ |

---

## Dos identidades que hay que tener presentes

Ambas verificadas numéricamente en `test_gradients.py` y visibles en los
resultados (las filas correspondientes salen con el **mismo** AUC):

1. **A $n = 1$: `cerezo` ≡ `infidelity`.**
   $L_{\text{Cerezo}} = (1 - s_y\langle Z\rangle)/2 = 1 - F_y$, con $s_y = 1-2y$.
   El proyector "global" $|0\rangle\langle 0|$ de un solo qubit ya es local, así que
   la separación global/local de Cerezo et al. **no existe** a un qubit.
2. **A $n = 1$ con lectura Z: `fidelity` ≡ `mse`.**
   Como $p = (1-\langle Z\rangle)/2$ y $1 - F_y = |p - y|$, se tiene
   $(1-F_y)^2 = (p-y)^2$ exactamente.

**Consecuencia:** a 1 qubit con lectura Z no hay cuatro objetivos distintos,
hay **tres**: lineal, cuadrático y logarítmico. Conviene decirlo en el informe
antes de que lo diga el jurado.

---

## Resultados obtenidos (preset `medium`: 2000 eventos/clase de entrenamiento, 3 semillas)

AUC sobre el mismo test fijo de 8000 eventos balanceados:

| Modelo | Pérdida | Escalado | Lectura | AUC (ensemble) |
|---|---|---|---|---|
| MLP-73 | — | standard | — | 0.8524 |
| RandomForest | — | standard | — | 0.8515 |
| **A-fid / A-mse** | cuadrática | quantile | z | **0.8494** |
| C-xyz-bce | BCE | quantile | xyz | 0.8490 |
| A-bce | BCE | quantile | z | 0.8457 |
| A-hybrid ($\alpha=0.5$) | híbrida | quantile | z | 0.8436 |
| LogReg | — | standard | — | 0.8382 |
| **A-infid / A-cerezo** | lineal | quantile | z | **0.8308** |

Diferencias pareadas con bootstrap estratificado (IC 95 %):

- cuadrática − lineal: **+0.0186** [+0.0147, +0.0224] → concluyente.
- BCE − lineal: **+0.0148** [+0.0113, +0.0185] → concluyente.
- BCE − cuadrática: −0.0037 [−0.0066, −0.0009] → concluyente pero pequeña.

**Lectura:** a 1 qubit, la pérdida "local de Cerezo" es la **peor** de las tres,
no la mejor. Es coherente con el §3 de la hoja de ruta: sin barren plateau
exponencial que mitigar, el coste lineal solo aporta un gradiente más plano
cerca de la clasificación correcta.

### El hallazgo sobre escalados

El título del proyecto financiado incluye "escalados para el pre-procesamiento",
y aquí aparece el resultado más limpio:

AUC de test (ensemble de 3 semillas), pérdida de fidelidad, 5 capas:

| Ansatz | Pesos de encoding | MinMax | Quantile |
|---|---|---|---|
| `perez_salinas` | fijos | 0.656 | **0.522 — azar** |
| `perez_salinas_w` | entrenables (sin sesgo) | 0.812 | **0.523 — azar** |
| `muten` | entrenables (con sesgo) | 0.842 | 0.849 |

La dispersión entre semillas también lo delata: `perez_salinas_w` con MinMax tiene
$\sigma = 0.043$ entre semillas, frente a $\sigma \le 0.01$ del ansatz de Muten.

Con encoding no entrenable, el escalado **decide si el modelo aprende o no**:
`QuantileTransformer` reparte la masa uniformemente por todo $[-\pi,\pi]$, y con
15 rotaciones de datos fijas por pasada el estado queda aleatorizado. `MinMax`
concentra la mayoría de los eventos en ángulos pequeños y deja al circuito en un
régimen entrenable. El ansatz de Muten es casi insensible porque sus pesos
$w_{\ell j}$ **absorben** el escalado.

Esto matiza la conclusión del Entregable 1 ("Quantile mejoró a MinMax en 0.011"):
es cierta para el ansatz con pesos entrenables, y **se invierte** cuando el
encoding es fijo. Conecta directamente con Schuld, Sweke & Meyer (2021): el
escalado fija el espectro de frecuencias accesible.

---

## Diagnóstico de gradientes (`barren_plateaus.py`, 150 inicializaciones)

**A. Profundidad a $n=1$.** $\mathrm{Var}[\partial L/\partial w_k]$ es **plana** entre
$L=1$ y $L=24$: no hay plateau inducido por profundidad a un qubit. Lo que sí
aparece es una separación de dos órdenes de magnitud entre objetivos:

| Pérdida | Var$[\partial L]$ | $\mathbb{E}\lvert\partial L\rvert$ |
|---|---|---|
| BCE | $\sim 2.2\times10^{-1}$ | $\sim 3.4\times10^{-1}$ |
| fidelidad (cuadrática) | $\sim 5.0\times10^{-3}$ | $\sim 5.8\times10^{-2}$ |
| Cerezo (lineal) | $\sim 4.4\times10^{-3}$ | $\sim 5.3\times10^{-2}$ |

BCE tiene gradientes ~50× mayores, pero en el entrenamiento **no** converge antes
(necesita ~6 épocas para alcanzar a las demás). Es un resultado de
condicionamiento, no de entrenabilidad.

**B. Número de qubits a $L=4$.** Coste local $\langle Z_0\rangle$ y coste global
$\langle Z_0\cdots Z_{n-1}\rangle$ decaen **igual**, de $4.4\times10^{-3}$ ($n=1$) a
$1.1\times10^{-4}$ ($n=6$): un factor ~2 por qubit, más suave que la referencia
$4^{-n}$, y **sin separación local/global**.

No es un fallo: la garantía polinómica de Cerezo et al. para el coste local exige
profundidad $O(\log n)$, y a $L=4$ con anillo de CNOT el cono de luz de $Z_0$ ya
cubre los seis qubits. **El experimento correcto es el barrido conjunto $(n, L)$.**

**C. Barrido $(n, L)$ — aquí sí aparece la separación.** Cociente
$\mathrm{Var}[\partial L]_{\text{local}} / \mathrm{Var}[\partial L]_{\text{global}}$,
promediado sobre todos los parámetros:

| $L$ | $n=2$ | $n=4$ | $n=6$ |
|---|---|---|---|
| 2 | 0.98 | **1.78** | **2.65** |
| 4 | 1.00 | 1.08 | 1.11 |
| 8 | 1.03 | 1.05 | 1.04 |

A profundidad **$L=2$** la ventaja del coste local **crece con el número de
qubits** — que es exactamente la predicción de Cerezo et al. — y a $L \ge 4$
desaparece, tragada por el cono de luz. Es un resultado preliminar (150
inicializaciones, hasta $n=6$), pero indica dónde hay que mirar: **el eje que
importa no es la pérdida a $n$ fijo, es la pareja $(n, L)$.**

$L=1$ queda fuera de la lectura: con una sola capa los dos CNOT del anillo
convierten $Z_0$ en $Z_1$ en la imagen de Heisenberg y la sonda se anula
estructuralmente.

---

## Verificación

`test_gradients.py` — 26 configuraciones (3 arquitecturas × 6 pérdidas ×
{1, 3} qubits × {z, parity, xyz}), adjoint vs. diferencias finitas:
error relativo entre $10^{-10}$ y $10^{-8}$.

`test_vs_pennylane.py` — contra PennyLane 0.45.1:

| Comparación | Error máximo |
|---|---|
| $\langle X\rangle, \langle Y\rangle, \langle Z\rangle$ del `circuit_bloch` del Entregable 1 | $1.3\times10^{-15}$ |
| pérdida y gradiente del `qcircuit_susy` (`qml.Hermitian`) del SEED | $5.8\times10^{-16}$ |
| gradiente completo de BCE + lectura XYZ contra `qml.grad` | $6.0\times10^{-10}$ |

---

## Advertencia de inicialización

La restricción 9 del `SEED_contexto_INF659_QML.md` dice "inicializar siempre
uniforme en $[0, 2\pi]$". Es correcta para `perez_salinas`, donde **todos** los
parámetros son ángulos. Es un error para `perez_salinas_w` y `muten`, donde los
$w$ **multiplican** a $x \in [-\pi, \pi]$: un $w \sim U(0, 2\pi)$ genera ángulos de
hasta ~20 rad y el entrenamiento colapsa a AUC ≈ 0.50. `init_params` lo maneja
(pesos cerca de 1, sesgos en $[0,2\pi]$), pero conviene saberlo si se porta el
código a PennyLane.
