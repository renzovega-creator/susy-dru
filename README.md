# SUSY_DRU — clasificador cuántico con *data re-uploading* sobre SUSY

Código del proyecto *Clasificador cuántico variacional con data re-uploading para la detección de partículas
supersimétricas: comparación de funciones de pérdida y escalados para el pre-procesamiento del dataset SUSY*
(Concurso de fortalecimiento de actividades de investigación — Ciencias 2026, FCI-PUCP).
Asesor: Mg. Eduardo Massoni. Entrega final: 23/11/2026.

> **Estado (15/11/2026): corrida final hecha (7/11/2026) e incluida en `corrida_final/`.** Es la que alimenta los
> capítulos 3 y 4 del informe: diseño firmado M1–M4 y familias del objetivo 2, 5 semillas, test evaluado una sola vez.
> Los resultados de `replica_dru/resultados/` (4/08/2026) son preliminares y **no son la versión del informe**.
> El repositorio pasará a público al cierre del proyecto, como compromete la propuesta firmada.

## Estructura

```
susy-dru/
├── corrida_final/        corrida final del 7/11/2026: script, pre-registro (config JSON), resultados de test,
│                         comparaciones DeLong + Holm, 115 corridas cuánticas en scores/, figuras. Ver su README.
├── colab/                notebook de Google Colab: reproduce la corrida y trae el mismo circuito en PennyLane,
│                         verificado contra el motor NumPy, listo para escalar datos, épocas o semillas
├── conciliacion/
│   └── comparacion_uniforme/   los 4 códigos del equipo sobre el split firmado (26 000 → 5000/1000/5000):
│                               split común, scripts, scores y evaluación (DeLong + Holm). Ver su README.
└── replica_dru/          simulador NumPy de n qubits con gradiente adjoint, entrenamiento layerwise,
    │                     grid de pérdidas x escalados, baselines clásicos y diagnóstico de barren plateaus
    ├── README.md         documentación del paquete (API, arquitecturas, pérdidas, resultados preliminares)
    ├── susy_data.py      descarga de SUSY (UCI), split balanceado con test congelado, escalados
    ├── dru_core.py       circuito, lecturas, 6 pérdidas, gradiente adjoint, Adam
    ├── train.py          layerwise + ajuste global con early stopping por AUC de validación
    ├── run_grid.py       experimento principal (presets quick / medium / paper)
    ├── barren_plateaus.py
    ├── test_gradients.py     adjoint vs diferencias finitas + identidades analíticas
    ├── test_vs_pennylane.py  equivalencia con PennyLane (opcional)
    └── resultados/       CSV y figuras de las corridas quick y medium del 4/08/2026
```

## Cómo correrlo

Requiere Python ≥ 3.10 con `numpy`, `pandas`, `scikit-learn` y `matplotlib` (CPU basta).

```bash
cd replica_dru
python -m pip install -r requirements.txt
python test_gradients.py             # ~10 s, debe terminar en "TODO OK"
python run_grid.py --preset quick    # ~30 s, verifica el flujo completo
python run_grid.py --preset medium   # ~5 min en una laptop de 8 núcleos
```

`test_vs_pennylane.py` necesita además PennyLane (probado con 0.42.3 y 0.45.1).
Las rutas `C:\Users\...` que aparecen en `replica_dru/README.md` son del equipo donde se desarrolló; ignóralas.

La corrida final usa el motor de `replica_dru/dru_core.py` con la receta de entrenamiento de `corrida_final/corrida_final.py`
y las versiones de `corrida_final/requirements_corrida_final.txt`. Cómo reproducirla: `corrida_final/README.md`.

### En Google Colab (PennyLane)

Abrir `colab/SUSY_DRU_PennyLane_colab.ipynb` (*Archivo → Abrir cuaderno → GitHub*, con *Incluir repositorios privados*).
Antes, guardar en los *Secrets* de Colab un token de GitHub de solo lectura para este repo con el nombre `GH_TOKEN`.
El notebook explica cada paso; ver también `colab/README.md`.

## Datos

Se usa el dataset **SUSY** del UCI Machine Learning Repository (Whiteson, D., *SUSY*, 2014,
<https://doi.org/10.24432/C54606>, licencia CC BY 4.0), introducido en Baldi, Sadowski y Whiteson,
*Nature Communications* 5, 4308 (2014). No se versiona: `susy_data.load_susy()` descarga por streaming
las primeras filas y las guarda en `replica_dru/data/`, que está en `.gitignore`.

## Advertencias conocidas (verificadas el 13/10/2026)

- Las corridas con **BCE y lectura Z** (`A-bce`, `B-bce-mm`) no son reproducibles a tres decimales entre versiones
  de NumPy: la derivada de la BCE divide por p(1−p). El resto de `medium` se reproduce hasta 1e-6.
- `resultados/bootstrap_medium.csv` se generó con una versión anterior de `run_grid.py` (10 comparaciones; la actual
  define 12).
- En `replica_dru`, `"fidelity"` es la fidelidad **cuadrática** ⟨(1−F)²⟩; otros códigos del equipo llaman "fidelidad"
  a la lineal ⟨1−F⟩ (`"infidelity"` aquí).

## Equipo

Renzo Vega · César Huamán · Angelo Zamalloa · Sebastian Hernandez — Pontificia Universidad Católica del Perú.

## Licencia

Código bajo licencia MIT (ver `LICENSE`).
