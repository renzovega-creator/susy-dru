# SUSY_DRU: clasificador cuántico con *data re-uploading* sobre SUSY

Código y resultados del proyecto *Clasificador cuántico variacional con data re-uploading para la detección de partículas
supersimétricas: comparación de funciones de pérdida y escalados para el pre-procesamiento del dataset SUSY*
(Concurso de fortalecimiento de actividades de investigación, Ciencias 2026, Facultad de Ciencias e Ingeniería de la PUCP).
Asesor: Mg. Eduardo Massoni.

> **Estado (23/11/2026): proyecto cerrado.** El informe final está en `informe/SUSY_DRU_informe_final.pdf`.
> `corrida_final/` es la corrida pre-registrada del 7/11/2026 que alimenta los capítulos 3 y 4 de ese informe, y `experimentos_adicionales/` contiene los dos experimentos exploratorios de su
> sección 4.8. Los resultados de `replica_dru/resultados/` (4/08/2026) y de `conciliacion/` (14/10/2026) son etapas
> anteriores y **no son los números del informe**.

## Qué se estudió

Un clasificador variacional de **un qubit** con *data re-uploading* (Pérez-Salinas et al., 2020) separa señal y fondo en
el conjunto SUSY (Baldi et al., 2014) con siete variables y una muestra balanceada de 11 000 eventos
(5000 de entrenamiento, 1000 de validación y 5000 de prueba, tomados de las primeras 26 000 filas).
Se variaron la función de pérdida y la lectura, el escalado (MinMax y Quantile), el rango angular, la inicialización,
la profundidad y el protocolo de entrenamiento, y se midió la varianza del gradiente con hasta seis qubits para detectar
*barren plateaus*.
Las 23 configuraciones y las familias de comparación se fijaron antes de entrenar; cada configuración se entrenó con
cinco semillas y el conjunto de prueba se puntuó una sola vez.

## Resultados principales (prueba, 5000 eventos)

| Modelo | AUC (*ensemble* de 5 semillas) | IC 95 % |
|---|---|---|
| *Random forest* (ajustado en validación) | 0.8559 | [0.8452, 0.8664] |
| M2: Quantile, lectura XYZ, entropía cruzada binaria | 0.8549 | [0.8443, 0.8654] |
| M3: Quantile, lectura Z, fidelidad cuadrática | 0.8535 | [0.8432, 0.8637] |
| MLP-73 (perceptrón multicapa de 73 parámetros) | 0.8532 | [0.8427, 0.8636] |
| Regresión logística | 0.8382 | [0.8265, 0.8489] |
| M3: Quantile, lectura Z, fidelidad lineal | 0.8360 | [0.8242, 0.8475] |

- Ningún modelo cuántico supera de forma significativa a los clásicos no lineales (DeLong con corrección de Holm);
  M2 supera a los modelos lineales en unos 0.017 (p < 1e-4).
- La pérdida lineal 1 − F queda unos 0.02 por debajo de la cuadrática (1 − F)² y de la entropía cruzada binaria,
  que quedan a menos de 0.001 entre sí.
- Con pesos de codificación entrenables, MinMax y Quantile no se distinguen. Pocas capas sí restan AUC
  (0.8055 con L = 1 frente a 0.8549 con L = 5).
- Con un qubit no hay *barren plateau*; con varios qubits la varianza del gradiente decae, aproximadamente a la mitad
  por qubit con L = 4.
- Un qubit se simula clásicamente y la simulación no tiene ruido, así que este trabajo no puede mostrar una ventaja
  cuántica: su aporte es metodológico.

Tablas completas: `corrida_final/resultados_test.csv` y `corrida_final/comparaciones_test.csv`.

## Estructura

```
susy-dru/
├── informe/                  informe final en PDF (60 páginas)
├── corrida_final/            corrida pre-registrada del 7/11/2026 (caps. 3 y 4 del informe): script, configuración
│                             congelada, 115 corridas cuánticas en scores/, resultados, comparaciones y los scripts
│                             que generan las figuras y tablas del informe. Ver su README.
├── experimentos_adicionales/ sec. 4.8 del informe (exploratoria): ocho variables primarias y clasificadores de dos y
│                             tres qubits con costo local y global, en notebooks de Colab con PennyLane. Ver su README.
├── colab/                    notebook de Colab que reproduce la corrida final y trae el mismo circuito en PennyLane,
│                             verificado contra el motor NumPy.
├── replica_dru/              motor NumPy: simulador de n qubits con gradiente adjoint, pérdidas, entrenamiento capa
│                             por capa, diagnóstico de barren plateaus y pruebas (adjoint frente a diferencias finitas
│                             y frente a PennyLane). Sus resultados/ son preliminares (4/08/2026).
└── conciliacion/             comparación de los cuatro códigos del equipo sobre la partición firmada (14/10/2026),
                              de la que sale split_comun.npz, que la corrida final usa para verificar la partición.
```

## Cómo reproducirlo

Requiere Python ≥ 3.10; la corrida final se hizo con Python 3.12.3 y las versiones de
`corrida_final/requirements_corrida_final.txt`. Basta una CPU.

```bash
python -m pip install -r corrida_final/requirements_corrida_final.txt
python replica_dru/test_gradients.py     # gradiente adjoint frente a diferencias finitas, ~10 s
```

La corrida final completa tarda unos 14 minutos con cinco procesos; cómo repetirla sin sobrescribir los resultados
guardados está en `corrida_final/README.md`.
Los notebooks de `colab/` y `experimentos_adicionales/` se abren en Google Colab
(*Archivo → Abrir cuaderno → GitHub* y buscar `renzovega-creator/susy-dru`).

## Datos

Se usa el conjunto **SUSY** del UCI Machine Learning Repository (Whiteson, D., *SUSY*, 2014,
<https://doi.org/10.24432/C54606>, licencia CC BY 4.0), introducido en Baldi, Sadowski y Whiteson,
*Nature Communications* 5, 4308 (2014), <https://doi.org/10.1038/ncomms5308>.
No se versiona: `replica_dru/susy_data.py` descarga por *streaming* las primeras filas y las guarda en
`replica_dru/data/`, que está en `.gitignore`.

## Advertencias

- Los nombres de las pérdidas en el código no son los del informe. En `replica_dru`, `"fidelity"` es la fidelidad
  **cuadrática** ⟨(1 − F)²⟩ e `"infidelity"` es la **lineal** ⟨1 − F⟩; el informe las llama pérdida cuadrática y
  pérdida lineal.
- Los experimentos de `experimentos_adicionales/` no estaban pre-registrados y usan el conjunto de prueba ya puntuado
  en la corrida final: son exploratorios.
- Las rutas `C:\Users\...` de `replica_dru/README.md` son del equipo donde se desarrolló; ignóralas.

## Equipo

Renzo Vega, César Huamán, Angelo Zamalloa y Sebastian Hernandez (Pontificia Universidad Católica del Perú).

## Licencia

Código bajo licencia MIT (ver `LICENSE`).
