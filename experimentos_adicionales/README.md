# Experimentos adicionales (exploratorios)

Dos experimentos que complementan la corrida final y forman la sección 4.8 del informe.
**No estaban pre-registrados** y usan el conjunto de prueba que ya se había puntuado en la corrida final, así que son
exploratorios y no cambian las conclusiones del diseño firmado.

Parten de los mismos datos, la misma partición (5000/1000/5000) y el mismo escalado (Quantile, [−π, π]) que la corrida
final. Se corrieron en Google Colab con PennyLane 0.45.1 (gradiente por *backpropagation*). Antes de entrenar, cada
notebook comprueba que su simulador y su bucle de entrenamiento reproducen la corrida final: con la receta de M3 con
pérdida lineal y la semilla 101, las puntuaciones de prueba coinciden con las guardadas hasta 1e-11.
Los AUC son del *ensemble* de las semillas, con intervalos del 95 % por *bootstrap* y valores p de DeLong con la
corrección de Holm dentro de cada familia.

## `8_variables/`: siete variables frente a las ocho primarias

Cambia el conjunto de variables y deja todo lo demás como en la corrida final. Las ocho primarias son las de bajo nivel
de SUSY (`lep1_pt`, `lep2_pt`, `met`, `lep1_eta`, `lep2_eta`, `lep1_phi`, `lep2_phi`, `met_phi`): salen `m_tr_2` y
`m_delta_r` y entran los tres ángulos azimutales.

- Tres pérdidas: entropía cruzada binaria con lectura XYZ (como M2), y pérdidas lineal y cuadrática con lectura Z (como M3).
- Búsqueda en validación: η ∈ {0.005, 0.01, 0.02} y *batch* de 64 o 128, con tres semillas (101, 202, 303): 108 corridas.
  El ajuste de cada modelo se eligió con validación (`tablas/seleccion.json`) antes de puntuar el conjunto de prueba.
- Resultado: las ocho primarias restan entre 0.0074 y 0.0085 de AUC con las tres pérdidas (p_Holm ≤ 0.0004), aunque el
  modelo tenga diez parámetros más.

| Archivo | Qué es |
|---|---|
| `SUSY_DRU_7v8_PennyLane.ipynb` | Notebook completo, con las salidas de la corrida |
| `tablas/validacion_por_corrida.csv`, `tablas/validacion_por_ajuste.csv` | AUC de validación de las 108 corridas y por ajuste |
| `tablas/seleccion.json` | Ajuste elegido con validación para cada grupo |
| `tablas/test_resumen.csv`, `tablas/test_por_semilla.csv` | AUC de prueba del *ensemble* y por semilla |
| `tablas/test_diferencia_8_menos_7.csv` | Diferencia 8 − 7 variables con DeLong y Holm |
| `tablas/equivalencias_corrida_final.csv` | Comprobación contra la corrida final |
| `figuras/` | Búsqueda en validación, curvas de validación, ROC, precisión-*recall*, calibración y distribuciones |

## `2_y_3_qubits/`: costo local y global con varios qubits

Con un qubit, el costo local y el global coinciden; con dos ya son distintos. Este experimento entrena clasificadores
de dos y tres qubits para ver si la elección del costo cambia el AUC de prueba.

- En cada capa, cada qubit carga las siete variables con sus propios pesos y sesgos, seguido de una capa de CNOT en
  anillo. Costo local: p = (1 − ⟨Z₀⟩)/2; costo global: p = (1 − ⟨Z₀Z₁⋯Z₍ₙ₋₁₎⟩)/2.
- Receta de la corrida final, sin búsqueda de hiperparámetros, con las cinco semillas de la corrida final.
  Dos qubits con L = 5 (140 parámetros) y las pérdidas lineal, cuadrática y entropía cruzada binaria; tres qubits con la
  pérdida lineal, L = 2 (84 parámetros) y L = 5 (210 parámetros). Diez modelos y 50 corridas.
- La predicción se escribió antes de correr. Las referencias de un qubit son las puntuaciones guardadas de la corrida
  final, sin reentrenar.
- Resultado: con dos qubits el AUC sube entre 0.003 y 0.004 con la pérdida cuadrática y la entropía cruzada binaria, y
  no cambia de forma significativa con la lineal. Con tres qubits y L = 2, el costo global queda 0.0222 por debajo del
  local (p_Holm < 1e-4); con L = 5 la diferencia baja a 0.0039.

| Archivo | Qué es |
|---|---|
| `SUSY_DRU_2y3_qubits_PennyLane.ipynb` | Notebook completo, con las salidas de la corrida |
| `tablas/resultados_2q_3q.csv`, `tablas/metricas_prueba_ensemble.csv` | AUC de prueba y métricas del *ensemble* |
| `tablas/comparaciones_2q_3q.csv` | Comparaciones por familia con DeLong y Holm |
| `tablas/diagnostico_gradiente.csv`, `tablas/diagnostico_gradiente_cocientes.csv` | Varianza del gradiente con la inicialización del entrenamiento (200 inicializaciones) |
| `tablas/equivalencias_corrida_final.csv` | Comprobación contra la corrida final |
| `figuras/` | AUC de prueba, comparaciones, curvas de validación y diagnóstico del gradiente |

## Cómo correrlos

Abrir el notebook en Colab (*Archivo → Abrir cuaderno → GitHub*, `renzovega-creator/susy-dru`), entorno de CPU.
El paso 1 clona este repositorio, instala PennyLane 0.45.1 y monta Google Drive, donde se guarda cada corrida en cuanto
termina; si Colab se desconecta, al volver a ejecutar se saltan las corridas ya guardadas.
`MODO = "rapido"` verifica el flujo en pocos minutos (sus números de prueba no significan nada);
`MODO = "completo"` es el experimento (unas 3 h el de ocho variables y unas 2 h el de dos y tres qubits).
Las corridas individuales quedan en Drive; en este repositorio están solo las tablas y figuras finales.
