# Comparación uniforme de los códigos del proyecto (14/10/2026)

Los cuatro códigos que tenía el equipo (`replica_dru`, el Entregable 1, el notebook de encoding de
Sebastián y `final_quantum_susy_1q_four_models.ipynb`) se corrieron **sobre el mismo split**, cada uno
con su propia receta de modelo y entrenamiento, y se evaluaron con las mismas reglas. Es la base del
pipeline que se congelará para el informe.

**Split común = el de la propuesta firmada:** primeras 26 000 filas de SUSY (UCI) → 5000 / 1000 / 5000
eventos balanceados (2500 / 500 / 2500 por clase), `SPLIT_SEED = 2021`.

## Resultado principal

AUC de test del ensemble de 3 semillas (101, 202, 303), con IC 95 % bootstrap. Tabla completa en
`resultados_uniforme.csv` y `evaluar.out`; comparaciones en `comparaciones_uniforme.csv`.

| Modelo | AUC test | IC 95 % |
|---|---|---|
| four_models M2 (Quantile + XYZ + BCE) | 0.8526 | [0.8426, 0.8628] |
| Entregable 1 Q3 (Quantile + Z + BCE) | 0.8519 | [0.8416, 0.8620] |
| `replica_dru` C-xyz-bce | 0.8477 | [0.8373, 0.8582] |
| Sebastián E0 (6 variables crudas) | 0.8458 | [0.8353, 0.8565] |
| Random forest con tuning (clásico) | 0.8559 | [0.8453, 0.8659] |

- four_models y el Entregable 1 empatan (DeLong + Holm, p = 0.64) y superan a los otros dos (p ≤ 0.008).
- Ningún modelo cuántico supera al random forest; los dos mejores son compatibles con él (p = 0.06 y 0.10).

## Archivos

| Archivo | Qué es |
|---|---|
| `split_comun.py` → `split_comun.npz`, `split_comun.out` | genera los índices del split común y su SHA-256 |
| `run_replica.py` | corre la receta de `replica_dru` (preset `paper`) sobre el split común |
| `build_subs.py`, `nb2py.py` | convierten los notebooks del equipo en scripts que usan el split común y guardan sus scores |
| `lanzar.sh` | lanza todo en dos tandas en paralelo, con prioridad baja |
| `evaluar.py`, `estadistica.py` | evaluación común: ensemble, IC bootstrap, DeLong (Sun y Xu) y Holm |
| `scores/`, `scores_sebastian/` | scores de validación y test de cada método y semilla (lo que lee `evaluar.py`) |
| `logs/estado.txt` | tiempos de cada proceso en la corrida del 14/10 (36 min de reloj, 8 núcleos, solo CPU) |

## Cómo reproducir

```bash
# Solo la evaluación (segundos): recalcula tablas y comparaciones desde los scores guardados
python evaluar.py

# Solo replica_dru (~5 min):
bash lanzar.sh

# Todo (~40 min): requiere los 3 notebooks del equipo en una carpeta
SUSY_NOTEBOOKS=/ruta/a/los/notebooks bash lanzar.sh
```

Requisitos: numpy, pandas, scikit-learn, scipy, matplotlib; además PennyLane y torch para los notebooks.
Con PennyLane 0.42 + NumPy 2.5, `build_subs.py` aplica un parche de compatibilidad (invierte el orden de
algunos productos en la BCE y la fidelidad, sin cambiar las matemáticas); los notebooks se escribieron
para PennyLane 0.45.0 + NumPy 2.0.2.

## Qué no está aquí y por qué

- **Los notebooks de Angello y Sebastián, y los `run_*.py` generados a partir de ellos**, no están en el
  repositorio: entran solo con el visto bueno de sus autores. `.gitignore` excluye los scripts generados.
- `split_comun.py` incluye la función de muestreo de `final_quantum_susy_1q_four_models.ipynb` (celda 19,
  ~15 líneas), copiada literalmente para que el split sea idéntico al del notebook.
- `estadistica.py` es una implementación propia. Se verificó que da los mismos resultados que la del
  notebook de Sebastián (celda 13): diferencia máxima de 3e-13 en 276 pares de DeLong, Holm y bootstrap
  idénticos.
- El análisis completo (conciliación, verificación y recomendación) está en la carpeta
  `05_CONCILIACION/` del Drive del proyecto.
