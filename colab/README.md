# Colab · variante con PennyLane

`SUSY_DRU_PennyLane_colab.ipynb` es el punto de partida para rehacer la corrida final con PennyLane y escalarla
(más eventos, épocas o semillas) en Google Colab.

1. Clona este repo privado con un token guardado en los *Secrets* de Colab (`GH_TOKEN`, solo lectura) e instala PennyLane.
2. Baja SUSY desde UCI y verifica que se regenera el split firmado (`conciliacion/comparacion_uniforme/split_comun.npz`).
3. Reentrena M2 (semilla 101) con el motor NumPy de `corrida_final/` y lo compara con los puntajes guardados.
4. Define el mismo circuito en PennyLane (`ModeloPL`, misma interfaz que `dru_core.DRUModel`) y verifica probabilidades,
   pérdidas y gradientes para M2, M3-lin, M3-cua, la híbrida α = 0.5 y BCE con lectura Z.
5. Entrena con PennyLane con la misma receta que la corrida final. `split_ampliado` deja fijo el test oficial y saca
   train/val más grandes del resto de las filas, para que los AUC sigan siendo comparables.
6. Evalúa contra `corrida_final/resultados_test.csv` y guarda los puntajes (en `colab/salidas/` o en Drive).

Comprobado el 15/11/2026 (WSL, PennyLane 0.42.3): el entrenamiento completo de M2 con PennyLane reproduce la corrida
final (misma mejor época, AUC de test 0.8506, diferencia máxima de 8e-9 en las probabilidades) en 85 s, contra 11 s
del motor NumPy.

El test oficial ya se usó una vez para la corrida pre-registrada: lo que se corra aquí es exploratorio y los
hiperparámetros se eligen con validación.

Para abrirlo: en Colab, *Archivo → Abrir cuaderno → GitHub*, marcar *Incluir repositorios privados* y buscar
`renzovega-creator/susy-dru`. También se puede subir el `.ipynb` a mano.
