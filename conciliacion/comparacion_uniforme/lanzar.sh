#!/usr/bin/env bash
# Comparación uniforme: dos tandas en paralelo, prioridad baja, un hilo por proceso.
# Uso (WSL):
#   SUSY_NOTEBOOKS=/ruta/notebooks bash lanzar.sh        # todo: replica_dru + los 3 notebooks
#   bash lanzar.sh                                       # sin SUSY_NOTEBOOKS: solo replica_dru
# Logs en logs/, scores en scores/ y scores_sebastian/. Para repetir una corrida de Sebastián desde cero,
# borrar antes scores_sebastian/<modo>/ (su notebook reanuda las corridas que encuentra hechas).
# PY: intérprete con numpy, pandas, scikit-learn, scipy, matplotlib, pennylane y torch.
set -u
cd "$(dirname "$0")"
PY=${PY:-~/qml_workshop/.venv/bin/python}
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs scores
run() { local name=$1; shift; local t0=$(date +%s)
  ( "$@" ) > "logs/$name.log" 2>&1; local rc=$?
  echo "$(date +%T) $name rc=$rc $(( $(date +%s) - t0 ))s" | tee -a logs/estado.txt; }

$PY split_comun.py
if [ -n "${SUSY_NOTEBOOKS:-}" ]; then
  $PY build_subs.py || exit 1
  mkdir -p wd_four_101 wd_four_202 wd_four_303 wd_angello wd_seb
fi

echo "$(date +%T) TANDA 1" | tee -a logs/estado.txt
if [ -n "${SUSY_NOTEBOOKS:-}" ]; then
  for s in 101 202 303; do
    run four_$s bash -c "cd wd_four_$s && UNIF_SEED=$s nice -n 19 $PY -u ../run_four.py" &
  done
fi
run replica nice -n 19 $PY -u run_replica.py &
wait

if [ -n "${SUSY_NOTEBOOKS:-}" ]; then
  echo "$(date +%T) TANDA 2" | tee -a logs/estado.txt
  run angello bash -c "cd wd_angello && nice -n 19 $PY -u ../run_angello.py" &
  run sebastian_receta bash -c "cd wd_seb && UNIF_VARIANTS=E0,E1,E2 UNIF_NFEAT=6 UNIF_NORM=none UNIF_RESDIR=../scores_sebastian/receta nice -n 19 $PY -u ../run_sebastian.py" &
  run sebastian_explora bash -c "cd wd_seb && UNIF_VARIANTS=E0_7,E2_7 UNIF_NFEAT=7 UNIF_NORM=quantile UNIF_RESDIR=../scores_sebastian/explora nice -n 19 $PY -u ../run_sebastian.py" &
  wait
fi
$PY evaluar.py > evaluar.out 2>&1
echo "$(date +%T) FIN" | tee -a logs/estado.txt
