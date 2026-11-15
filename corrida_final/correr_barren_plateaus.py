"""Corre replica_dru/barren_plateaus.py con --samples 500 SIN tocar replica_dru/resultados/.

Solo redirige las carpetas de salida (OUT, FIG) del modulo; el calculo es el del script original.
Uso (WSL): python correr_barren_plateaus.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # en el repo, corrida_final/ cuelga de la raíz (el original estaba un nivel más abajo)
sys.path.insert(0, str(PROJECT / "replica_dru"))

import barren_plateaus as bp  # noqa: E402

bp.OUT = HERE / "barren_plateaus_500"
bp.FIG = bp.OUT / "figuras"
sys.argv = ["barren_plateaus.py", "--samples", "500"]
bp.main()
