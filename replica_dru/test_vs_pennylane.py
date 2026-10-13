"""
Equivalencia con PennyLane.

Comprueba que el simulador NumPy de `dru_core.py` reproduce exactamente el
circuito de PennyLane que usa el grupo (el del `Entregable1_propuesta(1).ipynb`
y el del SEED del proyecto), tanto en valores esperados como en gradientes.

Este archivo es la garantia de que acelerar el grid con NumPy no cambia la
fisica: si este test pasa, cualquier resultado del paquete es defendible ante
el tribunal como "PennyLane, vectorizado".

Requiere PennyLane. Ejecutar:
    <python-con-pennylane> test_vs_pennylane.py
"""

from __future__ import annotations

import numpy as np

import pennylane as qml
from pennylane import numpy as pnp

from dru_core import DRUModel, build_circuit

N_LAYERS = 5
RNG = np.random.default_rng(2026)


# ---------------------------------------------------------------------------
# Referencia 1: circuito del Entregable 1 (arquitectura "muten")
# ---------------------------------------------------------------------------
dev = qml.device("default.qubit", wires=1)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def circuit_bloch(inputs, theta):
    """Copia literal del `circuit_bloch` del notebook del Entregable 1."""
    n_features = theta.shape[1]
    for layer_index in range(N_LAYERS):
        angles = theta[layer_index, :, 0] * inputs + theta[layer_index, :, 1]
        qml.Rot(angles[0], angles[1], angles[2], wires=0)
        qml.Rot(angles[3], angles[4], angles[5], wires=0)
        if n_features == 7:
            qml.RY(angles[6], wires=0)
    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))


def test_muten_expvals():
    d = 7
    circ = build_circuit("muten", d, N_LAYERS, 1, "xyz")
    model = DRUModel(circ, loss="bce")
    p = model.init_params(seed=17, scale=0.9)

    # Reordenar los parametros del vector plano al tensor (L, d, 2) del notebook.
    theta = np.zeros((N_LAYERS, d, 2))
    for k, g in enumerate(circ.gates):
        layer, feat = circ.layer_of_gate[k], g.feat
        theta[layer, feat, 0] = p[g.w_idx]
        theta[layer, feat, 1] = p[g.b_idx]

    x = RNG.uniform(-np.pi, np.pi, size=d)
    ref = np.array([float(v) for v in circuit_bloch(pnp.array(x, requires_grad=False),
                                                    pnp.array(theta, requires_grad=False))])

    from dru_core import _forward_states, _observable_expval
    state, _, _ = _forward_states(circ, p, x.reshape(1, -1))
    mine = np.array([_observable_expval(state, o)[0][0] for o in ("X", "Y", "Z")])

    err = np.max(np.abs(ref - mine))
    print(f"  [{'OK ' if err < 1e-12 else 'FALLA'}] <X>,<Y>,<Z> muten   err_max={err:.2e}")
    print(f"        PennyLane = {np.round(ref, 10)}")
    print(f"        NumPy     = {np.round(mine, 10)}")
    return err < 1e-12


# ---------------------------------------------------------------------------
# Referencia 2: circuito del SEED del proyecto (arquitectura "perez_salinas")
# ---------------------------------------------------------------------------
def _make_ps_qnode(n_blocks, n_layers):
    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def qcircuit_susy(params, x, y_dm):
        """Copia literal de `qcircuit_susy` del SEED_contexto_INF659_QML.md."""
        for p in params:
            for j in range(n_blocks):
                s = j * 3
                qml.Rot(x[s], x[s + 1], x[s + 2], wires=0)
            qml.Rot(p[0], p[1], p[2], wires=0)
        return qml.expval(qml.Hermitian(y_dm, wires=[0]))

    return qcircuit_susy


def test_perez_salinas_fidelity_and_grad():
    d, n_layers = 9, 3
    n_blocks = d // 3
    circ = build_circuit("perez_salinas", d, n_layers, 1, "z")
    model_inf = DRUModel(circ, loss="infidelity")
    p = model_inf.init_params(seed=23, uniform_2pi=True)
    params_pl = pnp.array(p.reshape(n_layers, 3), requires_grad=True)

    qnode = _make_ps_qnode(n_blocks, n_layers)
    x = RNG.uniform(0.0, 2 * np.pi, size=d)
    x_pl = pnp.array(x, requires_grad=False)

    # Etiqueta y = 0  ->  estado |0>,  rho = |0><0|
    dm0 = pnp.array(np.array([[1, 0], [0, 0]], dtype=complex), requires_grad=False)

    def cost(par):
        return 1.0 - qnode(par, x_pl, dm0)

    ref_loss = float(cost(params_pl))
    ref_grad = np.asarray(qml.grad(cost)(params_pl)).ravel()

    mine_loss, mine_grad = model_inf.loss_and_grad(p, x.reshape(1, -1), np.array([0.0]))

    e_l = abs(ref_loss - mine_loss)
    e_g = np.max(np.abs(ref_grad - mine_grad))
    ok = e_l < 1e-12 and e_g < 1e-10
    print(f"  [{'OK ' if ok else 'FALLA'}] perez_salinas 1-F y grad   "
          f"err_loss={e_l:.2e}  err_grad_max={e_g:.2e}")
    print(f"        loss PennyLane={ref_loss:.14f}  NumPy={mine_loss:.14f}")
    return ok


def test_grad_muten_xyz():
    """Gradiente completo de una BCE con lectura XYZ, contra autograd."""
    d = 7
    circ = build_circuit("muten", d, 3, 1, "xyz")
    model = DRUModel(circ, loss="bce")
    p = model.init_params(seed=31, scale=0.8)
    X = RNG.uniform(-np.pi, np.pi, size=(12, d))
    y = RNG.integers(0, 2, size=12).astype(float)

    n_layers = 3

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def bloch(inputs, theta):
        for layer_index in range(n_layers):
            angles = theta[layer_index, :, 0] * inputs + theta[layer_index, :, 1]
            qml.Rot(angles[0], angles[1], angles[2], wires=0)
            qml.Rot(angles[3], angles[4], angles[5], wires=0)
            qml.RY(angles[6], wires=0)
        return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliY(0)), qml.expval(qml.PauliZ(0))

    gate_index = {}
    for k, g in enumerate(circ.gates):
        gate_index[(circ.layer_of_gate[k], g.feat)] = (g.w_idx, g.b_idx)

    def cost(flat):
        theta = pnp.stack([
            pnp.stack([
                pnp.stack([flat[gate_index[(l, f)][0]], flat[gate_index[(l, f)][1]]])
                for f in range(d)
            ]) for l in range(n_layers)
        ])
        readout = flat[-4:]
        total = 0.0
        for i in range(len(X)):
            bx, by, bz = bloch(pnp.array(X[i], requires_grad=False), theta)
            u = readout[0] * bx + readout[1] * by + readout[2] * bz + readout[3]
            prob = 1.0 / (1.0 + pnp.exp(-u))
            total = total + -(y[i] * pnp.log(prob + 1e-9)
                              + (1 - y[i]) * pnp.log(1 - prob + 1e-9))
        return total / len(X)

    flat = pnp.array(p, requires_grad=True)
    ref_loss = float(cost(flat))
    ref_grad = np.asarray(qml.grad(cost)(flat))
    mine_loss, mine_grad = model.loss_and_grad(p, X, y)

    e_l = abs(ref_loss - mine_loss)
    e_g = np.max(np.abs(ref_grad - mine_grad))
    ok = e_l < 1e-10 and e_g < 1e-9
    print(f"  [{'OK ' if ok else 'FALLA'}] muten+XYZ+BCE grad completo  "
          f"err_loss={e_l:.2e}  err_grad_max={e_g:.2e}")
    return ok


if __name__ == "__main__":
    print(f"PennyLane {qml.__version__}\n")
    results = [
        test_muten_expvals(),
        test_perez_salinas_fidelity_and_grad(),
        test_grad_muten_xyz(),
    ]
    print("\nRESULTADO:", "TODO OK" if all(results) else "HAY FALLAS")
    raise SystemExit(0 if all(results) else 1)
