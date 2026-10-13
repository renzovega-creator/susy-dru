"""
Nucleo del clasificador Data Re-Uploading (DRU) para el proyecto SUSY_DRU.

Simulador de estado puro de n qubits escrito en NumPy, con gradientes exactos
por el metodo *adjoint* (el mismo que usa `diff_method="adjoint"` de PennyLane).
No depende de PennyLane para correr; `test_vs_pennylane.py` verifica que el
circuito y los gradientes coinciden con PennyLane hasta ~1e-12.

Motivo de reimplementar en vez de usar PennyLane directamente:
  - el grid del proyecto es {4 perdidas} x {2 escalados} x {profundidades}
    x {>=5 semillas}; con PennyLane+autograd eso son horas de Colab, aqui son
    minutos, porque el forward y el adjoint estan vectorizados sobre el batch
    (estado de forma (B, 2**n), no un circuito por muestra);
  - permite instrumentar la varianza del gradiente parametro a parametro, que
    es lo que exige el analisis de barren plateaus de agosto.

Arquitecturas replicadas
------------------------
"perez_salinas"    Perez-Salinas et al., Quantum 4, 226 (2020), Sec. 2.
                   Por capa: bloques Rot(x_{3k}, x_{3k+1}, x_{3k+2}) NO
                   entrenables + una Rot(theta) entrenable. d debe ser
                   multiplo de 3.
"perez_salinas_w"  Variante "with weights" del mismo paper: Rot(w.x) con w
                   entrenable elemento a elemento.
"muten"            Muten, Tlimakhov & Tomut, IEEE QCE (2021) / QHack 2021, y
                   el notebook del Entregable 1: encoding y ansatz fusionados,
                   angulo = w_{l,j} * x_j + b_{l,j}.

Lecturas (readout)
------------------
"z"        p(senal) = (1 - <Z_0>)/2                       (Perez-Salinas / Muten)
"parity"   p(senal) = (1 - <Z_0 Z_1 ... Z_{n-1}>)/2       (coste GLOBAL, Cerezo)
"xyz"      p(senal) = sigmoid(a<X>+b<Y>+c<Z>+d)           (hibrida, Entregable 1)

Perdidas
--------
"fidelity"   L = mean (1 - F_y)^2          Perez-Salinas (cuadratica)
"infidelity" L = mean (1 - F_y)            lineal
"cerezo"     L = mean (1 - s_y <Z_0>)/2    coste local de Cerezo et al. (2021)
"bce"        L = mean -[y log p + (1-y) log(1-p)]
"mse"        L = mean (p - y)^2            = weighted-fidelity de Muten
"hybrid"     L = alpha * cerezo + (1-alpha) * fidelity

NOTA CIENTIFICA (verificada numericamente en `test_identidades.py`):
con n = 1 y lectura "z", las perdidas "infidelity" y "cerezo" son la MISMA
funcion, porque el proyector global |0><0| de un solo qubit ya es un observable
local. La separacion global/local de Cerezo et al. solo aparece con n >= 2
(comparar readout "parity" contra "z"). Ver HOJA_DE_RUTA, seccion 3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Algebra de un qubit
# --------------------------------------------------------------------------
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _apply_1q(state: np.ndarray, matrix: np.ndarray, wire: int, n_wires: int) -> np.ndarray:
    """Aplica una matriz 2x2 (o un lote de matrices (B,2,2)) al `wire` dado.

    state: (B, 2**n_wires) complejo.
    """
    batch = state.shape[0]
    left = 2 ** wire
    right = 2 ** (n_wires - wire - 1)
    view = state.reshape(batch, left, 2, right)
    if matrix.ndim == 2:
        out = np.einsum("ij,bljr->blir", matrix, view)
    else:  # matriz distinta por muestra
        out = np.einsum("bij,bljr->blir", matrix, view)
    return out.reshape(batch, -1)


def _rz(angle: np.ndarray) -> np.ndarray:
    """RZ(a) por muestra -> (B,2,2)."""
    half = angle / 2.0
    m = np.zeros((len(angle), 2, 2), dtype=complex)
    m[:, 0, 0] = np.exp(-1j * half)
    m[:, 1, 1] = np.exp(1j * half)
    return m


def _ry(angle: np.ndarray) -> np.ndarray:
    half = angle / 2.0
    c, s = np.cos(half), np.sin(half)
    m = np.zeros((len(angle), 2, 2), dtype=complex)
    m[:, 0, 0] = c
    m[:, 0, 1] = -s
    m[:, 1, 0] = s
    m[:, 1, 1] = c
    return m


_GEN = {  # d/da exp(-i a P/2) = (-i P/2) * exp(-i a P/2)
    "Z": -0.5j * _PAULI["Z"],
    "Y": -0.5j * _PAULI["Y"],
}
_MAT = {"Z": _rz, "Y": _ry}


# --------------------------------------------------------------------------
# Definicion del circuito
# --------------------------------------------------------------------------
@dataclass
class Gate:
    """Rotacion exp(-i a P/2) con a = coef * x[feat] + bias.

    feat < 0        -> la puerta no lee datos (rotacion puramente variacional).
    w_idx is None   -> coef fijo `w_const` (encoding NO entrenable, a la
                       Perez-Salinas).
    b_idx is None   -> sin sesgo.
    """

    pauli: str
    wire: int
    feat: int = -1
    w_idx: int | None = None
    b_idx: int | None = None
    w_const: float = 1.0


@dataclass
class Circuit:
    n_wires: int
    gates: list[Gate]
    n_params: int
    entangle_every_layer: bool = False
    cnot_pairs: list[tuple[int, int]] = field(default_factory=list)
    layer_of_gate: list[int] = field(default_factory=list)
    n_layers: int = 1
    readout: str = "z"
    readout_slice: slice | None = None
    name: str = ""
    w_indices: list[int] = field(default_factory=list)
    b_indices: list[int] = field(default_factory=list)

    def angles(self, params: np.ndarray, X: np.ndarray) -> np.ndarray:
        """(n_gates, B) con los angulos efectivos de cada puerta."""
        out = np.empty((len(self.gates), len(X)))
        for k, g in enumerate(self.gates):
            a = np.zeros(len(X))
            if g.feat >= 0:
                coef = g.w_const if g.w_idx is None else params[g.w_idx]
                a = coef * X[:, g.feat]
            if g.b_idx is not None:
                a = a + params[g.b_idx]
            out[k] = a
        return out


def _apply_cnot(state: np.ndarray, control: int, target: int, n_wires: int) -> np.ndarray:
    batch = state.shape[0]
    view = state.reshape((batch,) + (2,) * n_wires)
    view = view.copy()
    idx_c1 = [slice(None)] * (n_wires + 1)
    idx_c1[control + 1] = 1
    block = view[tuple(idx_c1)]  # subespacio control=1
    block = np.flip(block, axis=(target + 1) if target < control else target)
    view[tuple(idx_c1)] = block
    return view.reshape(batch, -1)


# --------------------------------------------------------------------------
# Constructores de arquitectura
# --------------------------------------------------------------------------
def build_circuit(
    arch: str,
    n_features: int,
    n_layers: int,
    n_wires: int = 1,
    readout: str = "z",
    entangle: bool = False,
) -> Circuit:
    gates: list[Gate] = []
    layer_of: list[int] = []
    p = 0  # contador de parametros

    def add(pauli, wire, feat=-1, trainable_w=False, bias=False, w_const=1.0, layer=0):
        nonlocal p
        w_idx = None
        b_idx = None
        if trainable_w:
            w_idx = p
            p += 1
        if bias:
            b_idx = p
            p += 1
        gates.append(Gate(pauli, wire, feat, w_idx, b_idx, w_const))
        layer_of.append(layer)

    def rot(wire, feats, trainable_w, bias, layer):
        """Rot(phi,theta,omega) = RZ(omega) RY(theta) RZ(phi) -> orden RZ,RY,RZ."""
        for pauli, f in zip(("Z", "Y", "Z"), feats):
            add(pauli, wire, f, trainable_w, bias, layer=layer)

    if arch in ("perez_salinas", "perez_salinas_w"):
        if n_features % 3 != 0:
            raise ValueError(
                f"{arch} necesita d multiplo de 3 (usa pad_to_multiple_of=3); d={n_features}"
            )
        trainable_w = arch.endswith("_w")
        blocks = n_features // 3
        for layer in range(n_layers):
            for wire in range(n_wires):
                for b in range(blocks):
                    rot(wire, (3 * b, 3 * b + 1, 3 * b + 2), trainable_w, False, layer)
                rot(wire, (-1, -1, -1), False, True, layer)  # Rot(theta) variacional

    elif arch == "muten":
        # Encoding y ansatz fusionados: angulo = w*x + b, tripletes -> Rot, resto -> RY.
        for layer in range(n_layers):
            for wire in range(n_wires):
                full = n_features // 3
                for b in range(full):
                    rot(wire, (3 * b, 3 * b + 1, 3 * b + 2), True, True, layer)
                for f in range(3 * full, n_features):
                    add("Y", wire, f, True, True, layer=layer)
    else:
        raise ValueError(f"arquitectura desconocida: {arch}")

    w_indices = [g.w_idx for g in gates if g.w_idx is not None]
    b_indices = [g.b_idx for g in gates if g.b_idx is not None]
    n_params = p
    readout_slice = None
    if readout == "xyz":
        readout_slice = slice(n_params, n_params + 4)
        n_params += 4

    circ = Circuit(
        n_wires=n_wires,
        gates=gates,
        n_params=n_params,
        layer_of_gate=layer_of,
        n_layers=n_layers,
        readout=readout,
        readout_slice=readout_slice,
        name=f"{arch}|L{n_layers}|n{n_wires}|{readout}",
        w_indices=w_indices,
        b_indices=b_indices,
    )
    if entangle and n_wires > 1:
        circ.cnot_pairs = [(w, (w + 1) % n_wires) for w in range(n_wires)]
        circ.entangle_every_layer = True
    return circ


# --------------------------------------------------------------------------
# Forward + adjoint
# --------------------------------------------------------------------------
def _observable_expval(state: np.ndarray, pauli_string: str) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (<O>, O|psi>) para O = tensor de Paulis dado por `pauli_string`."""
    n = len(pauli_string)
    out = state
    for wire, pauli in enumerate(pauli_string):
        if pauli == "I":
            continue
        out = _apply_1q(out, _PAULI[pauli], wire, n)
    val = np.real(np.einsum("bi,bi->b", np.conj(state), out))
    return val, out


def _forward_states(circ: Circuit, params: np.ndarray, X: np.ndarray):
    """Estado final y lista de matrices por puerta (para el adjoint)."""
    B = len(X)
    state = np.zeros((B, 2 ** circ.n_wires), dtype=complex)
    state[:, 0] = 1.0
    angles = circ.angles(params, X)
    mats = []
    gi = 0
    for layer in range(circ.n_layers):
        while gi < len(circ.gates) and circ.layer_of_gate[gi] == layer:
            g = circ.gates[gi]
            m = _MAT[g.pauli](angles[gi])
            mats.append(m)
            state = _apply_1q(state, m, g.wire, circ.n_wires)
            gi += 1
        if circ.entangle_every_layer:
            for c, t in circ.cnot_pairs:
                if c != t:
                    state = _apply_cnot(state, c, t, circ.n_wires)
    return state, angles, mats


def bloch_and_grads(
    circ: Circuit, params: np.ndarray, X: np.ndarray, observables: tuple[str, ...]
):
    """Valores esperados de `observables` y su gradiente respecto de cada angulo.

    Devuelve (vals, dvals) con
        vals  : (n_obs, B)
        dvals : (n_obs, n_gates, B)   d<O>/d(angulo de la puerta k)
    """
    state, angles, mats = _forward_states(circ, params, X)
    n_obs = len(observables)
    vals = np.empty((n_obs, len(X)))
    dvals = np.zeros((n_obs, len(circ.gates), len(X)))

    for oi, obs in enumerate(observables):
        v, o_psi = _observable_expval(state, obs)
        vals[oi] = v
        psi = state.copy()
        lam = o_psi.copy()
        gi = len(circ.gates) - 1
        for layer in reversed(range(circ.n_layers)):
            if circ.entangle_every_layer:
                for c, t in reversed(circ.cnot_pairs):
                    if c != t:
                        psi = _apply_cnot(psi, c, t, circ.n_wires)
                        lam = _apply_cnot(lam, c, t, circ.n_wires)
            while gi >= 0 and circ.layer_of_gate[gi] == layer:
                g = circ.gates[gi]
                m = mats[gi]
                m_dag = np.conj(np.transpose(m, (0, 2, 1)))
                psi = _apply_1q(psi, m_dag, g.wire, circ.n_wires)  # psi_{k-1}
                mu = _apply_1q(psi, m, g.wire, circ.n_wires)
                mu = _apply_1q(mu, _GEN[g.pauli], g.wire, circ.n_wires)
                dvals[oi, gi] = 2.0 * np.real(np.einsum("bi,bi->b", np.conj(lam), mu))
                lam = _apply_1q(lam, m_dag, g.wire, circ.n_wires)
                gi -= 1
    return vals, dvals


def _chain_to_params(circ: Circuit, dL_dangle: np.ndarray, X: np.ndarray, n_params: int):
    """Propaga dL/d(angulo) a dL/d(parametros) por la regla de la cadena."""
    grad = np.zeros(n_params)
    for k, g in enumerate(circ.gates):
        d = dL_dangle[k]
        if g.w_idx is not None and g.feat >= 0:
            grad[g.w_idx] += float(np.dot(d, X[:, g.feat]))
        if g.b_idx is not None:
            grad[g.b_idx] += float(np.sum(d))
    return grad


# --------------------------------------------------------------------------
# Modelo: lectura + perdida
# --------------------------------------------------------------------------
def _sigmoid(u):
    return 0.5 * (1.0 + np.tanh(0.5 * u))  # estable numericamente


LOSSES = ("fidelity", "infidelity", "cerezo", "bce", "mse", "hybrid")


@dataclass
class DRUModel:
    circuit: Circuit
    loss: str = "bce"
    alpha: float = 0.5           # solo para loss="hybrid"
    l2: float = 0.0

    def __post_init__(self):
        if self.loss not in LOSSES:
            raise ValueError(f"loss desconocida: {self.loss} (opciones: {LOSSES})")
        n = self.circuit.n_wires
        if self.circuit.readout == "parity":
            self._z_obs = "Z" * n
        else:
            self._z_obs = "Z" + "I" * (n - 1)
        if self.circuit.readout == "xyz":
            self._obs = ("X" + "I" * (n - 1), "Y" + "I" * (n - 1), self._z_obs)
        else:
            self._obs = (self._z_obs,)

    # ---- inicializacion ----
    def init_params(self, seed: int, scale: float = 0.05, uniform_2pi: bool = False):
        """Inicializacion de parametros.

        `uniform_2pi=False` -> convencion Muten / notebook del Entregable 1:
            todo ~ scale * N(0,1). Angulos pequenos, arranque cerca de |0>.
        `uniform_2pi=True`  -> convencion Perez-Salinas / SEED del proyecto:
            los angulos variacionales (sesgos) ~ U(0, 2pi); los PESOS del
            encoding, si existen, ~ 1 + 0.1 N(0,1). Inicializar tambien los
            pesos en U(0,2pi) los hace multiplicar x en [-pi,pi] hasta ~20 rad
            y destruye el entrenamiento (AUC ~ 0.5); es un error facil de
            cometer y por eso queda explicito aqui.
        """
        rng = np.random.default_rng(seed)
        n = self.circuit.n_params
        if uniform_2pi:
            p = rng.uniform(0.0, 2 * np.pi, size=n)
            if self.circuit.w_indices:
                p[self.circuit.w_indices] = 1.0 + 0.1 * rng.normal(size=len(self.circuit.w_indices))
        else:
            p = scale * rng.normal(size=n)
        sl = self.circuit.readout_slice
        if sl is not None:
            p[sl] = np.array([0.0, 0.0, -1.0, 0.0])  # arranque semantico ~ p(|1>)
        return p

    # ---- probabilidad p(senal) ----
    def _probability(self, params, vals):
        if self.circuit.readout == "xyz":
            a, b, c, d = params[self.circuit.readout_slice]
            u = a * vals[0] + b * vals[1] + c * vals[2] + d
            return _sigmoid(u), u
        z = vals[0]
        return 0.5 * (1.0 - z), None

    def predict_proba(self, params, X, batch_size: int = 4096):
        out = []
        for start in range(0, len(X), batch_size):
            chunk = X[start:start + batch_size]
            state, _, _ = _forward_states(self.circuit, params, chunk)
            vals = np.stack([_observable_expval(state, o)[0] for o in self._obs])
            out.append(self._probability(params, vals)[0])
        return np.concatenate(out)

    # ---- perdida + gradiente ----
    def loss_and_grad(self, params, X, y):
        circ = self.circuit
        vals, dvals = bloch_and_grads(circ, params, X, self._obs)
        B = len(X)
        grad = np.zeros(circ.n_params)

        z = vals[-1]  # <Z> del observable de lectura
        s = 1.0 - 2.0 * y  # +1 para fondo (y=0), -1 para senal (y=1)

        if self.loss in ("cerezo", "fidelity", "infidelity", "hybrid"):
            # F = fidelidad con el estado-etiqueta = (1 + s*z)/2
            F = 0.5 * (1.0 + s * z)
            if self.loss == "cerezo" or self.loss == "infidelity":
                value = float(np.mean(1.0 - F))
                dL_dz = -0.5 * s / B
            elif self.loss == "fidelity":
                value = float(np.mean((1.0 - F) ** 2))
                dL_dz = (-2.0 * (1.0 - F) * 0.5 * s) / B
            else:  # hybrid
                a = self.alpha
                value = float(np.mean(a * (1.0 - F) + (1.0 - a) * (1.0 - F) ** 2))
                dL_dz = (a * (-0.5 * s) + (1.0 - a) * (-2.0 * (1.0 - F) * 0.5 * s)) / B
            dL_dangle = np.einsum("b,kb->kb", dL_dz, dvals[-1])
            grad += _chain_to_params(circ, dL_dangle, X, circ.n_params)

        else:  # perdidas sobre la probabilidad p
            p, u = self._probability(params, vals)
            eps = 1e-9
            if self.loss == "bce":
                value = float(np.mean(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))
                dL_dp = (p - y) / np.clip(p * (1 - p), 1e-9, None) / B
            else:  # mse
                value = float(np.mean((p - y) ** 2))
                dL_dp = 2.0 * (p - y) / B

            if circ.readout == "xyz":
                a, b, c, d = params[circ.readout_slice]
                dsig = p * (1.0 - p)
                dL_du = dL_dp * dsig
                sl = circ.readout_slice
                grad[sl.start + 0] += float(np.dot(dL_du, vals[0]))
                grad[sl.start + 1] += float(np.dot(dL_du, vals[1]))
                grad[sl.start + 2] += float(np.dot(dL_du, vals[2]))
                grad[sl.start + 3] += float(np.sum(dL_du))
                dL_dvals = np.stack([dL_du * a, dL_du * b, dL_du * c])
                dL_dangle = np.einsum("ob,okb->kb", dL_dvals, dvals)
            else:
                dL_dz = dL_dp * (-0.5)
                dL_dangle = np.einsum("b,kb->kb", dL_dz, dvals[-1])
            grad += _chain_to_params(circ, dL_dangle, X, circ.n_params)

        if self.l2:
            mask = np.ones(circ.n_params)
            if circ.readout_slice is not None:
                mask[circ.readout_slice] = 0.0  # no regularizar la lectura
            value += float(self.l2 * np.mean((params * mask) ** 2))
            grad += 2.0 * self.l2 * params * mask / circ.n_params
        return value, grad


# --------------------------------------------------------------------------
# Optimizacion
# --------------------------------------------------------------------------
class Adam:
    def __init__(self, n, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0

    def step(self, params, grad, mask=None):
        self.t += 1
        if mask is not None:
            grad = grad * mask
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        upd = self.lr * mhat / (np.sqrt(vhat) + self.eps)
        if mask is not None:
            upd = upd * mask
        return params - upd


def layer_mask(circ: Circuit, layers) -> np.ndarray:
    """Mascara 0/1 sobre parametros: 1 en los parametros de `layers`.

    Implementa el crecimiento *layerwise* de Skolik et al. (2021) tal como lo
    usa Muten et al.: se entrena la capa l dejando congeladas las demas.
    """
    mask = np.zeros(circ.n_params)
    layers = {layers} if isinstance(layers, int) else set(layers)
    for k, g in enumerate(circ.gates):
        if circ.layer_of_gate[k] in layers:
            if getattr(g, "w_idx", None) is not None:
                mask[g.w_idx] = 1.0
            if getattr(g, "b_idx", None) is not None:
                mask[g.b_idx] = 1.0
    if circ.readout_slice is not None:
        mask[circ.readout_slice] = 1.0  # la lectura siempre se entrena
    return mask
