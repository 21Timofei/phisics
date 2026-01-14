"""
Модуль квантовых гейтов
Реализация унитарных операций с физическими проверками
"""

import numpy as np
from typing import Optional, List
from numpy.typing import NDArray
from .states import QuantumState, DensityMatrix


class QuantumGate:
    """
    Квантовый гейт как унитарная матрица
    U†U = UU† = I
    """

    def __init__(self, matrix: NDArray[np.complex128],
                 name: str = "Gate",
                 validate: bool = True):
        """
        Args:
            matrix: Унитарная матрица
            name: Название гейта
            validate: Проверять ли унитарность
        """
        self.matrix = np.array(matrix, dtype=np.complex128)
        self.name = name

        if self.matrix.ndim != 2:
            raise ValueError("Матрица гейта должна быть двумерной")

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Матрица гейта должна быть квадратной")

        dim = self.matrix.shape[0]
        if not self._is_power_of_two(dim):
            raise ValueError(f"Размерность {dim} не является степенью 2")

        self.n_qubits = int(np.log2(dim))

        if validate:
            self._validate_unitary()

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def _validate_unitary(self, tol: float = 1e-10):
        """Проверка унитарности: U†U = I"""
        identity = np.eye(len(self.matrix), dtype=np.complex128)
        product = self.matrix.conj().T @ self.matrix

        if not np.allclose(product, identity, atol=tol):
            error = np.linalg.norm(product - identity)
            raise ValueError(f"Матрица не является унитарной. ||U†U - I|| = {error}")

    def apply(self, state: QuantumState) -> QuantumState:
        """Применить гейт к состоянию: |ψ'⟩ = U|ψ⟩"""
        if state.n_qubits != self.n_qubits:
            raise ValueError(f"Несоответствие размерностей: гейт {self.n_qubits} кубитов, "
                           f"состояние {state.n_qubits} кубитов")

        new_statevector = self.matrix @ state.statevector
        return QuantumState(new_statevector, normalize=False)

    def apply_density(self, rho: DensityMatrix) -> DensityMatrix:
        """Применить гейт к матрице плотности: ρ' = UρU†"""
        if rho.n_qubits != self.n_qubits:
            raise ValueError(f"Несоответствие размерностей")

        new_matrix = self.matrix @ rho.matrix @ self.matrix.conj().T
        return DensityMatrix(new_matrix, validate=False)

    def dagger(self) -> 'QuantumGate':
        """Сопряжённый гейт: U†"""
        return QuantumGate(self.matrix.conj().T, name=f"{self.name}†", validate=False)

    def compose(self, other: 'QuantumGate') -> 'QuantumGate':
        """Композиция гейтов: V∘U (сначала U, потом V)"""
        if self.n_qubits != other.n_qubits:
            raise ValueError("Гейты должны иметь одинаковое число кубитов")

        composed_matrix = self.matrix @ other.matrix
        return QuantumGate(composed_matrix, name=f"{self.name}∘{other.name}", validate=False)

    def tensor(self, other: 'QuantumGate') -> 'QuantumGate':
        """Тензорное произведение: U ⊗ V"""
        tensor_matrix = np.kron(self.matrix, other.matrix)
        return QuantumGate(tensor_matrix, name=f"{self.name}⊗{other.name}", validate=False)

    def __matmul__(self, other):
        """Перегрузка оператора @ для композиции"""
        if isinstance(other, QuantumGate):
            return self.compose(other)
        elif isinstance(other, QuantumState):
            return self.apply(other)
        elif isinstance(other, DensityMatrix):
            return self.apply_density(other)
        else:
            raise TypeError(f"Неподдерживаемый тип: {type(other)}")

    def __repr__(self) -> str:
        return f"QuantumGate(name='{self.name}', n_qubits={self.n_qubits})"


class PauliGates:
    """Паули-гейты: базис для однокубитных операций"""

    # Паули матрицы (фундаментальные константы квантовой механики)
    I = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    @classmethod
    def identity(cls) -> QuantumGate:
        """Единичный гейт"""
        return QuantumGate(cls.I, name="I", validate=False)

    @classmethod
    def pauli_x(cls) -> QuantumGate:
        """Паули X (bit-flip, NOT gate)"""
        return QuantumGate(cls.X, name="X", validate=False)

    @classmethod
    def pauli_y(cls) -> QuantumGate:
        """Паули Y"""
        return QuantumGate(cls.Y, name="Y", validate=False)

    @classmethod
    def pauli_z(cls) -> QuantumGate:
        """Паули Z (phase-flip)"""
        return QuantumGate(cls.Z, name="Z", validate=False)

    @classmethod
    def hadamard(cls) -> QuantumGate:
        """
        Гейт Адамара: H = (X + Z)/√2
        Создаёт суперпозицию: H|0⟩ = |+⟩, H|1⟩ = |-⟩
        """
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        return QuantumGate(H, name="H", validate=False)

    @classmethod
    def s_gate(cls) -> QuantumGate:
        """
        S gate (фазовый): S = diag(1, i) = √Z
        S|+⟩ = |+i⟩
        """
        S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
        return QuantumGate(S, name="S", validate=False)

    @classmethod
    def t_gate(cls) -> QuantumGate:
        """
        T gate: T = diag(1, e^(iπ/4)) = √S
        """
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        return QuantumGate(T, name="T", validate=False)

    @classmethod
    def sdg_gate(cls) -> QuantumGate:
        """S† gate"""
        Sdg = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
        return QuantumGate(Sdg, name="S†", validate=False)

    @classmethod
    def tdg_gate(cls) -> QuantumGate:
        """T† gate"""
        Tdg = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128)
        return QuantumGate(Tdg, name="T†", validate=False)


class RotationGates:
    """Параметризованные вращения вокруг осей Блоха"""

    @staticmethod
    def rx(theta: float) -> QuantumGate:
        """
        Вращение вокруг оси X: Rx(θ) = exp(-iθX/2)
        = cos(θ/2)I - i sin(θ/2)X
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([
            [c, -1j * s],
            [-1j * s, c]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name=f"Rx({theta:.3f})", validate=False)

    @staticmethod
    def ry(theta: float) -> QuantumGate:
        """
        Вращение вокруг оси Y: Ry(θ) = exp(-iθY/2)
        = cos(θ/2)I - i sin(θ/2)Y
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([
            [c, -s],
            [s, c]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name=f"Ry({theta:.3f})", validate=False)

    @staticmethod
    def rz(theta: float) -> QuantumGate:
        """
        Вращение вокруг оси Z: Rz(θ) = exp(-iθZ/2)
        = diag(e^(-iθ/2), e^(iθ/2))
        """
        matrix = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name=f"Rz({theta:.3f})", validate=False)

    @staticmethod
    def u3(theta: float, phi: float, lambda_: float) -> QuantumGate:
        """
        Произвольный однокубитный гейт U3(θ, φ, λ)
        Любой однокубитный гейт может быть представлен в этой форме

        U3(θ, φ, λ) = [
            [cos(θ/2), -e^(iλ)sin(θ/2)],
            [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]
        ]
        """
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        matrix = np.array([
            [c, -np.exp(1j * lambda_) * s],
            [np.exp(1j * phi) * s, np.exp(1j * (phi + lambda_)) * c]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name=f"U3({theta:.2f},{phi:.2f},{lambda_:.2f})",
                          validate=False)

    @staticmethod
    def phase(theta: float) -> QuantumGate:
        """
        Фазовый гейт: P(θ) = diag(1, e^(iθ))
        """
        matrix = np.array([
            [1, 0],
            [0, np.exp(1j * theta)]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name=f"P({theta:.3f})", validate=False)


class TwoQubitGates:
    """Двухкубитные гейты"""

    @staticmethod
    def cnot(control: int = 0, target: int = 1) -> QuantumGate:
        """
        Controlled-NOT (CNOT, CX)
        |control,target⟩ → |control, target ⊕ control⟩

        Матрица для стандартного порядка (control=0, target=1):
        |00⟩ → |00⟩
        |01⟩ → |01⟩
        |10⟩ → |11⟩
        |11⟩ → |10⟩
        """
        if control == target:
            raise ValueError("Control и target должны быть разными кубитами")

        # Стандартная матрица CNOT для 2 кубитов
        if control == 0 and target == 1:
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=np.complex128)
        elif control == 1 and target == 0:
            # CNOT с перевёрнутым порядком
            matrix = np.array([
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ], dtype=np.complex128)
        else:
            raise ValueError("Для >2 кубитов используйте расширенную версию")

        return QuantumGate(matrix, name=f"CNOT({control},{target})", validate=False)

    @staticmethod
    def cz() -> QuantumGate:
        """
        Controlled-Z
        Применяет Z к target, если control = |1⟩
        CZ = diag(1, 1, 1, -1)
        """
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name="CZ", validate=False)

    @staticmethod
    def swap() -> QuantumGate:
        """
        SWAP gate: меняет местами два кубита
        |00⟩ → |00⟩, |01⟩ → |10⟩, |10⟩ → |01⟩, |11⟩ → |11⟩
        """
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name="SWAP", validate=False)

    @staticmethod
    def controlled_u(u: QuantumGate) -> QuantumGate:
        """
        Создать управляемую версию однокубитного гейта
        CU = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ U
        """
        if u.n_qubits != 1:
            raise ValueError("Можно контролировать только однокубитные гейты")

        I = np.eye(2, dtype=np.complex128)
        # Блочная матрица 4x4
        matrix = np.block([
            [I, np.zeros((2, 2), dtype=np.complex128)],
            [np.zeros((2, 2), dtype=np.complex128), u.matrix]
        ])
        return QuantumGate(matrix, name=f"C-{u.name}", validate=False)

    @staticmethod
    def iswap() -> QuantumGate:
        """
        iSWAP gate: SWAP с дополнительной фазой
        """
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)
        return QuantumGate(matrix, name="iSWAP", validate=False)


class ThreeQubitGates:
    """Трёхкубитные гейты"""

    @staticmethod
    def toffoli() -> QuantumGate:
        """
        Toffoli gate (CCNOT): NOT на target, если оба control = |1⟩
        Используется для классических вычислений
        """
        matrix = np.eye(8, dtype=np.complex128)
        # Меняем только последние два базисных состояния |110⟩ ↔ |111⟩
        matrix[6, 6] = 0
        matrix[6, 7] = 1
        matrix[7, 6] = 1
        matrix[7, 7] = 0
        return QuantumGate(matrix, name="Toffoli", validate=False)

    @staticmethod
    def fredkin() -> QuantumGate:
        """
        Fredkin gate (CSWAP): SWAP двух кубитов, если control = |1⟩
        """
        matrix = np.eye(8, dtype=np.complex128)
        # SWAP только для |1⟩ на первом кубите
        # |101⟩ ↔ |110⟩
        matrix[5, 5] = 0
        matrix[5, 6] = 1
        matrix[6, 5] = 1
        matrix[6, 6] = 0
        return QuantumGate(matrix, name="Fredkin", validate=False)


# Удобные алиасы для часто используемых гейтов
X = PauliGates.pauli_x
Y = PauliGates.pauli_y
Z = PauliGates.pauli_z
H = PauliGates.hadamard
S = PauliGates.s_gate
T = PauliGates.t_gate
I = PauliGates.identity

Rx = RotationGates.rx
Ry = RotationGates.ry
Rz = RotationGates.rz
U3 = RotationGates.u3

CNOT = TwoQubitGates.cnot
CZ = TwoQubitGates.cz
SWAP = TwoQubitGates.swap

Toffoli = ThreeQubitGates.toffoli
Fredkin = ThreeQubitGates.fredkin
