"""
Модуль квантовых гейтов для 1 кубита
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
