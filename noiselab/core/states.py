"""
Модуль квантовых состояний для 1 кубита
Реализация statevector и density matrix с корректными физическими проверками
"""

import numpy as np
from typing import Union, Optional, List
from numpy.typing import NDArray


class QuantumState:
    """
    Квантовое состояние в представлении statevector (чистое состояние)
    |ψ⟩ = Σᵢ αᵢ|i⟩, где Σᵢ|αᵢ|² = 1
    """

    def __init__(self, statevector: Union[NDArray[np.complex128], List[complex]],
                 normalize: bool = True):
        """
        Args:
            statevector: Комплексный вектор состояния
            normalize: Нормализовать ли вектор автоматически
        """
        self.statevector = np.array(statevector, dtype=np.complex128)

        # Проверка, что размерность - степень двойки (n кубитов)
        dim = len(self.statevector)
        if not self._is_power_of_two(dim):
            raise ValueError(f"Размерность {dim} не является степенью 2")

        self.n_qubits = int(np.log2(dim))

        if normalize:
            self._normalize()
        else:
            # Проверка нормировки
            norm = np.linalg.norm(self.statevector)
            if not np.isclose(norm, 1.0, atol=1e-10):
                raise ValueError(f"Вектор состояния не нормирован: ||ψ|| = {norm}")

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        """Проверка, что n - степень двойки"""
        return n > 0 and (n & (n - 1)) == 0

    def _normalize(self):
        """Нормализация вектора состояния"""
        norm = np.linalg.norm(self.statevector)
        if norm < 1e-15:
            raise ValueError("Нулевой вектор не может быть нормализован")
        self.statevector /= norm

    def to_density_matrix(self) -> 'DensityMatrix':
        """Преобразование в матрицу плотности: ρ = |ψ⟩⟨ψ|"""
        rho = np.outer(self.statevector, self.statevector.conj())
        return DensityMatrix(rho)

    def probability(self, basis_state: int) -> float:
        """
        Вероятность измерения в базисном состоянии |i⟩
        P(i) = |⟨i|ψ⟩|² = |αᵢ|²
        """
        if basis_state >= len(self.statevector):
            raise ValueError(f"Базисное состояние {basis_state} вне диапазона")
        return np.abs(self.statevector[basis_state]) ** 2

    def measure(self, shots: int = 1) -> NDArray[np.int32]:
        """
        Симуляция измерения в вычислительном базисе

        Args:
            shots: Число измерений

        Returns:
            Массив результатов измерений (индексы базисных состояний)
        """
        probabilities = np.abs(self.statevector) ** 2
        outcomes = np.random.choice(
            len(self.statevector),
            size=shots,
            p=probabilities
        )
        return outcomes

    def expectation_value(self, operator: NDArray[np.complex128]) -> complex:
        """
        Среднее значение оператора: ⟨ψ|Ô|ψ⟩
        """
        return np.vdot(self.statevector, operator @ self.statevector)

    def __repr__(self) -> str:
        return f"QuantumState(n_qubits={self.n_qubits}, dim={len(self.statevector)})"

    def __str__(self) -> str:
        """Красивое представление состояния в базисе |0⟩, |1⟩, ..."""
        terms = []
        for i, amp in enumerate(self.statevector):
            if np.abs(amp) > 1e-10:  # Показываем только значимые компоненты
                binary = format(i, f'0{self.n_qubits}b')
                real = amp.real
                imag = amp.imag

                if np.abs(imag) < 1e-10:
                    terms.append(f"{real:.4f}|{binary}⟩")
                elif np.abs(real) < 1e-10:
                    terms.append(f"{imag:.4f}i|{binary}⟩")
                else:
                    terms.append(f"({real:.4f}{imag:+.4f}i)|{binary}⟩")

        return " + ".join(terms) if terms else "0"

    @classmethod
    def zero_state(cls, n_qubits: int) -> 'QuantumState':
        """Создать состояние |0...0⟩"""
        dim = 2 ** n_qubits
        statevector = np.zeros(dim, dtype=np.complex128)
        statevector[0] = 1.0
        return cls(statevector, normalize=False)

    @classmethod
    def plus_state(cls, n_qubits: int) -> 'QuantumState':
        """Создать состояние |+...+⟩ = (|0⟩ + |1⟩)/√2 для каждого кубита"""
        dim = 2 ** n_qubits
        statevector = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
        return cls(statevector, normalize=False)

    @classmethod
    def computational_basis_state(cls, n_qubits: int, index: int) -> 'QuantumState':
        """Создать базисное состояние |i⟩"""
        dim = 2 ** n_qubits
        if index >= dim:
            raise ValueError(f"Индекс {index} вне диапазона для {n_qubits} кубитов")
        statevector = np.zeros(dim, dtype=np.complex128)
        statevector[index] = 1.0
        return cls(statevector, normalize=False)


class DensityMatrix:
    """
    Матрица плотности для представления квантовых состояний
    Может описывать как чистые, так и смешанные состояния

    Свойства:
    - ρ† = ρ (эрмитова)
    - ρ ≥ 0 (положительно полуопределённая)
    - Tr(ρ) = 1 (нормировка)
    - Для чистых состояний: ρ² = ρ, Tr(ρ²) = 1
    """

    def __init__(self, matrix: NDArray[np.complex128], validate: bool = True):
        """
        Args:
            matrix: Матрица плотности
            validate: Проверять ли физические условия
        """
        self.matrix = np.array(matrix, dtype=np.complex128)

        if self.matrix.ndim != 2:
            raise ValueError("Матрица плотности должна быть двумерной")

        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Матрица плотности должна быть квадратной")

        dim = self.matrix.shape[0]
        if not QuantumState._is_power_of_two(dim):
            raise ValueError(f"Размерность {dim} не является степенью 2")

        self.n_qubits = int(np.log2(dim))

        if validate:
            self._validate()

    def _validate(self):
        """Проверка физических условий"""
        # Проверка эрмитовости
        if not np.allclose(self.matrix, self.matrix.conj().T, atol=1e-10):
            raise ValueError("Матрица плотности не является эрмитовой")

        # Проверка нормировки
        trace = np.trace(self.matrix)
        if not np.isclose(trace, 1.0, atol=1e-10):
            raise ValueError(f"След матрицы плотности не равен 1: Tr(ρ) = {trace}")

        # Проверка положительной полуопределённости
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        if np.any(eigenvalues < -1e-10):
            raise ValueError(f"Матрица плотности не положительно полуопределённая: "
                           f"min eigenvalue = {eigenvalues.min()}")

    def is_pure(self, tol: float = 1e-10) -> bool:
        """
        Проверка, является ли состояние чистым
        Чистое состояние: Tr(ρ²) = 1
        """
        purity = np.trace(self.matrix @ self.matrix).real
        return np.isclose(purity, 1.0, atol=tol)

    def purity(self) -> float:
        """Чистота состояния: Tr(ρ²) ∈ [1/d, 1]"""
        return np.trace(self.matrix @ self.matrix).real


    def probability(self, basis_state: int) -> float:
        """Вероятность измерения в базисном состоянии |i⟩: P(i) = ⟨i|ρ|i⟩"""
        return self.matrix[basis_state, basis_state].real

    def measure(self, shots: int = 1) -> NDArray[np.int32]:
        """Симуляция измерения в вычислительном базисе"""
        probabilities = np.diag(self.matrix).real
        probabilities = np.clip(probabilities, 0, 1)  # Численная стабильность
        probabilities /= probabilities.sum()  # Ренормализация

        outcomes = np.random.choice(
            len(probabilities),
            size=shots,
            p=probabilities
        )
        return outcomes

    def expectation_value(self, operator: NDArray[np.complex128]) -> complex:
        """Среднее значение оператора: ⟨Ô⟩ = Tr(ρÔ)"""
        return np.trace(self.matrix @ operator)

    def fidelity(self, other: 'DensityMatrix') -> float:
        """
        Fidelity между двумя состояниями:
        F(ρ₁, ρ₂) = (Tr√(√ρ₁ ρ₂ √ρ₁))²

        Для чистых состояний: F = |⟨ψ₁|ψ₂⟩|²
        """
        # Упрощённая формула через матричный квадратный корень
        sqrt_rho1 = self._matrix_sqrt(self.matrix)
        M = sqrt_rho1 @ other.matrix @ sqrt_rho1
        sqrt_M = self._matrix_sqrt(M)
        fidelity = np.trace(sqrt_M).real ** 2
        return np.clip(fidelity, 0.0, 1.0)

    @staticmethod
    def _matrix_sqrt(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Квадратный корень из матрицы через диагонализацию"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Численная стабильность
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        return eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T

    def trace_distance(self, other: 'DensityMatrix') -> float:
        """
        Trace distance: D(ρ₁, ρ₂) = ½ Tr|ρ₁ - ρ₂|
        где |A| = √(A†A)
        """
        diff = self.matrix - other.matrix
        singular_values = np.linalg.svd(diff, compute_uv=False)
        return 0.5 * np.sum(singular_values)

    def __repr__(self) -> str:
        pure_str = "pure" if self.is_pure() else "mixed"
        return f"DensityMatrix(n_qubits={self.n_qubits}, {pure_str}, " \
               f"purity={self.purity():.4f})"


    @classmethod
    def maximally_mixed(cls, n_qubits: int) -> 'DensityMatrix':
        """Максимально смешанное состояние: ρ = I/d"""
        dim = 2 ** n_qubits
        matrix = np.eye(dim, dtype=np.complex128) / dim
        return cls(matrix, validate=False)