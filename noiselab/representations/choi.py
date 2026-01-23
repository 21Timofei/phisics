"""
Choi-Jamiolkowski representation квантовых каналов
"""

import numpy as np
from typing import Tuple, List
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ChoiRepresentation:
    """
    Представление квантового канала через матрицу Чои

    Choi-Jamiolkowski изоморфизм устанавливает биекцию между:
    - Линейными отображениями ε: ρ → ρ'
    - Операторами J действующими на расширенном пространстве

    J(ε) = (I ⊗ ε)(|Φ⁺⟩⟨Φ⁺|)

    где |Φ⁺⟩ = Σᵢ|i⟩|i⟩/√d - максимально запутанное состояние

    Физические свойства:
    1. ε - CPTP ⟺ J ≥ 0 и Tr_B(J) = I
    2. ε - унитарный ⟺ J = |Ψ_U⟩⟨Ψ_U| (ранг 1)
    3. Ранг J = минимальное число операторов Крауса
    """

    def __init__(self, choi_matrix: NDArray[np.complex128]):
        """
        Args:
            choi_matrix: Матрица Чои размера 4×4 (для 1 кубита)
        """
        self.n_qubits = 1
        self.dim = 2
        self.choi_matrix = choi_matrix

        if choi_matrix.shape != (4, 4):
            raise ValueError(f"Неправильная размерность Choi matrix: {choi_matrix.shape}, ожидается (4, 4)")

    def is_completely_positive(self, tol: float = 1e-10) -> bool:
        """
        Проверка completely positive: J ≥ 0

        Канал CP если все собственные значения J неотрицательны
        """
        eigenvalues = np.linalg.eigvalsh(self.choi_matrix)
        return np.all(eigenvalues >= -tol)

    def is_trace_preserving(self, tol: float = 1e-8) -> bool:
        """
        Проверка trace-preserving: Tr_B(J) = I

        Частичный след по второй подсистеме должен дать identity
        """
        from ..core.tensor import partial_trace

        dims = [self.dim, self.dim]
        reduced = partial_trace(self.choi_matrix, dims, trace_out=1)

        identity = np.eye(self.dim, dtype=np.complex128)
        error = np.linalg.norm(reduced - identity)

        return error < tol

    def is_cptp(self, tol: float = 1e-8) -> bool:
        """Проверка полных CPTP условий"""
        return self.is_completely_positive(tol) and self.is_trace_preserving(tol)

    def kraus_rank(self) -> int:
        """
        Ранг Крауса = ранг Choi matrix

        Минимальное число операторов Крауса необходимых для представления канала

        Returns:
            Ранг (число ненулевых собственных значений)
        """
        eigenvalues = np.linalg.eigvalsh(self.choi_matrix)
        rank = np.sum(eigenvalues > 1e-10)
        return rank

    def get_kraus_operators(self) -> List[NDArray[np.complex128]]:
        """
        Извлечь операторы Крауса из Choi matrix

        Используем спектральное разложение:
        J = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|
        Kᵢ = √λᵢ reshape(|vᵢ⟩, (d, d))

        Returns:
            Список операторов Крауса {Kᵢ}
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.choi_matrix)

        # Оставляем только положительные собственные значения
        threshold = 1e-12
        positive_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:, positive_indices]

        # Формируем операторы Крауса
        kraus_operators = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]
            # Reshape: vec (d²) → K (d×d)
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            kraus_operators.append(K)

        return kraus_operators

    def eigenvalue_spectrum(self) -> NDArray[np.float64]:
        """
        Спектр собственных значений Choi matrix

        Важно для:
        - Определения ранга канала
        - Анализа структуры шума
        - Оценки близости к унитарному каналу

        Returns:
            Отсортированные собственные значения (по убыванию)
        """
        eigenvalues = np.linalg.eigvalsh(self.choi_matrix)
        return np.sort(eigenvalues)[::-1]

    def purity(self) -> float:
        """
        "Чистота" канала: Tr(J²) / d

        = 1 для унитарных каналов (ранг 1)
        = 1/d для maximally mixed канала

        Returns:
            Чистота в [1/d, 1]
        """
        trace_j_squared = np.trace(self.choi_matrix @ self.choi_matrix).real
        return trace_j_squared / self.dim

    def visualize_real_imag(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Разделить Choi matrix на действительную и мнимую части
        Полезно для визуализации

        Returns:
            (real_part, imag_part)
        """
        real_part = self.choi_matrix.real
        imag_part = self.choi_matrix.imag

        return real_part, imag_part

    def distance_to_identity(self) -> float:
        """
        Расстояние до identity канала (нет шума)

        Измеряет насколько канал отличается от ε(ρ) = ρ

        Returns:
            Frobenius norm: ||J - J_id||_F
        """
        # Choi matrix identity канала
        max_entangled = self._max_entangled_state()
        J_identity = max_entangled @ max_entangled.conj().T

        distance = np.linalg.norm(self.choi_matrix - J_identity, ord='fro')
        return distance

    def _max_entangled_state(self) -> NDArray[np.complex128]:
        """
        Максимально запутанное состояние |Φ⁺⟩ = Σᵢ|i⟩|i⟩/√d

        Returns:
            Вектор состояния размера d²
        """
        # Создаём |Φ⁺⟩ в векторной форме
        phi_plus = np.zeros(self.dim ** 2, dtype=np.complex128)

        for i in range(self.dim):
            # |i⟩|i⟩ в тензорном произведении
            phi_plus[i * self.dim + i] = 1.0

        phi_plus /= np.sqrt(self.dim)

        return phi_plus

    @classmethod
    def from_channel(cls, channel) -> 'ChoiRepresentation':
        """
        Создать из квантового канала

        Args:
            channel: QuantumChannel объект (1 кубит)

        Returns:
            ChoiRepresentation
        """
        choi = channel.get_choi_matrix()
        return cls(choi)

    @classmethod
    def from_kraus(cls, kraus_operators: List[NDArray[np.complex128]]) -> 'ChoiRepresentation':
        """
        Создать из операторов Крауса

        J = Σᵢ vec(Kᵢ) vec(Kᵢ)†

        Args:
            kraus_operators: Список операторов Крауса (2×2 матрицы)

        Returns:
            ChoiRepresentation
        """
        choi = np.zeros((4, 4), dtype=np.complex128)

        for K in kraus_operators:
            # Векторизация K
            vec_K = K.reshape(-1, 1)
            choi += vec_K @ vec_K.conj().T

        return cls(choi)

    def __repr__(self) -> str:
        cptp_str = "CPTP" if self.is_cptp() else "non-CPTP"
        return (f"ChoiRepresentation(n_qubits={self.n_qubits}, "
                f"rank={self.kraus_rank()}, {cptp_str})")
