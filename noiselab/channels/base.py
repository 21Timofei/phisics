"""
Базовый класс для квантовых каналов
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import DensityMatrix


class QuantumChannel(ABC):
    """
    Абстрактный базовый класс для квантового канала

    Квантовый канал - это линейное CPTP (Completely Positive, Trace-Preserving)
    отображение матриц плотности: ε: ρ → ρ'

    Представления:
    1. Операторы Крауса: ε(ρ) = Σᵢ Kᵢ ρ Kᵢ†, где Σᵢ Kᵢ†Kᵢ = I
    2. Матрица Чои: J(ε) = (I ⊗ ε)(|Φ⁺⟩⟨Φ⁺|)
    3. Матрица переноса Паули (PTM): в базисе Паули
    """

    def __init__(self, n_qubits: int, name: str = "Channel"):
        """
        Args:
            n_qubits: Число кубитов
            name: Название канала
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.name = name

    @abstractmethod
    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """
        Применить канал к состоянию

        Args:
            rho: Входное состояние

        Returns:
            Выходное состояние ε(ρ)
        """
        pass

    @abstractmethod
    def get_kraus_operators(self) -> List[NDArray[np.complex128]]:
        """
        Получить операторы Крауса канала

        Returns:
            Список операторов Крауса {Kᵢ}
        """
        pass

    def validate_cptp(self, tol: float = 1e-10) -> bool:
        """
        Проверка CPTP условий:
        1. Trace-preserving: Σᵢ Kᵢ†Kᵢ = I
        2. Completely positive: Choi matrix ≥ 0

        Returns:
            True если канал CPTP
        """
        kraus_ops = self.get_kraus_operators()

        # Проверка trace-preserving
        sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
        identity = np.eye(self.dim, dtype=np.complex128)

        if not np.allclose(sum_kraus, identity, atol=tol):
            print(f"Канал не trace-preserving: ||ΣKᵢ†Kᵢ - I|| = {np.linalg.norm(sum_kraus - identity)}")
            return False

        # Проверка completely positive через Choi matrix
        choi = self.get_choi_matrix()
        eigenvalues = np.linalg.eigvalsh(choi)

        if np.any(eigenvalues < -tol):
            print(f"Choi matrix не положительно полуопределённая: min eigenvalue = {eigenvalues.min()}")
            return False

        return True

    def get_choi_matrix(self) -> NDArray[np.complex128]:
        """
        Вычислить матрицу Чои (Choi-Jamiolkowski representation)

        J(ε) = (I ⊗ ε)(|Φ⁺⟩⟨Φ⁺|)
        где |Φ⁺⟩ = Σᵢ|i⟩|i⟩/√d - максимально запутанное состояние

        Альтернативная формула через операторы Крауса:
        J(ε) = Σᵢ vec(Kᵢ) vec(Kᵢ)†
        где vec - векторизация матрицы

        Returns:
            Choi matrix размера (d² × d²)
        """
        kraus_ops = self.get_kraus_operators()

        # Используем формулу через операторы Крауса
        choi_dim = self.dim ** 2
        choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        for K in kraus_ops:
            # Векторизация: vec(K) - столбцовая развёртка матрицы K
            vec_K = K.reshape(-1, 1)
            choi += vec_K @ vec_K.conj().T

        return choi

    def compose(self, other: 'QuantumChannel') -> 'QuantumChannel':
        """
        Композиция каналов: ε₁ ∘ ε₂ (сначала ε₂, потом ε₁)

        Результат - новый канал с операторами Крауса:
        {K^(1)_i K^(2)_j} для всех пар i, j
        """
        from .kraus import KrausChannel

        if self.n_qubits != other.n_qubits:
            raise ValueError("Каналы должны иметь одинаковое число кубитов")

        kraus1 = self.get_kraus_operators()
        kraus2 = other.get_kraus_operators()

        # Композиция: все произведения K₁ᵢ @ K₂ⱼ
        composed_kraus = [
            K1 @ K2
            for K1 in kraus1
            for K2 in kraus2
        ]

        return KrausChannel(
            composed_kraus,
            name=f"{self.name}∘{other.name}"
        )

    def tensor(self, other: 'QuantumChannel') -> 'QuantumChannel':
        """
        Тензорное произведение каналов: ε₁ ⊗ ε₂

        Результат - канал на (n₁ + n₂) кубитах с операторами:
        {K^(1)_i ⊗ K^(2)_j}
        """
        from .kraus import KrausChannel

        kraus1 = self.get_kraus_operators()
        kraus2 = other.get_kraus_operators()

        # Тензорное произведение всех пар
        tensor_kraus = [
            np.kron(K1, K2)
            for K1 in kraus1
            for K2 in kraus2
        ]

        return KrausChannel(
            tensor_kraus,
            n_qubits=self.n_qubits + other.n_qubits,
            name=f"{self.name}⊗{other.name}"
        )

    def __call__(self, rho: DensityMatrix) -> DensityMatrix:
        """Позволяет использовать channel(rho) вместо channel.apply(rho)"""
        return self.apply(rho)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', n_qubits={self.n_qubits})"
