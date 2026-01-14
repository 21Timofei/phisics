"""
Реализация квантового канала через операторы Крауса
"""

import numpy as np
from typing import List, Optional
from numpy.typing import NDArray
from .base import QuantumChannel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import DensityMatrix


class KrausChannel(QuantumChannel):
    """
    Квантовый канал, заданный через операторы Крауса

    ε(ρ) = Σᵢ Kᵢ ρ Kᵢ†

    Условие CPTP: Σᵢ Kᵢ†Kᵢ = I (trace-preserving)
    """

    def __init__(self, kraus_operators: List[NDArray[np.complex128]],
                 n_qubits: Optional[int] = None,
                 name: str = "KrausChannel",
                 validate: bool = True):
        """
        Args:
            kraus_operators: Список операторов Крауса {Kᵢ}
            n_qubits: Число кубитов (определяется автоматически если None)
            name: Название канала
            validate: Проверять ли CPTP условия
        """
        self.kraus_operators = [np.array(K, dtype=np.complex128) for K in kraus_operators]

        if len(self.kraus_operators) == 0:
            raise ValueError("Нужен хотя бы один оператор Крауса")

        # Определение числа кубитов из размерности
        dim = self.kraus_operators[0].shape[0]
        if n_qubits is None:
            if not self._is_power_of_two(dim):
                raise ValueError(f"Размерность {dim} не является степенью 2")
            n_qubits = int(np.log2(dim))

        super().__init__(n_qubits, name)

        # Проверка размерностей
        for i, K in enumerate(self.kraus_operators):
            if K.shape != (self.dim, self.dim):
                raise ValueError(f"Оператор Крауса {i} имеет неправильную размерность: "
                               f"{K.shape} вместо ({self.dim}, {self.dim})")

        if validate:
            if not self.validate_cptp():
                raise ValueError("Операторы Крауса не удовлетворяют CPTP условиям")

    @staticmethod
    def _is_power_of_two(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    def apply(self, rho: DensityMatrix) -> DensityMatrix:
        """
        Применить канал к состоянию: ε(ρ) = Σᵢ Kᵢ ρ Kᵢ†
        """
        if rho.n_qubits != self.n_qubits:
            raise ValueError(f"Несоответствие размерностей: канал {self.n_qubits} кубитов, "
                           f"состояние {rho.n_qubits} кубитов")

        # Применяем операторы Крауса
        new_rho = np.zeros((self.dim, self.dim), dtype=np.complex128)

        for K in self.kraus_operators:
            new_rho += K @ rho.matrix @ K.conj().T

        return DensityMatrix(new_rho, validate=False)

    def get_kraus_operators(self) -> List[NDArray[np.complex128]]:
        """Получить операторы Крауса"""
        return self.kraus_operators.copy()

    def kraus_rank(self) -> int:
        """
        Ранг Крауса - минимальное число операторов, необходимых для представления канала
        Совпадает с рангом Choi matrix
        """
        choi = self.get_choi_matrix()
        eigenvalues = np.linalg.eigvalsh(choi)
        rank = np.sum(eigenvalues > 1e-10)
        return rank

    def minimize_kraus_rank(self) -> 'KrausChannel':
        """
        Минимизировать число операторов Крауса через диагонализацию Choi matrix

        J(ε) = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|
        Kᵢ = √λᵢ reshape(|vᵢ⟩, (d, d))
        """
        choi = self.get_choi_matrix()

        # Диагонализация
        eigenvalues, eigenvectors = np.linalg.eigh(choi)

        # Оставляем только ненулевые собственные значения
        threshold = 1e-12
        nonzero_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[nonzero_indices]
        eigenvectors = eigenvectors[:, nonzero_indices]

        # Создаём минимальный набор операторов Крауса
        minimal_kraus = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]
            # Reshape вектора обратно в матрицу
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            minimal_kraus.append(K)

        return KrausChannel(
            minimal_kraus,
            n_qubits=self.n_qubits,
            name=f"{self.name}_minimal",
            validate=False
        )

    @classmethod
    def from_unitary(cls, unitary: NDArray[np.complex128],
                     probability: float = 1.0,
                     name: str = "UnitaryChannel") -> 'KrausChannel':
        """
        Создать канал из унитарной эволюции

        ε(ρ) = p U ρ U† + (1-p) ρ

        Args:
            unitary: Унитарная матрица
            probability: Вероятность применения унитарного гейта
            name: Название канала
        """
        dim = len(unitary)

        if probability == 1.0:
            # Чисто унитарная эволюция
            kraus_ops = [unitary]
        else:
            # Смесь унитарной эволюции и identity
            kraus_ops = [
                np.sqrt(probability) * unitary,
                np.sqrt(1 - probability) * np.eye(dim, dtype=np.complex128)
            ]

        return cls(kraus_ops, name=name, validate=False)
