"""
Pauli Transfer Matrix (PTM) representation
Представление канала в базисе Паули
"""

import numpy as np
from typing import List
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tomography.state_prep import get_pauli_basis


class PauliTransferMatrix:
    """
    Pauli Transfer Matrix (PTM) - представление канала в базисе Паули

    Любой оператор можно разложить по базису Паули:
    ρ = Σᵢ rᵢ σᵢ / 2^n

    Канал действует линейно на коэффициенты:
    r' = R · r

    где R - PTM матрица размера (4^n × 4^n)

    Преимущества PTM:
    - Действительная матрица (в отличие от Choi)
    - Удобно для анализа марковских процессов
    - Собственные значения характеризуют декогеренцию

    Связь с Choi matrix:
    R_ij = Tr(σᵢ ε(σⱼ)) / 2^n
    """

    def __init__(self, ptm_matrix: NDArray[np.float64]):
        """
        Args:
            ptm_matrix: PTM матрица размера 4×4 (для 1 кубита)
        """
        self.n_qubits = 1
        self.dim = 2
        self.ptm_matrix = ptm_matrix

        if ptm_matrix.shape != (4, 4):
            raise ValueError(f"Неправильная размерность PTM: {ptm_matrix.shape}, ожидается (4, 4)")

    def is_trace_preserving(self, tol: float = 1e-10) -> bool:
        """
        Проверка trace-preserving

        Первая строка PTM должна быть [1, 0, 0, ..., 0]
        (identity компонента сохраняется)
        """
        first_row = self.ptm_matrix[0, :]
        expected = np.zeros(len(first_row))
        expected[0] = 1.0

        error = np.linalg.norm(first_row - expected)
        return error < tol

    def is_completely_positive(self) -> bool:
        """
        Проверка completely positive через Choi matrix
        PTM сама по себе не даёт прямого критерия CP
        """
        choi = self.to_choi()
        eigenvalues = np.linalg.eigvalsh(choi)
        return np.all(eigenvalues >= -1e-10)

    def eigenvalues(self) -> NDArray[np.complex128]:
        """
        Собственные значения PTM

        Характеризуют скорость декогеренции по разным направлениям

        Для унитарных каналов: все |λᵢ| = 1
        Для деполяризации: λᵢ → 0

        Returns:
            Массив собственных значений
        """
        return np.linalg.eigvals(self.ptm_matrix)

    def decay_rates(self) -> NDArray[np.float64]:
        """
        Скорости затухания (decay rates)

        Для марковского процесса: ε_t(ρ) = exp(tL)(ρ)
        PTM: R(t) = exp(tM)

        Собственные значения M дают скорости затухания

        Returns:
            Действительные части логарифмов собственных значений
        """
        eigenvals = self.eigenvalues()

        # log(λ) даёт скорость (для λ близких к 1)
        decay = -np.log(np.abs(eigenvals))

        return np.sort(decay)

    def apply_to_pauli_vector(self, pauli_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Применить канал к вектору разложения Паули

        r' = R · r

        Args:
            pauli_vec: Коэффициенты разложения оператора по базису Паули

        Returns:
            Преобразованный вектор
        """
        return self.ptm_matrix @ pauli_vec

    def to_choi(self) -> NDArray[np.complex128]:
        """
        Преобразовать PTM в Choi matrix

        Используем связь через базис Паули:
        J = Σᵢⱼ R_ij (σᵢ ⊗ σⱼ) / 2^n

        Returns:
            Choi matrix
        """
        pauli_basis = get_pauli_basis(self.n_qubits)
        choi_dim = self.dim ** 2

        choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        n_paulis = len(pauli_basis)
        for i in range(n_paulis):
            for j in range(n_paulis):
                # σᵢ ⊗ σⱼ
                tensor_prod = np.kron(pauli_basis[i], pauli_basis[j])
                choi += self.ptm_matrix[i, j] * tensor_prod

        choi /= self.dim

        return choi

    def visualize_heatmap_data(self) -> NDArray[np.float64]:
        """
        Подготовить данные для heatmap визуализации

        Returns:
            PTM matrix (действительная)
        """
        return self.ptm_matrix

    @classmethod
    def from_choi(cls, choi_matrix: NDArray[np.complex128]) -> 'PauliTransferMatrix':
        """
        Создать PTM из Choi matrix

        R_ij = Tr(σᵢ ε(σⱼ)) / 2

        Args:
            choi_matrix: Choi matrix (4×4)

        Returns:
            PauliTransferMatrix
        """
        pauli_basis = get_pauli_basis()

        ptm = np.zeros((4, 4), dtype=np.float64)

        for i in range(4):
            for j in range(4):
                sigma_i = pauli_basis[i]
                sigma_j = pauli_basis[j]

                # Векторизация
                vec_j = sigma_j.reshape(-1, 1)

                # Действие канала через Choi
                result_vec = choi_matrix @ vec_j
                result = result_vec.reshape(2, 2)

                # След с σᵢ
                ptm[i, j] = np.trace(sigma_i @ result).real / 2

        return cls(ptm)

    @classmethod
    def from_channel(cls, channel) -> 'PauliTransferMatrix':
        """
        Создать PTM из квантового канала

        Args:
            channel: QuantumChannel объект (1 кубит)

        Returns:
            PauliTransferMatrix
        """
        choi = channel.get_choi_matrix()
        return cls.from_choi(choi)

    def __repr__(self) -> str:
        tp_str = "TP" if self.is_trace_preserving() else "non-TP"
        return f"PauliTransferMatrix(n_qubits={self.n_qubits}, {tp_str})"
