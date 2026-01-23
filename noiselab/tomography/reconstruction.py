"""
Алгоритмы реконструкции каналов из томографических данных
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import DensityMatrix
from channels.kraus import KrausChannel


class LinearInversion:
    """
    Линейная инверсия (Least Squares) для QPT

    Решаем систему линейных уравнений:
    Eᵢⱼ = Tr(Mⱼ ε(ρᵢ))

    где:
    - Eᵢⱼ - экспериментальные данные (средние значения измерений)
    - Mⱼ - измерительные операторы
    - ρᵢ - входные состояния
    - ε - неизвестный канал

    Представляем канал в базисе Паули (PTM) или через Choi matrix
    """

    def __init__(self):
        self.n_qubits = 1
        self.dim = 2

    def reconstruct_choi(self,
                        input_states: List[DensityMatrix],
                        output_states: List[DensityMatrix]) -> NDArray[np.complex128]:
        """
        Восстановить Choi matrix из экспериментальных данных

        Правильная формула через операторную базу:
        Для базисных матриц {Eᵢⱼ = |i⟩⟨j|} измеряем ε(Eᵢⱼ)
        Тогда Choi matrix: J = Σᵢⱼ |i⟩⟨j| ⊗ ε(|i⟩⟨j|)

        Для переполненного набора используем псевдоинверсию.

        Args:
            input_states: Входные состояния {ρᵢ}
            output_states: Выходные состояния {ε(ρᵢ)} (экспериментальные)

        Returns:
            Choi matrix размера d²×d²
        """
        if len(input_states) != len(output_states):
            raise ValueError("Число входных и выходных состояний должно совпадать")

        choi_dim = self.dim ** 2

        # ПРАВИЛЬНАЯ линейная инверсия через partial trace
        #
        # Связь: ε(ρ) = Tr_in[(ρ^T ⊗ I) J]
        # В элементном виде: (ε(ρ))_{ij} = Σ_{k,l} J_{ik,jl} · ρ_{kl}
        #
        # Строим систему линейных уравнений для элементов Choi matrix

        n_states = len(input_states)

        # Для каждой пары (ρ_in, ρ_out) получаем d² уравнений
        # Всего d⁴ неизвестных (элементы Choi matrix)
        equations = []
        rhs = []

        for rho_in, rho_out in zip(input_states, output_states):
            # Для каждого элемента выходной матрицы (i,j)
            for i in range(self.dim):
                for j in range(self.dim):
                    # Строим уравнение: (ε(ρ))_{ij} = Σ_{k,l} J_{ik,jl} · ρ_{kl}
                    # Вектор коэффициентов для этого уравнения
                    equation = np.zeros(choi_dim * choi_dim, dtype=np.complex128)

                    for k in range(self.dim):
                        for l in range(self.dim):
                            # Индекс элемента J_{ik,jl} в векторе vec(J)
                            # J имеет индексацию [i*d+k, j*d+l]
                            row_idx = i * self.dim + k
                            col_idx = j * self.dim + l
                            vec_idx = row_idx * choi_dim + col_idx

                            # Коэффициент - это элемент входной матрицы
                            equation[vec_idx] = rho_in.matrix[k, l]

                    equations.append(equation)
                    rhs.append(rho_out.matrix[i, j])

        # Собираем систему
        A = np.array(equations)
        b = np.array(rhs)

        # Решаем через LSQ
        choi_vec, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        choi = choi_vec.reshape((choi_dim, choi_dim))

        # Нормализация: Tr(J) = d
        current_trace = np.trace(choi).real
        if abs(current_trace) > 1e-10:
            choi = choi * (self.dim / current_trace)

        # Choi matrix может не быть физичной (не CPTP)
        # Применяем коррекции
        choi = self._make_hermitian(choi)

        return choi

    def reconstruct_ptm(self,
                       input_states: List[DensityMatrix],
                       measurement_results: Dict[int, Dict[str, float]]) -> NDArray[np.float64]:
        """
        Восстановить Pauli Transfer Matrix (PTM)

        PTM описывает действие канала в базисе Паули:
        r' = R·r

        где r - вектор разложения ρ по паули-базису

        Args:
            input_states: Входные состояния
            measurement_results: Результаты паули-измерений {state_idx: {basis: expectation}}

        Returns:
            PTM матрица размера 4×4
        """
        from .state_prep import get_pauli_basis, decompose_in_pauli_basis

        pauli_basis = get_pauli_basis()

        # Матрица PTM
        ptm = np.zeros((4, 4), dtype=np.float64)

        # Для каждого входного состояния
        for state_idx, rho_in in enumerate(input_states):
            # Разложение входного состояния в базисе Паули
            r_in = decompose_in_pauli_basis(rho_in.matrix)

            # Для каждого паули-измерения
            if state_idx in measurement_results:
                expectations = measurement_results[state_idx]

                # Собираем вектор выходного состояния
                # (упрощённая версия, полная требует всех паули-измерений)
                # TODO: реализовать полную реконструкцию через все базисы
                pass

        return ptm

    @staticmethod
    def _make_hermitian(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Сделать матрицу эрмитовой"""
        return (matrix + matrix.conj().T) / 2

    @staticmethod
    def _project_to_positive(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Проецировать матрицу на конус положительно полуопределённых матриц
        Убираем отрицательные собственные значения
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 0)  # Обнуляем отрицательные
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T

    def _project_to_cptp(self, choi: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Проецировать Choi matrix на множество CPTP каналов

        Два условия:
        1. Completely Positive: J ≥ 0
        2. Trace-Preserving: Tr_B(J) = I

        Используем простой итеративный алгоритм
        """
        from ..core.tensor import partial_trace

        choi = self._make_hermitian(choi)

        # Итеративная проекция
        max_iterations = 50
        tol = 1e-6

        for iteration in range(max_iterations):
            # 1. Проекция на положительную полуопределённость
            choi = self._project_to_positive(choi)

            # 2. Проекция на trace-preserving: Tr_B(J) = I
            dims = [self.dim, self.dim]

            try:
                reduced = partial_trace(choi, dims, 1)
            except:
                # Если partial_trace не работает, пропускаем
                break

            identity = np.eye(self.dim, dtype=np.complex128)

            # Простая коррекция: масштабируем чтобы получить правильный след
            trace_factor = np.trace(reduced).real / self.dim
            if trace_factor > 1e-10:
                choi = choi / trace_factor

            # Проверка сходимости
            try:
                reduced_new = partial_trace(choi, dims, 1)
                error = np.linalg.norm(reduced_new - identity)

                if error < tol:
                    break
            except:
                break

        # Финальная проекция на положительность
        choi = self._project_to_positive(choi)
        choi = self._make_hermitian(choi)

        return choi
