"""
Регуляризованные методы реконструкции каналов
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import DensityMatrix
from .reconstruction import LinearInversion


class RegularizedReconstruction:
    """
    Регуляризованные методы реконструкции каналов

    Используются когда:
    - Мало экспериментальных данных
    - Система уравнений плохо обусловлена
    - Нужно продвигать определённые свойства решения (спарсность, низкий ранг)

    Методы:
    1. Tikhonov (L2) - минимизирует ||J||_F² (сглаживание)
    2. Maximum Entropy - максимизирует S(J) (максимальная случайность)
    3. Compressed Sensing (L1) - минимизирует ||J||_1 (спарсность)
    """

    def __init__(self, n_qubits: int):
        """
        Args:
            n_qubits: Число кубитов
        """
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def reconstruct_tikhonov(self,
                            input_states: List[DensityMatrix],
                            output_states: List[DensityMatrix],
                            lambda_reg: float = 0.01) -> NDArray[np.complex128]:
        """
        Tikhonov регуляризация (L2)

        Минимизируем: ||A·vec(J) - b||² + λ||vec(J)||²

        Аналитическое решение: vec(J) = (A^H·A + λI)^{-1} A^H·b

        Args:
            input_states: Входные состояния
            output_states: Выходные состояния
            lambda_reg: Параметр регуляризации

        Returns:
            Choi matrix
        """
        if len(input_states) != len(output_states):
            raise ValueError("Число входных и выходных состояний должно совпадать")

        choi_dim = self.dim ** 2

        # Строим систему уравнений (как в LinearInversion)
        equations = []
        rhs = []

        for rho_in, rho_out in zip(input_states, output_states):
            for i in range(self.dim):
                for j in range(self.dim):
                    equation = np.zeros(choi_dim * choi_dim, dtype=np.complex128)

                    for k in range(self.dim):
                        for l in range(self.dim):
                            row_idx = i * self.dim + k
                            col_idx = j * self.dim + l
                            vec_idx = row_idx * choi_dim + col_idx

                            equation[vec_idx] = rho_in.matrix[k, l]

                    equations.append(equation)
                    rhs.append(rho_out.matrix[i, j])

        A = np.array(equations)
        b = np.array(rhs)

        # Tikhonov решение: (A^H·A + λI)^{-1} A^H·b
        AH_A = A.conj().T @ A
        AH_b = A.conj().T @ b

        regularization_matrix = lambda_reg * np.eye(choi_dim * choi_dim, dtype=np.complex128)

        choi_vec = np.linalg.solve(AH_A + regularization_matrix, AH_b)
        choi = choi_vec.reshape((choi_dim, choi_dim))

        # Постобработка
        choi = self._make_hermitian(choi)

        # Нормализация: Tr(J) = d
        current_trace = np.trace(choi).real
        if abs(current_trace) > 1e-10:
            choi = choi * (self.dim / current_trace)

        # Проекция на CPTP
        lin_inv = LinearInversion(self.n_qubits)
        choi = lin_inv._project_to_cptp(choi)

        return choi

    def reconstruct_max_entropy(self,
                               input_states: List[DensityMatrix],
                               output_states: List[DensityMatrix],
                               max_iterations: int = 100,
                               tolerance: float = 1e-6) -> NDArray[np.complex128]:
        """
        Maximum Entropy реконструкция

        Максимизируем энтропию фон Неймана: S(J) = -Tr(J·log(J))

        При ограничениях:
        - Согласованность с данными: ε(ρᵢ) = σᵢ
        - CPTP условия

        Используем итеративный градиентный подъём с проекцией

        Args:
            input_states: Входные состояния
            output_states: Выходные состояния
            max_iterations: Максимальное число итераций
            tolerance: Порог сходимости

        Returns:
            Choi matrix
        """
        choi_dim = self.dim ** 2

        # Начальное приближение: максимально смешанное состояние
        choi = np.eye(choi_dim, dtype=np.complex128) / choi_dim
        choi = choi * self.dim  # Нормализация Tr(J) = d

        lin_inv = LinearInversion(self.n_qubits)
        learning_rate = 0.1

        for iteration in range(max_iterations):
            # 1. Вычисляем градиент энтропии
            # ∇S(J) = -(log(J) + I)
            eigenvalues, eigenvectors = np.linalg.eigh(choi)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Избегаем log(0)

            log_eigenvalues = np.log(eigenvalues)
            log_choi = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.conj().T

            gradient = -(log_choi + np.eye(choi_dim, dtype=np.complex128))

            # 2. Градиентный шаг (подъём для максимизации)
            choi_new = choi + learning_rate * gradient

            # 3. Проекция на согласованность с данными
            # Корректируем чтобы удовлетворять ε(ρᵢ) ≈ σᵢ
            for rho_in, rho_out in zip(input_states, output_states):
                # Вычисляем текущий выход
                current_output = self._apply_choi(choi_new, rho_in.matrix)

                # Вычисляем ошибку
                error = rho_out.matrix - current_output

                # Коррекция (упрощённая)
                # В идеале нужен метод множителей Лагранжа
                correction_weight = 0.5
                choi_new = choi_new + correction_weight * self._build_choi_correction(
                    rho_in.matrix, error
                )

            # 4. Проекция на CPTP
            choi_new = self._make_hermitian(choi_new)
            choi_new = lin_inv._project_to_cptp(choi_new)

            # Проверка сходимости
            change = np.linalg.norm(choi_new - choi)
            choi = choi_new

            if change < tolerance:
                break

        return choi

    def reconstruct_compressed_sensing(self,
                                       input_states: List[DensityMatrix],
                                       output_states: List[DensityMatrix],
                                       lambda_reg: float = 0.01) -> NDArray[np.complex128]:
        """
        Compressed Sensing (L1 регуляризация)

        Минимизируем: ||A·vec(J) - b||² + λ||vec(J)||₁

        Продвигает спарсные решения (низкий ранг канала)

        Требует CVXPY

        Args:
            input_states: Входные состояния
            output_states: Выходные состояния
            lambda_reg: Параметр регуляризации

        Returns:
            Choi matrix
        """
        try:
            import cvxpy as cp
        except ImportError:
            print("CVXPY не установлен. Используем Tikhonov регуляризацию вместо L1.")
            return self.reconstruct_tikhonov(input_states, output_states, lambda_reg)

        if len(input_states) != len(output_states):
            raise ValueError("Число входных и выходных состояний должно совпадать")

        choi_dim = self.dim ** 2

        # Переменные (блочное представление для комплексных чисел)
        J_real = cp.Variable((choi_dim, choi_dim), symmetric=True)
        J_imag = cp.Variable((choi_dim, choi_dim))

        constraints = []

        # Антисимметричность J_imag
        constraints.append(J_imag == -J_imag.T)

        # Положительная полуопределённость
        J_block = cp.bmat([
            [J_real, -J_imag],
            [J_imag,  J_real]
        ])
        constraints.append(J_block >> 0)

        # Trace-preserving
        for i in range(self.dim):
            for k in range(self.dim):
                trace_sum_real = sum(J_real[i*self.dim + j, k*self.dim + j]
                                    for j in range(self.dim))
                trace_sum_imag = sum(J_imag[i*self.dim + j, k*self.dim + j]
                                    for j in range(self.dim))

                if i == k:
                    constraints.append(trace_sum_real == 1.0)
                else:
                    constraints.append(trace_sum_real == 0.0)

                constraints.append(trace_sum_imag == 0.0)

        # Строим уравнения согласованности с данными
        residuals = []

        for rho_in, rho_out in zip(input_states, output_states):
            for i in range(self.dim):
                for j in range(self.dim):
                    # Коэффициенты связи
                    coeffs = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

                    for k in range(self.dim):
                        for l in range(self.dim):
                            row_idx = i * self.dim + k
                            col_idx = j * self.dim + l
                            coeffs[row_idx, col_idx] = rho_in.matrix[k, l]

                    coeffs_real = np.real(coeffs)
                    coeffs_imag = np.imag(coeffs)

                    # Модель
                    model_val = cp.sum(cp.multiply(coeffs_real, J_real)) + \
                               cp.sum(cp.multiply(coeffs_imag, J_imag))

                    # Данные
                    data_val_real = np.real(rho_out.matrix[i, j])
                    data_val_imag = np.imag(rho_out.matrix[i, j])

                    residuals.append(model_val - data_val_real)
                    # Мнимая часть (если нужна)
                    # residuals.append(... - data_val_imag)

        # Целевая функция: ||residuals||² + λ||J||₁
        data_fidelity = cp.sum_squares(cp.hstack(residuals)) if residuals else 0
        l1_penalty = lambda_reg * (cp.norm(J_real, 1) + cp.norm(J_imag, 1))

        objective = cp.Minimize(data_fidelity + l1_penalty)

        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"L1 решение не оптимально (статус: {problem.status})")
                print("Используем Tikhonov регуляризацию")
                return self.reconstruct_tikhonov(input_states, output_states, lambda_reg)

            choi = J_real.value + 1j * J_imag.value

        except Exception as e:
            print(f"Ошибка при L1 оптимизации: {e}")
            return self.reconstruct_tikhonov(input_states, output_states, lambda_reg)

        # Постобработка
        choi = self._make_hermitian(choi)
        choi = LinearInversion._project_to_positive(choi)

        return choi

    def select_regularization_parameter(self,
                                        input_states: List[DensityMatrix],
                                        output_states: List[DensityMatrix],
                                        method: str = 'tikhonov',
                                        lambda_range: Optional[List[float]] = None,
                                        k_folds: int = 5) -> float:
        """
        Выбор оптимального параметра регуляризации через кросс-валидацию

        Разбиваем данные на k частей, обучаем на k-1, тестируем на 1
        Выбираем λ с минимальной средней ошибкой

        Args:
            input_states: Входные состояния
            output_states: Выходные состояния
            method: 'tikhonov' или 'l1'
            lambda_range: Список значений λ для проверки
            k_folds: Число фолдов для кросс-валидации

        Returns:
            Оптимальное значение λ
        """
        if lambda_range is None:
            # Логарифмическая сетка
            lambda_range = [10**i for i in range(-4, 1)]  # [0.0001, 0.001, 0.01, 0.1, 1.0]

        n_states = len(input_states)

        if n_states < k_folds:
            k_folds = n_states

        # Случайная перестановка индексов
        indices = np.random.permutation(n_states)
        fold_size = n_states // k_folds

        best_lambda = lambda_range[0]
        best_error = float('inf')

        for lambda_val in lambda_range:
            errors = []

            for fold in range(k_folds):
                # Разбиваем на train/test
                test_start = fold * fold_size
                test_end = test_start + fold_size if fold < k_folds - 1 else n_states

                test_indices = indices[test_start:test_end]
                train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

                train_input = [input_states[i] for i in train_indices]
                train_output = [output_states[i] for i in train_indices]
                test_input = [input_states[i] for i in test_indices]
                test_output = [output_states[i] for i in test_indices]

                # Обучаем с данным λ
                if method == 'tikhonov':
                    choi = self.reconstruct_tikhonov(train_input, train_output, lambda_val)
                elif method == 'l1':
                    choi = self.reconstruct_compressed_sensing(train_input, train_output, lambda_val)
                else:
                    raise ValueError(f"Неизвестный метод: {method}")

                # Оцениваем на тесте
                fold_error = 0.0
                for rho_in, rho_out in zip(test_input, test_output):
                    predicted = self._apply_choi(choi, rho_in.matrix)
                    fold_error += np.linalg.norm(predicted - rho_out.matrix, 'fro') ** 2

                errors.append(fold_error)

            # Средняя ошибка по всем фолдам
            mean_error = np.mean(errors)

            if mean_error < best_error:
                best_error = mean_error
                best_lambda = lambda_val

        return best_lambda

    def _apply_choi(self, choi: NDArray[np.complex128], rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Применить канал (через Choi matrix) к состоянию

        ε(ρ) = Tr_A[(ρ^T ⊗ I) J]

        Args:
            choi: Choi matrix
            rho: Входное состояние

        Returns:
            Выходное состояние
        """
        # Простая формула через элементы
        # (ε(ρ))_{ij} = Σ_{kl} J_{ik,jl} * ρ_{kl}

        output = np.zeros((self.dim, self.dim), dtype=np.complex128)

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        row_idx = i * self.dim + k
                        col_idx = j * self.dim + l
                        output[i, j] += choi[row_idx, col_idx] * rho[k, l]

        return output

    def _build_choi_correction(self,
                               rho: NDArray[np.complex128],
                               error: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Построить коррекцию Choi matrix для уменьшения ошибки

        Если ε(ρ) - σ = error, то строим ΔJ такую что ε_ΔJ(ρ) ≈ error

        Args:
            rho: Входное состояние
            error: Ошибка выхода

        Returns:
            Коррекция ΔJ
        """
        choi_dim = self.dim ** 2
        delta_choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        # Обратная задача: находим ΔJ из уравнения
        # Σ_{kl} ΔJ_{ik,jl} * ρ_{kl} = error_{ij}

        # Упрощённое решение: используем псевдоинверсию
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        row_idx = i * self.dim + k
                        col_idx = j * self.dim + l

                        # Простая аппроксимация
                        if abs(rho[k, l]) > 1e-10:
                            delta_choi[row_idx, col_idx] += error[i, j] * rho[k, l].conj() / (
                                np.linalg.norm(rho, 'fro') ** 2 + 1e-10
                            )

        return delta_choi

    @staticmethod
    def _make_hermitian(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Сделать матрицу эрмитовой"""
        return (matrix + matrix.conj().T) / 2
