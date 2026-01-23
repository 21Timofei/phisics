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
        """
        Линейная инверсия для 1 кубита
        """
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


class MaximumLikelihood:
    """
    Maximum Likelihood Estimation (MLE) для QPT

    Максимизируем функцию правдоподобия:
    L = Π Tr(Mⱼ ε(ρᵢ))^(nᵢⱼ)

    с ограничениями CPTP для канала ε

    Это convex optimization задача, решаемая через SDP (semidefinite programming)
    Используем CVXPY
    """

    def __init__(self):
        """
        Maximum Likelihood для 1 кубита
        """
        self.n_qubits = 1
        self.dim = 2

    def reconstruct(self,
                   input_states: List[DensityMatrix],
                   measurement_data: List[Dict[str, int]],
                   method: str = 'choi') -> KrausChannel:
        """
        MLE реконструкция канала

        Args:
            input_states: Входные состояния
            measurement_data: Данные измерений {label: count} для каждого входа
            method: 'choi' или 'kraus' - параметризация канала

        Returns:
            Реконструированный канал
        """
        try:
            import cvxpy as cp
        except ImportError:
            print("CVXPY не установлен. Используем упрощённую реконструкцию.")
            return self._simplified_reconstruction(input_states, measurement_data)

        if method == 'choi':
            return self._reconstruct_via_choi(input_states, measurement_data)

    def _reconstruct_via_choi(self,
                             input_states: List[DensityMatrix],
                             measurement_data: List[Dict[str, int]]) -> KrausChannel:
        """
        MLE через оптимизацию Choi matrix

        Переменная оптимизации: Choi matrix J
        Ограничения:
        1. J ≥ 0 (SDP constraint)
        2. Tr_B(J) = I (trace-preserving)

        Целевая функция: Weighted Least Squares (приближение MLE)
        Минимизируем: Σᵢⱼ wᵢⱼ (pᵢⱼ^exp - pᵢⱼ^model(J))²

        где pᵢⱼ^exp - экспериментальные вероятности,
            pᵢⱼ^model - вероятности из модели (зависят от J),
            wᵢⱼ - веса (обратная дисперсия)
        """
        import cvxpy as cp

        choi_dim = self.dim ** 2

        # CVXPY работает только с вещественными матрицами для hermitian=True
        # Используем блочное представление: J = J_real + i*J_imag
        # Эрмитова матрица: J_real симметрична, J_imag антисимметрична

        # Создаём две вещественные матрицы
        J_real = cp.Variable((choi_dim, choi_dim), symmetric=True)
        J_imag = cp.Variable((choi_dim, choi_dim))

        # Ограничения
        constraints = []

        # 1. J_imag антисимметрична (для эрмитовости)
        constraints.append(J_imag == -J_imag.T)

        # 2. Положительная полуопределённость (блочная форма)
        # Для комплексной эрмитовой матрицы J = J_real + i*J_imag
        # Условие J >> 0 эквивалентно блочной матрице:
        # [[J_real, -J_imag],
        #  [J_imag,  J_real]] >> 0
        J_block = cp.bmat([
            [J_real, -J_imag],
            [J_imag,  J_real]
        ])
        constraints.append(J_block >> 0)

        # 3. Trace-preserving: Tr_B(J) = I
        # Правильная формула: (Tr_B(J))_{ik} = Σⱼ J_{(i,j),(k,j)}
        # В линейной индексации: Σⱼ J[i*d+j, k*d+j] = δ_{ik}

        for i in range(self.dim):
            for k in range(self.dim):
                # Вычисляем частичный след
                trace_sum_real = sum(J_real[i*self.dim + j, k*self.dim + j]
                                    for j in range(self.dim))

                if i == k:
                    # Диагональный элемент должен быть 1
                    constraints.append(trace_sum_real == 1.0)
                else:
                    # Недиагональные элементы должны быть 0 (реальная часть)
                    constraints.append(trace_sum_real == 0.0)

                # Мнимая часть должна быть 0 всегда (т.к. Tr_B(J) = I вещественна)
                trace_sum_imag = sum(J_imag[i*self.dim + j, k*self.dim + j]
                                    for j in range(self.dim))
                constraints.append(trace_sum_imag == 0.0)

        # 4. Целевая функция: Weighted Least Squares
        # Собираем экспериментальные вероятности и строим модель

        residuals = []

        for state_idx, (rho_in, counts) in enumerate(zip(input_states, measurement_data)):
            total_shots = sum(counts.values())

            if total_shots == 0:
                continue

            # Для каждого результата измерения
            for label, count in counts.items():
                # Экспериментальная вероятность
                p_exp = count / total_shots

                # Вес (обратная дисперсия биномиального распределения)
                # Var(p) = p(1-p)/N
                # Используем p_exp для оценки дисперсии
                variance = p_exp * (1 - p_exp) / total_shots if p_exp > 0 and p_exp < 1 else 1.0 / total_shots
                weight = np.sqrt(1.0 / (variance + 1e-10))  # Избегаем деления на 0

                # Модельная вероятность: p = Tr(M * ε(ρ))
                # где ε(ρ) выражается через Choi matrix
                # ε(ρ) = Tr_A[(ρ^T ⊗ I) J]
                # p = Tr(M * ε(ρ)) = Tr((M^T ⊗ I) * (ρ^T ⊗ I) * J)

                # Получаем POVM элемент для данного измерения
                M = self._get_povm_element(label, state_idx, input_states, measurement_data)

                if M is None:
                    continue

                # Вычисляем коэффициенты для линейной связи p = Σ c_ij * J_ij
                coeffs_real, coeffs_imag = self._build_measurement_coefficients(
                    rho_in.matrix, M
                )

                # Модельная вероятность как линейная комбинация элементов J
                p_model_real = cp.sum(cp.multiply(coeffs_real, J_real))
                p_model_imag = cp.sum(cp.multiply(coeffs_imag, J_imag))
                p_model = p_model_real + p_model_imag  # Должна быть вещественной

                # Добавляем взвешенную невязку
                residuals.append(weight * (p_exp - p_model))

        # Минимизируем сумму квадратов невязок
        if len(residuals) > 0:
            objective = cp.Minimize(cp.sum_squares(cp.hstack(residuals)))
        else:
            # Если нет данных, просто минимизируем норму
            objective = cp.Minimize(cp.norm(J_real, 'fro'))

        # Решаем задачу
        problem = cp.Problem(objective, constraints)

        try:
            problem.solve(solver=cp.SCS, verbose=False, eps=1e-6, max_iters=5000)

            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print(f"Внимание: SDP решение не оптимально (статус: {problem.status})")
                print("Используем упрощённую реконструкцию")
                return self._simplified_reconstruction(input_states, measurement_data)

            # Восстанавливаем комплексную Choi matrix
            choi = J_real.value + 1j * J_imag.value

        except Exception as e:
            print(f"Ошибка при решении SDP: {e}")
            print("Используем упрощённую реконструкцию")
            return self._simplified_reconstruction(input_states, measurement_data)

        # Финальные коррекции для физичности
        choi = self._make_hermitian(choi)
        choi = LinearInversion._project_to_positive(choi)

        # Извлекаем операторы Крауса
        return self._choi_to_kraus(choi)

    def _simplified_reconstruction(self,
                                   input_states: List[DensityMatrix],
                                   measurement_data: List[Dict[str, int]]) -> KrausChannel:
        """
        Упрощённая реконструкция без CVXPY
        Использует линейную инверсию с последующей проекцией на CPTP
        """
        # Оцениваем выходные состояния из данных измерений
        output_states = []

        for counts in measurement_data:
            total = sum(counts.values())

            # Восстанавливаем матрицу плотности из частот
            # (упрощение: предполагаем измерения в вычислительном базисе)
            rho_out = np.zeros((self.dim, self.dim), dtype=np.complex128)

            for label, count in counts.items():
                if label.isdigit() or all(c in '01' for c in label):
                    # Интерпретируем как битовую строку
                    basis_index = int(label, 2) if len(label) > 1 else int(label)
                    if basis_index < self.dim:
                        rho_out[basis_index, basis_index] = count / total
                else:
                    # Для других базисов нужна более сложная логика
                    pass

            output_states.append(DensityMatrix(rho_out, validate=False))

        # Линейная реконструкция
        lin_inv = LinearInversion(self.n_qubits)
        choi = lin_inv.reconstruct_choi(input_states, output_states)

        # Проекция на CPTP
        choi = lin_inv._project_to_cptp(choi)

        # Извлекаем операторы Крауса
        eigenvalues, eigenvectors = np.linalg.eigh(choi)

        # Оставляем положительные собственные значения
        threshold = 1e-10
        positive_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:, positive_indices]

        # Формируем операторы Крауса
        kraus_operators = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            kraus_operators.append(K)

        return KrausChannel(
            kraus_operators,
            n_qubits=1,
            name="MLE_reconstructed",
            validate=False
        )

    def _get_povm_element(self,
                         label: str,
                         state_idx: int,
                         input_states: List[DensityMatrix],
                         measurement_data: List[Dict[str, int]]) -> Optional[NDArray[np.complex128]]:
        """
        Получить POVM элемент для данного результата измерения

        Парсит метку измерения (например, "0", "1", "01", "ZZ", "XY")
        и возвращает соответствующий оператор измерения

        Args:
            label: Метка результата измерения
            state_idx: Индекс входного состояния
            input_states: Все входные состояния
            measurement_data: Все данные измерений

        Returns:
            POVM элемент (матрица размера d×d) или None
        """
        # Определяем базис измерения из метки
        # Поддерживаемые форматы:
        # - "0", "1", ... - вычислительный базис (Z)
        # - "00", "01", "10", "11" - вычислительный базис для нескольких кубитов
        # - "Z", "X", "Y" - паули-измерения одного кубита
        # - "ZZ", "XY", ... - паули-измерения нескольких кубитов

        # Простой случай: битовая строка
        if all(c in '01' for c in label):
            # Вычислительный базис |label⟩⟨label|
            if len(label) == self.n_qubits or (self.n_qubits == 1 and len(label) == 1):
                basis_index = int(label, 2)
                if basis_index < self.dim:
                    M = np.zeros((self.dim, self.dim), dtype=np.complex128)
                    M[basis_index, basis_index] = 1.0
                    return M

        # Паули-измерения
        pauli_chars = {'I', 'X', 'Y', 'Z'}
        if all(c in pauli_chars for c in label):
            # Строка паули-операторов
            if len(label) == self.n_qubits:
                return self._build_pauli_projector(label)

        # Не удалось распознать формат
        return None

    def _build_pauli_projector(self, pauli_string: str) -> NDArray[np.complex128]:
        """
        Построить проектор для паули-измерения

        Для паули-строки "XYZ" измерение даёт результат ±1
        Проектор на собственное значение +1: P₊ = (I + σ)/2

        Args:
            pauli_string: Строка паули-операторов (например, "XYZ")

        Returns:
            Проектор на +1 собственное значение
        """
        from .state_prep import get_pauli_basis

        # Получаем паули-матрицу
        pauli_matrices = {
            'I': np.eye(2, dtype=np.complex128),
            'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
        }

        # Тензорное произведение
        pauli = pauli_matrices[pauli_string[0]]
        for char in pauli_string[1:]:
            pauli = np.kron(pauli, pauli_matrices[char])

        # Проектор на +1: P₊ = (I + σ)/2
        projector = (np.eye(self.dim, dtype=np.complex128) + pauli) / 2

        return projector

    def _build_measurement_coefficients(self,
                                       rho: NDArray[np.complex128],
                                       M: NDArray[np.complex128]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Построить коэффициенты для вероятности измерения

        Вероятность: p = Tr(M * ε(ρ))

        Связь с Choi matrix:
        ε(ρ) = Tr_A[(ρ^T ⊗ I) J]

        Полная формула:
        p = Tr(M * ε(ρ)) = Tr[(M ⊗ I) * (I ⊗ ρ^T) * J]
          = Σ_{ijkl} M_{ij} * ρ_{lk} * J_{il,jk}

        Возвращаем коэффициенты c_real и c_imag такие что:
        p = Σ c_real[i,j] * J_real[i,j] + Σ c_imag[i,j] * J_imag[i,j]

        Args:
            rho: Входное состояние
            M: POVM элемент

        Returns:
            (coeffs_real, coeffs_imag) - матрицы коэффициентов
        """
        choi_dim = self.dim ** 2

        coeffs = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

        # p = Σ_{ijkl} M_{ij} * ρ_{lk} * J_{il,jk}
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    for l in range(self.dim):
                        # Индексы в Choi matrix
                        row_idx = i * self.dim + l
                        col_idx = j * self.dim + k

                        # Коэффициент
                        coeffs[row_idx, col_idx] += M[i, j] * rho[l, k]

        # Разделяем на реальную и мнимую части
        coeffs_real = np.real(coeffs)
        coeffs_imag = np.imag(coeffs)

        return coeffs_real, coeffs_imag

    def _choi_to_kraus(self, choi: NDArray[np.complex128]) -> KrausChannel:
        """
        Преобразовать Choi matrix в операторы Крауса

        Используем спектральное разложение:
        J = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|

        Операторы Крауса: Kᵢ = √λᵢ * reshape(vᵢ, (d, d))

        Args:
            choi: Choi matrix

        Returns:
            KrausChannel
        """
        # Спектральное разложение
        eigenvalues, eigenvectors = np.linalg.eigh(choi)

        # Оставляем положительные собственные значения
        threshold = 1e-10
        positive_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:, positive_indices]

        # Формируем операторы Крауса
        kraus_operators = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]

            # Reshape: вектор размера d² -> матрица d×d
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            kraus_operators.append(K)

        if len(kraus_operators) == 0:
            # Пустой канал - возвращаем единичный
            kraus_operators = [np.eye(self.dim, dtype=np.complex128)]

        return KrausChannel(
            kraus_operators,
            n_qubits=1,
            name="MLE_reconstructed",
            validate=False
        )

    @staticmethod
    def _make_hermitian(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Сделать матрицу эрмитовой"""
        return (matrix + matrix.conj().T) / 2
