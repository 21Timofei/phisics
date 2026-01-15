"""
Двухкубитные шумовые каналы
Моделируют коррелированные ошибки и кросс-токи
"""

import numpy as np
from typing import List
from numpy.typing import NDArray
from .kraus import KrausChannel


class TwoQubitDepolarizing(KrausChannel):
    """
    Двухкубитный деполяризующий канал

    ε(ρ) = (1-p)ρ + p·I/4

    Обобщение однокубитного канала на 2 кубита
    Применяются случайные паули-операторы на оба кубита

    Операторы Крауса: {√(wᵢ) σᵢ ⊗ σⱼ}
    где σᵢ ∈ {I, X, Y, Z} для каждого кубита
    Всего 16 операторов
    """

    def __init__(self, p: float):
        """
        Args:
            p: Параметр деполяризации
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Параметр p должен быть в [0, 1]: p = {p}")

        self.p = p

        # Паули матрицы
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        paulis = [I, X, Y, Z]

        # 16 операторов Крауса
        kraus_ops = []
        for i, P1 in enumerate(paulis):
            for j, P2 in enumerate(paulis):
                # Тензорное произведение
                op = np.kron(P1, P2)

                if i == 0 and j == 0:
                    # I ⊗ I: коэффициент (1 - 15p/16)
                    coeff = np.sqrt(1 - 15*p/16)
                else:
                    # Остальные 15 операторов: коэффициент p/16
                    coeff = np.sqrt(p/16)

                kraus_ops.append(coeff * op)

        super().__init__(kraus_ops, n_qubits=2,
                        name=f"TwoQubitDepol(p={p:.4f})", validate=True)


class CorrelatedNoise(KrausChannel):
    """
    Коррелированный шум между кубитами

    Моделирует ситуацию, когда ошибки на двух кубитах не независимы
    Например, оба кубита переворачиваются одновременно с некоторой вероятностью

    ε(ρ) = (1-p_corr) ε_uncorr(ρ) + p_corr (σ ⊗ σ) ρ (σ ⊗ σ)†

    где σ - одинаковая паули-операция на оба кубита
    """

    def __init__(self, p_single: float, p_correlated: float,
                 correlation_type: str = 'XX'):
        """
        Args:
            p_single: Вероятность некоррелированной ошибки на каждом кубите
            p_correlated: Вероятность коррелированной ошибки
            correlation_type: Тип корреляции ('XX', 'YY', 'ZZ')
        """
        if not 0 <= p_single <= 1:
            raise ValueError("p_single должен быть в [0, 1]")
        if not 0 <= p_correlated <= 1:
            raise ValueError("p_correlated должен быть в [0, 1]")

        self.p_single = p_single
        self.p_correlated = p_correlated
        self.correlation_type = correlation_type

        # Паули матрицы
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Выбор коррелированной операции
        pauli_map = {'XX': X, 'YY': Y, 'ZZ': Z}
        if correlation_type not in pauli_map:
            raise ValueError(f"Неизвестный тип корреляции: {correlation_type}")

        sigma = pauli_map[correlation_type]

        # Операторы Крауса
        # 1. Без ошибок
        II = np.kron(I, I)

        # 2. Ошибка на первом кубите
        XI = np.kron(X, I)

        # 3. Ошибка на втором кубите
        IX = np.kron(I, X)

        # 4. Коррелированная ошибка
        SS = np.kron(sigma, sigma)

        # Вычисляем коэффициенты
        p_none = 1 - 2*p_single - p_correlated

        if p_none < 0:
            raise ValueError("Сумма вероятностей слишком большая")

        kraus_ops = [
            np.sqrt(p_none) * II,
            np.sqrt(p_single) * XI,
            np.sqrt(p_single) * IX,
            np.sqrt(p_correlated) * SS
        ]

        super().__init__(kraus_ops, n_qubits=2,
                        name=f"Correlated({correlation_type},p_s={p_single:.3f},p_c={p_correlated:.3f})",
                        validate=True)


class CrosstalkChannel(KrausChannel):
    """
    Канал кросс-токов (Crosstalk)

    Моделирует нежелательное взаимодействие между кубитами
    Когда гейт применяется к одному кубиту, он частично воздействует на соседний

    Например, при применении X к первому кубиту с вероятностью ε применяется
    CNOT(0,1) вместо X⊗I

    Физическая причина: паразитные взаимодействия, емкостная связь
    """

    def __init__(self, epsilon: float, interaction_type: str = 'CNOT'):
        """
        Args:
            epsilon: Сила кросс-тока (вероятность нежелательного взаимодействия)
            interaction_type: Тип взаимодействия ('CNOT', 'CZ', 'SWAP')
        """
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon должен быть в [0, 1]")

        self.epsilon = epsilon
        self.interaction_type = interaction_type

        I = np.eye(2, dtype=np.complex128)
        II = np.kron(I, I)

        # Матрицы взаимодействия
        if interaction_type == 'CNOT':
            interaction = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ], dtype=np.complex128)

        elif interaction_type == 'CZ':
            interaction = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1]
            ], dtype=np.complex128)

        elif interaction_type == 'SWAP':
            interaction = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=np.complex128)

        else:
            raise ValueError(f"Неизвестный тип взаимодействия: {interaction_type}")

        # Операторы Крауса
        kraus_ops = [
            np.sqrt(1 - epsilon) * II,
            np.sqrt(epsilon) * interaction
        ]

        super().__init__(kraus_ops, n_qubits=2,
                        name=f"Crosstalk({interaction_type},ε={epsilon:.4f})",
                        validate=True)


class BellStateDecay(KrausChannel):
    """
    Декогеренция Белловых пар

    Моделирует затухание запутанного состояния к сепарабельному
    Важно для квантовой коммуникации и квантовых сетей

    |Φ⁺⟩ = (|00⟩ + |11⟩)/√2 → смешанное состояние

    Параметры:
    - p_dephase: дефазировка (сохраняет диагональ)
    - p_relax: релаксация (переходы между уровнями)
    """

    def __init__(self, p_dephase: float, p_relax: float = 0.0):
        """
        Args:
            p_dephase: Вероятность дефазировки
            p_relax: Вероятность релаксации
        """
        if not 0 <= p_dephase <= 1:
            raise ValueError("p_dephase должен быть в [0, 1]")
        if not 0 <= p_relax <= 1:
            raise ValueError("p_relax должен быть в [0, 1]")

        if p_dephase + p_relax > 1:
            raise ValueError("Сумма вероятностей превышает 1")

        self.p_dephase = p_dephase
        self.p_relax = p_relax

        # Паули матрицы
        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

        # Операторы
        II = np.kron(I, I)
        ZZ = np.kron(Z, Z)
        XX = np.kron(X, X)

        # Операторы Крауса
        p_none = 1 - p_dephase - p_relax

        kraus_ops = [
            np.sqrt(p_none) * II,
            np.sqrt(p_dephase) * ZZ,  # Дефазировка Белловой пары
        ]

        if p_relax > 0:
            kraus_ops.append(np.sqrt(p_relax) * XX)  # Релаксация

        super().__init__(kraus_ops, n_qubits=2,
                        name=f"BellDecay(p_d={p_dephase:.3f},p_r={p_relax:.3f})",
                        validate=True)


class GeneralCorrelatedNoise(KrausChannel):
    """
    Обобщённый коррелированный шум для двух кубитов

    Полная параметризация всех 16 паули-ошибок:
    ε(ρ) = Σᵢⱼ pᵢⱼ (σᵢ ⊗ σⱼ) ρ (σᵢ ⊗ σⱼ)†

    где σ₀ = I, σ₁ = X, σ₂ = Y, σ₃ = Z
    и Σᵢⱼ pᵢⱼ = 1

    Поддерживает:
    - Асимметричные вероятности ошибок (p1 ≠ p2)
    - Смешанные корреляции (σᵢ⊗σⱼ где i≠j)
    - Полную матрицу 4x4 вероятностей
    """

    # Паули матрицы как атрибуты класса
    PAULIS = {
        'I': np.eye(2, dtype=np.complex128),
        'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
    }
    PAULI_LIST = ['I', 'X', 'Y', 'Z']

    def __init__(self,
                 error_probabilities: NDArray = None,
                 p_single_1: float = 0.0,
                 p_single_2: float = 0.0,
                 p_correlated: dict = None,
                 validate_sum: bool = True):
        """
        Два способа инициализации:

        1. Полная матрица вероятностей error_probabilities[i,j] = p(σᵢ⊗σⱼ)

        2. Упрощённые параметры:
           - p_single_1: вероятность ошибки (X,Y,Z равновероятны) на кубите 1
           - p_single_2: вероятность ошибки (X,Y,Z равновероятны) на кубите 2
           - p_correlated: словарь {'XX': p_xx, 'XY': p_xy, ...}

        Args:
            error_probabilities: Матрица 4x4 вероятностей паули-ошибок
            p_single_1: Вероятность однокубитной ошибки на кубите 1
            p_single_2: Вероятность однокубитной ошибки на кубите 2
            p_correlated: Словарь коррелированных ошибок
            validate_sum: Проверять ли что сумма = 1
        """
        if error_probabilities is not None:
            # Способ 1: полная матрица
            self.error_probs = np.array(error_probabilities, dtype=np.float64)

            if self.error_probs.shape != (4, 4):
                raise ValueError("error_probabilities должна быть матрицей 4x4")

            if np.any(self.error_probs < -1e-10):
                raise ValueError("Вероятности не могут быть отрицательными")

            prob_sum = np.sum(self.error_probs)
            if validate_sum and not np.isclose(prob_sum, 1.0, atol=1e-8):
                raise ValueError(f"Сумма вероятностей = {prob_sum}, должна быть 1")
        else:
            # Способ 2: упрощённые параметры
            self.error_probs = self._build_probability_matrix(
                p_single_1, p_single_2, p_correlated or {}
            )

        # Строим операторы Крауса
        kraus_ops = self._build_kraus_operators()

        super().__init__(
            kraus_ops,
            n_qubits=2,
            name=self._generate_name(),
            validate=True
        )

    def _build_probability_matrix(self,
                                  p_single_1: float,
                                  p_single_2: float,
                                  p_correlated: dict) -> NDArray:
        """
        Строит полную матрицу вероятностей из упрощённых параметров

        Args:
            p_single_1: Вероятность ошибки на кубите 1 (распределяется по X,Y,Z)
            p_single_2: Вероятность ошибки на кубите 2 (распределяется по X,Y,Z)
            p_correlated: Словарь коррелированных ошибок {'XX': 0.1, 'ZZ': 0.05, ...}
        """
        probs = np.zeros((4, 4), dtype=np.float64)

        # Однокубитные ошибки на кубите 1: X⊗I, Y⊗I, Z⊗I (равновероятны)
        if p_single_1 > 0:
            p_each_1 = p_single_1 / 3
            probs[1, 0] = p_each_1  # X⊗I
            probs[2, 0] = p_each_1  # Y⊗I
            probs[3, 0] = p_each_1  # Z⊗I

        # Однокубитные ошибки на кубите 2: I⊗X, I⊗Y, I⊗Z (равновероятны)
        if p_single_2 > 0:
            p_each_2 = p_single_2 / 3
            probs[0, 1] = p_each_2  # I⊗X
            probs[0, 2] = p_each_2  # I⊗Y
            probs[0, 3] = p_each_2  # I⊗Z

        # Коррелированные ошибки
        for error_type, prob in p_correlated.items():
            if len(error_type) != 2:
                raise ValueError(f"Неверный формат ошибки: {error_type}. Ожидается 2 символа (например 'XX', 'XY')")

            try:
                i = self.PAULI_LIST.index(error_type[0])
                j = self.PAULI_LIST.index(error_type[1])
            except ValueError:
                raise ValueError(f"Неизвестный паули-оператор в {error_type}. Допустимы: I, X, Y, Z")

            probs[i, j] += prob

        # Вероятность отсутствия ошибок (I⊗I)
        p_total_errors = np.sum(probs)
        if p_total_errors > 1.0 + 1e-10:
            raise ValueError(f"Сумма вероятностей ошибок > 1: {p_total_errors}")

        probs[0, 0] = max(0, 1.0 - p_total_errors)

        return probs

    def _build_kraus_operators(self) -> List[NDArray]:
        """
        Строит операторы Крауса из матрицы вероятностей

        Kᵢⱼ = √pᵢⱼ (σᵢ ⊗ σⱼ)
        """
        kraus_ops = []

        for i, pauli_1 in enumerate(self.PAULI_LIST):
            for j, pauli_2 in enumerate(self.PAULI_LIST):
                prob = self.error_probs[i, j]

                if prob > 1e-15:  # Только ненулевые вероятности
                    sigma_1 = self.PAULIS[pauli_1]
                    sigma_2 = self.PAULIS[pauli_2]

                    K = np.sqrt(prob) * np.kron(sigma_1, sigma_2)
                    kraus_ops.append(K)

        return kraus_ops

    def _generate_name(self) -> str:
        """Генерирует информативное имя канала"""
        # Находим доминирующие ошибки (кроме II)
        significant = []
        for i, p1 in enumerate(self.PAULI_LIST):
            for j, p2 in enumerate(self.PAULI_LIST):
                if i == 0 and j == 0:
                    continue
                prob = self.error_probs[i, j]
                if prob > 0.001:
                    significant.append(f"{p1}{p2}:{prob:.3f}")

        if not significant:
            return "GeneralCorrelated(no_errors)"

        return f"GeneralCorrelated({', '.join(significant[:3])}{'...' if len(significant) > 3 else ''})"

    def get_error_rates(self) -> dict:
        """
        Возвращает словарь всех ненулевых вероятностей ошибок

        Returns:
            Словарь {error_type: probability}
        """
        rates = {}
        for i, p1 in enumerate(self.PAULI_LIST):
            for j, p2 in enumerate(self.PAULI_LIST):
                prob = self.error_probs[i, j]
                if prob > 1e-15:
                    rates[f"{p1}{p2}"] = float(prob)
        return rates

    def get_marginal_error_rates(self) -> tuple:
        """
        Вычисляет маргинальные вероятности ошибок для каждого кубита

        Returns:
            Tuple[Dict[str, float], Dict[str, float]] - (errors_qubit1, errors_qubit2)
        """
        errors_1 = {'I': 0.0, 'X': 0.0, 'Y': 0.0, 'Z': 0.0}
        errors_2 = {'I': 0.0, 'X': 0.0, 'Y': 0.0, 'Z': 0.0}

        for i, p1 in enumerate(self.PAULI_LIST):
            for j, p2 in enumerate(self.PAULI_LIST):
                prob = self.error_probs[i, j]
                errors_1[p1] += prob
                errors_2[p2] += prob

        return errors_1, errors_2

    def correlation_strength(self) -> float:
        """
        Вычисляет силу корреляций как отклонение от независимых ошибок

        C = Σᵢⱼ |pᵢⱼ - pᵢ·pⱼ| для i,j > 0

        где pᵢ, pⱼ - маргинальные вероятности

        Returns:
            Мера корреляции ∈ [0, ~1]
        """
        errors_1, errors_2 = self.get_marginal_error_rates()

        correlation = 0.0
        for i, p1 in enumerate(self.PAULI_LIST):
            for j, p2 in enumerate(self.PAULI_LIST):
                if i == 0 or j == 0:
                    continue  # Пропускаем I-компоненты

                p_joint = self.error_probs[i, j]
                p_independent = errors_1[p1] * errors_2[p2]

                correlation += abs(p_joint - p_independent)

        return correlation

    @classmethod
    def depolarizing_like(cls, p: float) -> 'GeneralCorrelatedNoise':
        """
        Создаёт канал, эквивалентный двухкубитному деполяризующему

        pᵢⱼ = p/16 для всех (i,j) ≠ (0,0)
        p₀₀ = 1 - 15p/16

        Args:
            p: Параметр деполяризации

        Returns:
            GeneralCorrelatedNoise эквивалентный TwoQubitDepolarizing(p)
        """
        if not 0 <= p <= 1:
            raise ValueError(f"p должен быть в [0, 1]: p = {p}")

        probs = np.full((4, 4), p/16, dtype=np.float64)
        probs[0, 0] = 1 - 15*p/16

        return cls(error_probabilities=probs, validate_sum=True)

    @classmethod
    def asymmetric_depolarizing(cls,
                               p1: float,
                               p2: float,
                               p_corr: float = 0.0) -> 'GeneralCorrelatedNoise':
        """
        Асимметричный деполяризующий канал

        Позволяет задать разные вероятности ошибок на разных кубитах

        Args:
            p1: Вероятность ошибки на кубите 1 (распределяется по X,Y,Z)
            p2: Вероятность ошибки на кубите 2 (распределяется по X,Y,Z)
            p_corr: Вероятность коррелированных ошибок (XX+YY+ZZ, равновероятны)

        Returns:
            GeneralCorrelatedNoise с асимметричными ошибками
        """
        probs = np.zeros((4, 4), dtype=np.float64)

        # Однокубитные ошибки на кубите 1: X⊗I, Y⊗I, Z⊗I
        if p1 > 0:
            probs[1, 0] = p1 / 3  # X⊗I
            probs[2, 0] = p1 / 3  # Y⊗I
            probs[3, 0] = p1 / 3  # Z⊗I

        # Однокубитные ошибки на кубите 2: I⊗X, I⊗Y, I⊗Z
        if p2 > 0:
            probs[0, 1] = p2 / 3  # I⊗X
            probs[0, 2] = p2 / 3  # I⊗Y
            probs[0, 3] = p2 / 3  # I⊗Z

        # Коррелированные ошибки XX, YY, ZZ
        if p_corr > 0:
            probs[1, 1] = p_corr / 3  # X⊗X
            probs[2, 2] = p_corr / 3  # Y⊗Y
            probs[3, 3] = p_corr / 3  # Z⊗Z

        # Остаток - без ошибок
        probs[0, 0] = 1.0 - np.sum(probs)

        if probs[0, 0] < -1e-10:
            raise ValueError(f"Сумма вероятностей ошибок > 1: {1 - probs[0, 0]}")

        probs[0, 0] = max(0, probs[0, 0])

        return cls(error_probabilities=probs, validate_sum=True)

    @classmethod
    def from_correlation_matrix(cls,
                                p_errors: dict,
                                correlation_matrix: NDArray) -> 'GeneralCorrelatedNoise':
        """
        Создаёт канал из маргинальных вероятностей и матрицы корреляций

        Args:
            p_errors: Словарь {'X1': p, 'Y1': p, ...} для маргинальных ошибок
            correlation_matrix: Матрица 3x3 корреляций между X,Y,Z ошибками

        Returns:
            GeneralCorrelatedNoise с заданными корреляциями
        """
        # Маргинальные вероятности для каждого кубита
        p1 = {'X': p_errors.get('X1', 0), 'Y': p_errors.get('Y1', 0), 'Z': p_errors.get('Z1', 0)}
        p2 = {'X': p_errors.get('X2', 0), 'Y': p_errors.get('Y2', 0), 'Z': p_errors.get('Z2', 0)}

        probs = np.zeros((4, 4), dtype=np.float64)

        pauli_nontrivial = ['X', 'Y', 'Z']

        # Совместные вероятности: P(i,j) = P(i)*P(j) + Corr(i,j)
        for i, pi in enumerate(pauli_nontrivial):
            for j, pj in enumerate(pauli_nontrivial):
                idx_i = i + 1  # +1 потому что I на позиции 0
                idx_j = j + 1

                p_indep = p1[pi] * p2[pj]
                corr = correlation_matrix[i, j] if correlation_matrix is not None else 0

                probs[idx_i, idx_j] = max(0, p_indep + corr)

        # Однокубитные ошибки (когда другой кубит без ошибки)
        for i, pi in enumerate(pauli_nontrivial):
            idx_i = i + 1
            # Вероятность ошибки только на кубите 1
            probs[idx_i, 0] = max(0, p1[pi] - sum(probs[idx_i, 1:]))
            # Вероятность ошибки только на кубите 2
            probs[0, idx_i] = max(0, p2[pi] - sum(probs[1:, idx_i]))

        # Без ошибок
        probs[0, 0] = max(0, 1.0 - np.sum(probs))

        return cls(error_probabilities=probs, validate_sum=True)
