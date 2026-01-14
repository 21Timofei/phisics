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
