"""
Стандартные модели шума для однокубитных систем
Все модели физически корректны (CPTP)
"""

import numpy as np
from typing import List
from numpy.typing import NDArray
from .kraus import KrausChannel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DepolarizingChannel(KrausChannel):
    """
    Деполяризующий канал (Depolarizing channel) для 1 кубита

    ε(ρ) = (1-p)ρ + p·I/2

    С вероятностью (1-p) состояние остаётся неизменным,
    с вероятностью p оно заменяется на максимально смешанное состояние

    Физическая интерпретация: полностью случайный шум
    Паули-ошибки X, Y, Z происходят с равной вероятностью

    Операторы Крауса:
    K₀ = √(1 - 3p/4) I
    K₁ = √(p/4) X
    K₂ = √(p/4) Y
    K₃ = √(p/4) Z

    Параметр p ∈ [0, 4/3] для физичности (1 qubit)
    На практике p ∈ [0, 1] - типичный диапазон
    """

    def __init__(self, p: float):
        """
        Инициализация деполяризующего канала

        Args:
            p: Параметр деполяризации ∈ [0, 4/3]
               - p=0: идеальный канал (нет шума)
               - p=0.75 (3/4): максимальный физический шум для 1 кубита
               - p=1: полностью деполяризующий канал (выход = I/2 для любого входа)

       """
        # === ВАЛИДАЦИЯ ПАРАМЕТРА ===
        if p < 0:
            raise ValueError(f"Параметр p должен быть неотрицательным: p = {p}")

        # Максимальное физически допустимое значение для 1 кубита
        # При p > 4/3 коэффициенты операторов Крауса становятся комплексными
        max_p = 4 / 3
        if p > max_p:
            raise ValueError(f"Параметр p слишком большой: p = {p} > {max_p}")

        self.p = p

        I = np.eye(2, dtype=np.complex128)        # Единица
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        kraus_ops = [
            np.sqrt(1 - 3*p/4) * I,  # Вероятность "нет ошибки"
            np.sqrt(p/4) * X,        # Вероятность X-ошибки
            np.sqrt(p/4) * Y,        # Вероятность Y-ошибки
            np.sqrt(p/4) * Z         # Вероятность Z-ошибки
        ]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"Depolarizing(p={p:.4f})", validate=True)


class AmplitudeDampingChannel(KrausChannel):
    """
    Амплитудное затухание (Amplitude Damping)

    Моделирует потерю энергии: |1⟩ → |0⟩ с вероятностью γ
    Это необратимый процесс релаксации к основному состоянию

    Физическая интерпретация:
    - Спонтанное испускание фотона
    - Релаксация к термическому равновесию при T=0
    - T₁ процесс в квантовых системах

    Операторы Крауса:
    K₀ = [[1, 0], [0, √(1-γ)]]
    K₁ = [[0, √γ], [0, 0]]

    Действие:
    ε(|0⟩⟨0|) = |0⟩⟨0|  (основное состояние стабильно)
    ε(|1⟩⟨1|) = (1-γ)|1⟩⟨1| + γ|0⟩⟨0|  (затухание к |0⟩)
    ε(|+⟩⟨+|) теряет когерентность

    Параметр γ ∈ [0, 1]
    """

    def __init__(self, gamma: float):
        """
        Args:
            gamma: Параметр затухания (вероятность перехода |1⟩ → |0⟩)
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f"Параметр γ должен быть в [0, 1]: γ = {gamma}")

        self.gamma = gamma

        # Операторы Крауса
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=np.complex128)

        K1 = np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ], dtype=np.complex128)

        kraus_ops = [K0, K1]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"AmplitudeDamping(gamma={gamma:.4f})", validate=True)


class PhaseDampingChannel(KrausChannel):
    """
    Фазовое затухание (Phase Damping / Dephasing)

    Разрушает когерентность без изменения популяций
    Недиагональные элементы матрицы плотности затухают

    Физическая интерпретация:
    - Случайные флуктуации фазы
    - T₂ процесс (чистый dephasing, T₂* процесс)
    - Взаимодействие с медленным флуктуирующим окружением

    Операторы Крауса:
    K₀ = √(1-λ) I
    K₁ = √λ Z

    Действие:
    ε(ρ) = (1-λ)ρ + λ Z ρ Z

    Диагональные элементы не меняются: ⟨0|ε(ρ)|0⟩ = ⟨0|ρ|0⟩
    Недиагональные затухают: ⟨0|ε(ρ)|1⟩ = (1-2λ)⟨0|ρ|1⟩

    Параметр λ ∈ [0, 1/2] для сохранения положительности
    """

    def __init__(self, lambda_: float):
        """
        Args:
            lambda_: Параметр дефазировки
        """
        if not 0 <= lambda_ <= 0.5:
            raise ValueError(f"Параметр lambda должен быть в [0, 0.5]: lambda = {lambda_}")

        self.lambda_ = lambda_

        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Операторы Крауса
        K0 = np.sqrt(1 - lambda_) * I
        K1 = np.sqrt(lambda_) * Z

        kraus_ops = [K0, K1]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"PhaseDamping(lambda={lambda_:.4f})", validate=True)
