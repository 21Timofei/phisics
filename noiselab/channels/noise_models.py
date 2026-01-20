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
        Args:
            p: Параметр деполяризации (вероятность полного шума)
        """
        if p < 0:
            raise ValueError(f"Параметр p должен быть неотрицательным: p = {p}")

        max_p = 4 / 3
        if p > max_p:
            raise ValueError(f"Параметр p слишком большой: p = {p} > {max_p}")

        self.p = p

        # Паули матрицы
        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Операторы Крауса
        kraus_ops = [
            np.sqrt(1 - 3*p/4) * I,
            np.sqrt(p/4) * X,
            np.sqrt(p/4) * Y,
            np.sqrt(p/4) * Z
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




class BitFlipChannel(KrausChannel):
    """
    Канал переворота бита (Bit Flip)

    С вероятностью p применяется X (NOT), с вероятностью (1-p) ничего

    ε(ρ) = (1-p)ρ + p X ρ X

    Физическая интерпретация:
    - Классическая ошибка переворота бита
    - Спонтанный переход |0⟩ ↔ |1⟩

    Операторы Крауса:
    K₀ = √(1-p) I
    K₁ = √p X
    """

    def __init__(self, p: float):
        """
        Args:
            p: Вероятность переворота бита
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Вероятность должна быть в [0, 1]: p = {p}")

        self.p = p

        I = np.eye(2, dtype=np.complex128)
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

        kraus_ops = [
            np.sqrt(1 - p) * I,
            np.sqrt(p) * X
        ]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"BitFlip(p={p:.4f})", validate=True)


class PhaseFlipChannel(KrausChannel):
    """
    Канал переворота фазы (Phase Flip)

    С вероятностью p применяется Z, с вероятностью (1-p) ничего

    ε(ρ) = (1-p)ρ + p Z ρ Z

    Физическая интерпретация:
    - Случайное изменение знака фазы
    - Эквивалентно bit flip в базисе X

    Операторы Крауса:
    K₀ = √(1-p) I
    K₁ = √p Z
    """

    def __init__(self, p: float):
        """
        Args:
            p: Вероятность переворота фазы
        """
        if not 0 <= p <= 1:
            raise ValueError(f"Вероятность должна быть в [0, 1]: p = {p}")

        self.p = p

        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        kraus_ops = [
            np.sqrt(1 - p) * I,
            np.sqrt(p) * Z
        ]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"PhaseFlip(p={p:.4f})", validate=True)


class GeneralizedAmplitudeDamping(KrausChannel):
    """
    Обобщённое амплитудное затухание (Generalized Amplitude Damping)

    Моделирует релаксацию к термическому равновесию при конечной температуре

    Комбинация амплитудного затухания вверх и вниз:
    - Затухание |1⟩ → |0⟩ с вероятностью γ(1-pₜₕ)
    - Возбуждение |0⟩ → |1⟩ с вероятностью γpₜₕ

    где pₜₕ = 1/(1 + exp(ΔE/kT)) - тепловая популяция

    Параметры:
    - γ ∈ [0, 1]: скорость релаксации
    - p_th ∈ [0, 1]: тепловая популяция возбуждённого состояния

    При p_th = 0 редуцируется к обычному amplitude damping (T=0)
    """

    def __init__(self, gamma: float, p_th: float):
        """
        Args:
            gamma: Параметр затухания
            p_th: Тепловая популяция |1⟩
        """
        if not 0 <= gamma <= 1:
            raise ValueError(f"Параметр γ должен быть в [0, 1]: γ = {gamma}")
        if not 0 <= p_th <= 1:
            raise ValueError(f"Параметр p_th должен быть в [0, 1]: p_th = {p_th}")

        self.gamma = gamma
        self.p_th = p_th

        # Четыре оператора Крауса
        # Затухание с популяцией (1-p_th)
        K0 = np.sqrt(1 - p_th) * np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=np.complex128)

        K1 = np.sqrt(1 - p_th) * np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ], dtype=np.complex128)

        # Возбуждение с популяцией p_th
        K2 = np.sqrt(p_th) * np.array([
            [np.sqrt(1 - gamma), 0],
            [0, 1]
        ], dtype=np.complex128)

        K3 = np.sqrt(p_th) * np.array([
            [0, 0],
            [np.sqrt(gamma), 0]
        ], dtype=np.complex128)

        kraus_ops = [K0, K1, K2, K3]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"GAD(gamma={gamma:.3f},p_th={p_th:.3f})", validate=True)


