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
    Деполяризующий канал (Depolarizing channel)

    ε(ρ) = (1-p)ρ + p·I/d

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

    def __init__(self, p: float, n_qubits: int = 1):
        """
        Args:
            p: Параметр деполяризации (вероятность полного шума)
            n_qubits: Число кубитов
        """
        if p < 0:
            raise ValueError(f"Параметр p должен быть неотрицательным: p = {p}")

        max_p = (4 ** n_qubits) / (4 ** n_qubits - 1)
        if p > max_p:
            raise ValueError(f"Параметр p слишком большой: p = {p} > {max_p}")

        self.p = p

        # Для однокубитного случая
        if n_qubits == 1:
            dim = 2

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

        else:
            # Многокубитный случай: используем тензорное произведение
            # Для 2 кубитов генерируем все 16 операторов

            I = np.eye(2, dtype=np.complex128)
            X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

            # Параметр для каждого кубита (приближение)
            p_eff = p / n_qubits
            p_eff = min(p_eff, 0.5)  # Ограничение для стабильности

            single_kraus = [
                np.sqrt(1 - 3*p_eff/4) * I,
                np.sqrt(p_eff/4) * X,
                np.sqrt(p_eff/4) * Y,
                np.sqrt(p_eff/4) * Z
            ]

            # Создаем тензорное произведение
            # Начинаем с первого кубита
            kraus_ops = single_kraus.copy()

            # Добавляем остальные кубиты
            for _ in range(1, n_qubits):
                new_kraus = []
                for K1 in kraus_ops:
                    for K2 in single_kraus:
                        new_kraus.append(np.kron(K1, K2))
                kraus_ops = new_kraus

        super().__init__(kraus_ops, n_qubits=n_qubits,
                        name=f"Depolarizing(p={p:.4f})", validate=False)


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
                        name=f"AmplitudeDamping(γ={gamma:.4f})", validate=True)


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
            raise ValueError(f"Параметр λ должен быть в [0, 0.5]: λ = {lambda_}")

        self.lambda_ = lambda_

        I = np.eye(2, dtype=np.complex128)
        Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        # Операторы Крауса
        K0 = np.sqrt(1 - lambda_) * I
        K1 = np.sqrt(lambda_) * Z

        kraus_ops = [K0, K1]

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"PhaseDamping(λ={lambda_:.4f})", validate=True)


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
                        name=f"GAD(γ={gamma:.3f},p_th={p_th:.3f})", validate=True)


class ThermalRelaxationChannel(KrausChannel):
    """
    Канал тепловой релаксации

    Моделирует реалистичную релаксацию с временами T₁ и T₂

    Параметры:
    - T₁: время энергетической релаксации (amplitude damping)
    - T₂: время дефазировки (phase damping)
    - t: время эволюции
    - p_th: тепловая популяция

    Физическое ограничение: T₂ ≤ 2T₁

    Реализуется как композиция GAD и phase damping
    """

    def __init__(self, T1: float, T2: float, t: float, p_th: float = 0.0):
        """
        Args:
            T1: Время T₁ релаксации
            T2: Время T₂ дефазировки
            t: Время эволюции
            p_th: Тепловая популяция
        """
        if T1 <= 0 or T2 <= 0:
            raise ValueError("T1 и T2 должны быть положительными")

        if T2 > 2 * T1:
            raise ValueError(f"Физически невозможно: T₂ > 2T₁ ({T2} > {2*T1})")

        if t < 0:
            raise ValueError("Время должно быть неотрицательным")

        # Параметры каналов
        gamma = 1 - np.exp(-t / T1) if T1 != np.inf else 0

        # Чистая дефазировка (T₂* процесс)
        # 1/T₂ = 1/(2T₁) + 1/T₂*
        # λ соответствует чистой дефазировке без учёта вклада от T₁
        rate_phi = 1/T2 - 1/(2*T1) if T1 != np.inf else 1/T2
        lambda_ = (1 - np.exp(-t * rate_phi)) / 2 if rate_phi > 0 else 0
        lambda_ = max(0, min(0.5, lambda_))  # Обрезка для численной стабильности

        # Создаём композицию GAD и phase damping
        # Сначала применяем GAD
        gad = GeneralizedAmplitudeDamping(gamma, p_th)

        # Затем phase damping
        if lambda_ > 1e-10:
            pd = PhaseDampingChannel(lambda_)
            # Композиция каналов
            composed = gad.compose(pd)
            kraus_ops = composed.get_kraus_operators()
        else:
            kraus_ops = gad.get_kraus_operators()

        super().__init__(kraus_ops, n_qubits=1,
                        name=f"ThermalRelax(T1={T1:.2f},T2={T2:.2f},t={t:.2f})",
                        validate=False)

        self.T1 = T1
        self.T2 = T2
        self.t = t
        self.p_th = p_th
