"""
Метрики верности (fidelity) для квантовых каналов

Process fidelity измеряет насколько хорошо один канал аппроксимирует другой
"""

import numpy as np
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def process_fidelity_choi(choi1: NDArray[np.complex128],
                          choi2: NDArray[np.complex128]) -> float:
    """
    Process fidelity между двумя каналами через Choi matrices

    ПРАВИЛЬНАЯ формула (Gilchrist et al., 2005):
    F_proc(ε₁, ε₂) = Tr(J₁† J₂) / √(Tr(J₁† J₁) · Tr(J₂† J₂))

    Это нормализованное скалярное произведение Гильберта-Шмидта,
    инвариантное относительно нормализации Choi matrices.

    Args:
        choi1: Choi matrix первого канала
        choi2: Choi matrix второго канала

    Returns:
        Process fidelity ∈ [0, 1]
    """
    if choi1.shape != choi2.shape:
        raise ValueError("Choi matrices должны иметь одинаковую размерность")

    # Вычисляем скалярное произведение Гильберта-Шмидта
    # ⟨J₁, J₂⟩ = Tr(J₁† J₂)
    inner_product = np.trace(choi1.conj().T @ choi2)

    # Вычисляем нормы
    # ||J||² = Tr(J† J)
    norm1_sq = np.trace(choi1.conj().T @ choi1).real
    norm2_sq = np.trace(choi2.conj().T @ choi2).real

    if norm1_sq < 1e-15 or norm2_sq < 1e-15:
        return 0.0

    # ПРАВИЛЬНАЯ формула: F = Tr(J₁† J₂) / √(Tr(J₁† J₁) · Tr(J₂† J₂))
    fidelity = inner_product.real / np.sqrt(norm1_sq * norm2_sq)

    # Обрезка для численной стабильности
    fidelity = np.clip(fidelity, 0.0, 1.0)

    return fidelity


def process_fidelity(channel1, channel2) -> float:
    """
    Process fidelity между двумя квантовыми каналами

    Args:
        channel1: QuantumChannel объект
        channel2: QuantumChannel объект

    Returns:
        Process fidelity
    """
    if channel1.n_qubits != channel2.n_qubits:
        raise ValueError("Каналы должны иметь одинаковое число кубитов")

    choi1 = channel1.get_choi_matrix()
    choi2 = channel2.get_choi_matrix()

    return process_fidelity_choi(choi1, choi2)


def average_gate_fidelity(channel1, channel2) -> float:
    """
    Average gate fidelity (средняя верность гейта)

    F_avg = ∫ dψ ⟨ψ|ε₂(ε₁(|ψ⟩⟨ψ|))|ψ⟩

    Усреднение по всем чистым состояниям по мере Хаара

    Связь с process fidelity:
    F_avg = (d·F_proc + 1) / (d + 1)

    где d - размерность системы

    Args:
        channel1: Первый канал (обычно идеальный)
        channel2: Второй канал (реальный с шумом)

    Returns:
        Average gate fidelity ∈ [0, 1]
    """
    F_proc = process_fidelity(channel1, channel2)
    d = 2 ** channel1.n_qubits

    F_avg = (d * F_proc + 1) / (d + 1)

    return F_avg


def entanglement_fidelity(channel1, channel2,
                          pure_state=None) -> float:
    """
    Entanglement fidelity для конкретного состояния

    F_e = ⟨Φ|(ε₁ ⊗ I)(ε₂ ⊗ I)(|Φ⟩⟨Φ|)|Φ⟩

    где |Φ⟩ - запутанное состояние (обычно |Φ⁺⟩)

    Измеряет насколько хорошо канал сохраняет запутанность

    Args:
        channel1: Идеальный канал
        channel2: Реальный канал
        pure_state: Чистое состояние (если None, используется |Φ⁺⟩)

    Returns:
        Entanglement fidelity
    """
    from ..core.states import QuantumState, DensityMatrix

    d = 2 ** channel1.n_qubits

    # Максимально запутанное состояние |Φ⁺⟩ = Σᵢ|i⟩|i⟩/√d
    if pure_state is None:
        phi_plus = np.zeros(d ** 2, dtype=np.complex128)
        for i in range(d):
            phi_plus[i * d + i] = 1.0
        phi_plus /= np.sqrt(d)
        pure_state = QuantumState(phi_plus, normalize=False)

    rho = pure_state.to_density_matrix()

    # Применяем каналы (расширение на удвоенную систему)
    # Упрощение: используем Choi representation
    choi1 = channel1.get_choi_matrix()
    choi2 = channel2.get_choi_matrix()

    # Fidelity между Choi matrices (упрощённая версия)
    # F_e = Tr(ρ (J₁† J₂))
    fidelity = np.trace(rho.matrix @ (choi1.conj().T @ choi2)).real

    return np.clip(fidelity, 0.0, 1.0)


def gate_infidelity(channel_ideal, channel_noisy) -> float:
    """
    Gate infidelity: r = 1 - F_avg

    Часто используется мера ошибки гейта

    Args:
        channel_ideal: Идеальный канал
        channel_noisy: Зашумлённый канал

    Returns:
        Infidelity ∈ [0, 1]
    """
    F = average_gate_fidelity(channel_ideal, channel_noisy)
    return 1.0 - F


def state_fidelity(rho1: NDArray[np.complex128],
                   rho2: NDArray[np.complex128]) -> float:
    """
    Fidelity между двумя квантовыми состояниями (матрицами плотности)

    F(ρ₁, ρ₂) = (Tr√(√ρ₁ ρ₂ √ρ₁))²

    Упрощённая формула для чистых состояний:
    F = |⟨ψ₁|ψ₂⟩|²

    Args:
        rho1: Первая матрица плотности
        rho2: Вторая матрица плотности

    Returns:
        Fidelity ∈ [0, 1]
    """
    # Квадратный корень из ρ₁
    eigenvalues1, eigenvectors1 = np.linalg.eigh(rho1)
    eigenvalues1 = np.maximum(eigenvalues1, 0)
    sqrt_rho1 = eigenvectors1 @ np.diag(np.sqrt(eigenvalues1)) @ eigenvectors1.conj().T

    # √ρ₁ ρ₂ √ρ₁
    M = sqrt_rho1 @ rho2 @ sqrt_rho1

    # Квадратный корень из M
    eigenvalues_M, eigenvectors_M = np.linalg.eigh(M)
    eigenvalues_M = np.maximum(eigenvalues_M, 0)
    sqrt_M = eigenvectors_M @ np.diag(np.sqrt(eigenvalues_M)) @ eigenvectors_M.conj().T

    # Fidelity
    fidelity = np.trace(sqrt_M).real ** 2

    return np.clip(fidelity, 0.0, 1.0)


def channel_fidelity_monte_carlo(channel1, channel2,
                                 n_samples: int = 1000) -> dict:
    """
    Monte Carlo оценка различных fidelity метрик

    Сэмплируем случайные чистые состояния и усредняем fidelity

    Args:
        channel1: Первый канал
        channel2: Второй канал
        n_samples: Число случайных состояний

    Returns:
        Словарь со статистикой
    """
    from ..core.states import QuantumState
    from ..channels.random import random_unitary

    d = 2 ** channel1.n_qubits
    fidelities = []

    for _ in range(n_samples):
        # Случайное чистое состояние
        U = random_unitary(d)
        psi = U[:, 0]  # Первый столбец

        state = QuantumState(psi, normalize=False)
        rho = state.to_density_matrix()

        # Применяем каналы
        rho1 = channel1.apply(rho)
        rho2 = channel2.apply(rho)

        # Fidelity между выходными состояниями
        F = state_fidelity(rho1.matrix, rho2.matrix)
        fidelities.append(F)

    return {
        "mean_fidelity": np.mean(fidelities),
        "std_fidelity": np.std(fidelities),
        "min_fidelity": np.min(fidelities),
        "max_fidelity": np.max(fidelities),
        "median_fidelity": np.median(fidelities)
    }
