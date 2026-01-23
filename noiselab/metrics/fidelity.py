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
    Average gate fidelity (средняя верность гейта) для 1 кубита

    F_avg = ∫ dψ ⟨ψ|ε₂(ε₁(|ψ⟩⟨ψ|))|ψ⟩

    Усреднение по всем чистым состояниям по мере Хаара

    Связь с process fidelity для 1 кубита:
    F_avg = (2·F_proc + 1) / 3

    Args:
        channel1: Первый канал (обычно идеальный)
        channel2: Второй канал (реальный с шумом)

    Returns:
        Average gate fidelity ∈ [0, 1]
    """
    F_proc = process_fidelity(channel1, channel2)
    d = 2  # Для 1 кубита

    F_avg = (d * F_proc + 1) / (d + 1)

    return F_avg


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
