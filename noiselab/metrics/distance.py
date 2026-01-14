"""
Метрики расстояния между квантовыми каналами
"""

import numpy as np
from typing import Optional
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def frobenius_distance(choi1: NDArray[np.complex128],
                       choi2: NDArray[np.complex128]) -> float:
    """
    Frobenius distance между Choi matrices

    D_F(ε₁, ε₂) = ||J₁ - J₂||_F = √Tr((J₁-J₂)†(J₁-J₂))

    Простая метрика, но не операционально значимая

    Args:
        choi1: Первая Choi matrix
        choi2: Вторая Choi matrix

    Returns:
        Frobenius distance
    """
    if choi1.shape != choi2.shape:
        raise ValueError("Choi matrices должны иметь одинаковую размерность")

    diff = choi1 - choi2
    distance = np.linalg.norm(diff, ord='fro')

    return distance


def trace_distance_states(rho1: NDArray[np.complex128],
                          rho2: NDArray[np.complex128]) -> float:
    """
    Trace distance между двумя квантовыми состояниями

    D(ρ₁, ρ₂) = ½ Tr|ρ₁ - ρ₂|

    где |A| = √(A†A) - модуль матрицы через сингулярные значения

    Args:
        rho1: Первое состояние
        rho2: Второе состояние

    Returns:
        Trace distance ∈ [0, 1]
    """
    diff = rho1 - rho2

    # Сингулярные значения (собственные значения |A|)
    singular_values = np.linalg.svd(diff, compute_uv=False)

    distance = 0.5 * np.sum(singular_values)

    return distance


def trace_distance_channels(channel1, channel2,
                            state=None) -> float:
    """
    Trace distance между каналами для данного состояния

    D(ε₁, ε₂, ρ) = ½ Tr|ε₁(ρ) - ε₂(ρ)|

    Зависит от входного состояния!
    Для полной характеристики нужна diamond distance

    Args:
        channel1: Первый канал
        channel2: Второй канал
        state: Входное состояние (если None, используется |0...0⟩)

    Returns:
        Trace distance для данного состояния
    """
    from ..core.states import QuantumState

    if channel1.n_qubits != channel2.n_qubits:
        raise ValueError("Каналы должны иметь одинаковое число кубитов")

    # Входное состояние
    if state is None:
        state = QuantumState.zero_state(channel1.n_qubits)

    rho = state.to_density_matrix()

    # Применяем каналы
    rho1 = channel1.apply(rho)
    rho2 = channel2.apply(rho)

    # Trace distance между выходами
    return trace_distance_states(rho1.matrix, rho2.matrix)


def diamond_distance(channel1, channel2,
                    use_sdp: bool = False) -> float:
    """
    Diamond distance (diamond norm) между каналами

    D◇(ε₁, ε₂) = sup_ρ ½ ||(ε₁ - ε₂) ⊗ I(ρ)||₁

    Максимальная trace distance по всем входным состояниям
    (включая запутанные с окружением)

    Это операционально значимая метрика - максимальная различимость каналов

    Точное вычисление требует SDP (semidefinite programming)
    Здесь используем приближение через Choi matrices

    Args:
        channel1: Первый канал
        channel2: Второй канал
        use_sdp: Использовать ли точное SDP решение (требует CVXPY)

    Returns:
        Diamond distance (приближение или точное значение)
    """
    if channel1.n_qubits != channel2.n_qubits:
        raise ValueError("Каналы должны иметь одинаковое число кубитов")

    if use_sdp:
        try:
            return _diamond_distance_sdp(channel1, channel2)
        except ImportError:
            print("CVXPY не установлен, используем приближение")
            use_sdp = False

    # Приближение через спектральную норму Choi matrix
    choi1 = channel1.get_choi_matrix()
    choi2 = channel2.get_choi_matrix()

    choi_diff = choi1 - choi2

    # Спектральная норма (наибольшее сингулярное значение)
    singular_values = np.linalg.svd(choi_diff, compute_uv=False)
    spectral_norm = singular_values[0]

    # Diamond distance ≈ спектральная норма (для некоторых случаев)
    # Точная формула: D◇ = ||Choi_diff||₁
    # ||A||₁ = сумма сингулярных значений

    diamond_approx = np.sum(singular_values) / 2

    return diamond_approx


def _diamond_distance_sdp(channel1, channel2) -> float:
    """
    Точное вычисление diamond distance через SDP

    Формулируется как задача semidefinite programming:

    maximize  Tr(Y)
    subject to [[Choi_diff, Y],
                [Y†, I]] ≥ 0

    где Choi_diff = J₁ - J₂

    Args:
        channel1: Первый канал
        channel2: Второй канал

    Returns:
        Точная diamond distance
    """
    import cvxpy as cp

    choi1 = channel1.get_choi_matrix()
    choi2 = channel2.get_choi_matrix()
    choi_diff = choi1 - choi2

    dim = choi_diff.shape[0]

    # Переменная оптимизации
    Y = cp.Variable((dim, dim), hermitian=True)

    # Блочная матрица для SDP ограничения
    block_matrix = cp.bmat([
        [choi_diff, Y],
        [Y.H, np.eye(dim)]
    ])

    # Ограничения
    constraints = [block_matrix >> 0]  # Положительно полуопределённая

    # Целевая функция: maximize Tr(Y)
    objective = cp.Maximize(cp.real(cp.trace(Y)))

    # Решение
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
        print(f"Предупреждение: SDP не сошлось оптимально (status: {problem.status})")
        # Возвращаем приближение
        return frobenius_distance(choi1, choi2)

    diamond_dist = problem.value

    return diamond_dist


def bures_distance(rho1: NDArray[np.complex128],
                   rho2: NDArray[np.complex128]) -> float:
    """
    Bures distance между квантовыми состояниями

    D_B(ρ₁, ρ₂) = √(2(1 - √F(ρ₁, ρ₂)))

    где F - fidelity

    Связана с trace distance и fidelity

    Args:
        rho1: Первое состояние
        rho2: Второе состояние

    Returns:
        Bures distance
    """
    from .fidelity import state_fidelity

    F = state_fidelity(rho1, rho2)
    bures = np.sqrt(2 * (1 - np.sqrt(F)))

    return bures


def hilbert_schmidt_distance(choi1: NDArray[np.complex128],
                             choi2: NDArray[np.complex128]) -> float:
    """
    Hilbert-Schmidt distance (то же что Frobenius)

    Альтернативное название для удобства
    """
    return frobenius_distance(choi1, choi2)
