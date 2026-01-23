"""
Модуль тензорных операций для многокубитных систем
"""

import numpy as np
from typing import List, Union
from numpy.typing import NDArray


def tensor_product(*matrices: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Тензорное произведение матриц: A ⊗ B ⊗ C ⊗ ...

    Для векторов состояния:
    |ψ₁⟩ ⊗ |ψ₂⟩ = |ψ₁ψ₂⟩

    Args:
        matrices: Матрицы или векторы для тензорного произведения

    Returns:
        Результат тензорного произведения
    """
    if len(matrices) == 0:
        raise ValueError("Нужна хотя бы одна матрица")

    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)

    return result


def partial_trace(rho: NDArray[np.complex128],
                  dims: List[int],
                  trace_out: Union[int, List[int]]) -> NDArray[np.complex128]:
    """
    Частичный след (partial trace) многокубитной матрицы плотности

    Для системы AB: Tr_B(ρ_AB) = ρ_A

    Args:
        rho: Матрица плотности
        dims: Размерности подсистем [d₁, d₂, ..., dₙ]
        trace_out: Индекс(ы) подсистемы для вычисления следа (0-based)

    Returns:
        Матрица плотности редуцированной системы

    Example:
        # Для 2-кубитной системы: dims=[2, 2]
        # trace_out=1 даёт редуцированное состояние первого кубита
        rho_A = partial_trace(rho_AB, [2, 2], 1)
    """
    if isinstance(trace_out, int):
        trace_out = [trace_out]

    n_subsystems = len(dims)
    total_dim = int(np.prod(dims))

    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"Размерность матрицы {rho.shape} не соответствует dims={dims}")

    # Проверка индексов
    for idx in trace_out:
        if idx < 0 or idx >= n_subsystems:
            raise ValueError(f"Индекс {idx} вне диапазона [0, {n_subsystems-1}]")

    # Индексы подсистем, которые остаются
    keep_indices = [i for i in range(n_subsystems) if i not in trace_out]

    if len(keep_indices) == 0:
        # Если выбрасываем все подсистемы, возвращаем след (скаляр как 1x1 матрица)
        return np.array([[np.trace(rho)]], dtype=np.complex128)

    # Размерности оставшейся системы
    keep_dims = [dims[i] for i in keep_indices]
    reduced_dim = int(np.prod(keep_dims))

    # Reshape для работы с тензорной структурой
    # rho: (d₁...dₙ, d₁...dₙ) -> (d₁, ..., dₙ, d₁, ..., dₙ)
    shape = dims + dims
    rho_tensor = rho.reshape(shape)

    # Вычисляем след по ненужным индексам
    # Для каждой выбрасываемой подсистемы суммируем по диагонали
    for idx in sorted(trace_out, reverse=True):
        # idx в первой половине индексов, idx+n_subsystems во второй
        rho_tensor = np.trace(rho_tensor, axis1=idx, axis2=idx + n_subsystems)
        n_subsystems -= 1

    # Reshape обратно в матрицу
    result = rho_tensor.reshape(reduced_dim, reduced_dim)

    return result
