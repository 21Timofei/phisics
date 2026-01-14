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


def partial_transpose(rho: NDArray[np.complex128],
                      dims: List[int],
                      transpose: Union[int, List[int]]) -> NDArray[np.complex128]:
    """
    Частичное транспонирование (partial transpose)
    Используется для проверки запутанности (PPT criterion)

    Args:
        rho: Матрица плотности
        dims: Размерности подсистем
        transpose: Индекс(ы) подсистемы для транспонирования

    Returns:
        Частично транспонированная матрица
    """
    if isinstance(transpose, int):
        transpose = [transpose]

    n_subsystems = len(dims)
    total_dim = int(np.prod(dims))

    if rho.shape != (total_dim, total_dim):
        raise ValueError(f"Размерность матрицы {rho.shape} не соответствует dims={dims}")

    # Reshape в тензорную форму
    shape = dims + dims
    rho_tensor = rho.reshape(shape)

    # Транспонируем указанные подсистемы
    axes = list(range(2 * n_subsystems))
    for idx in transpose:
        if idx < 0 or idx >= n_subsystems:
            raise ValueError(f"Индекс {idx} вне диапазона")
        # Меняем местами idx и idx+n_subsystems
        axes[idx], axes[idx + n_subsystems] = axes[idx + n_subsystems], axes[idx]

    rho_pt = np.transpose(rho_tensor, axes)

    # Reshape обратно
    result = rho_pt.reshape(total_dim, total_dim)

    return result


def schmidt_decomposition(state_vector: NDArray[np.complex128],
                          dim_A: int,
                          dim_B: int) -> tuple:
    """
    Разложение Шмидта для чистого состояния |ψ⟩_AB

    |ψ⟩ = Σᵢ √λᵢ |aᵢ⟩|bᵢ⟩

    где λᵢ - коэффициенты Шмидта (собственные значения ρ_A или ρ_B)

    Args:
        state_vector: Вектор состояния системы AB
        dim_A: Размерность подсистемы A
        dim_B: Размерность подсистемы B

    Returns:
        (schmidt_coefficients, basis_A, basis_B)
        schmidt_coefficients: √λᵢ
        basis_A: базисные векторы в A
        basis_B: базисные векторы в B
    """
    if len(state_vector) != dim_A * dim_B:
        raise ValueError(f"Размерность вектора {len(state_vector)} != {dim_A}×{dim_B}")

    # Reshape в матрицу dim_A × dim_B
    psi_matrix = state_vector.reshape(dim_A, dim_B)

    # SVD разложение: ψ = U Σ V†
    U, sigma, Vh = np.linalg.svd(psi_matrix, full_matrices=False)

    # Коэффициенты Шмидта
    schmidt_coefficients = sigma[sigma > 1e-15]  # Убираем нулевые

    # Базисы Шмидта
    rank = len(schmidt_coefficients)
    basis_A = U[:, :rank]
    basis_B = Vh.conj().T[:, :rank]

    return schmidt_coefficients, basis_A, basis_B


def entanglement_entropy(state_vector: NDArray[np.complex128],
                         dim_A: int,
                         dim_B: int) -> float:
    """
    Энтропия запутанности (entanglement entropy)
    S = -Σᵢ λᵢ log₂(λᵢ)
    где λᵢ - коэффициенты Шмидта в квадрате

    S = 0 для сепарабельных состояний
    S = log₂(min(dim_A, dim_B)) для максимально запутанных
    """
    schmidt_coeffs, _, _ = schmidt_decomposition(state_vector, dim_A, dim_B)

    # λᵢ = (коэффициент Шмидта)²
    eigenvalues = schmidt_coeffs ** 2
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))

    return entropy


def apply_single_qubit_gate(gate: NDArray[np.complex128],
                            qubit_index: int,
                            n_qubits: int) -> NDArray[np.complex128]:
    """
    Применить однокубитный гейт к указанному кубиту в многокубитной системе

    Результат: I ⊗ ... ⊗ I ⊗ gate ⊗ I ⊗ ... ⊗ I
                               ↑
                          qubit_index

    Args:
        gate: Однокубитная матрица 2×2
        qubit_index: Индекс кубита (0-based)
        n_qubits: Общее число кубитов

    Returns:
        Полная матрица гейта размера 2^n × 2^n
    """
    if gate.shape != (2, 2):
        raise ValueError("Гейт должен быть матрицей 2×2")

    if qubit_index < 0 or qubit_index >= n_qubits:
        raise ValueError(f"Индекс кубита {qubit_index} вне диапазона [0, {n_qubits-1}]")

    I = np.eye(2, dtype=np.complex128)

    # Создаём список матриц для тензорного произведения
    matrices = []
    for i in range(n_qubits):
        if i == qubit_index:
            matrices.append(gate)
        else:
            matrices.append(I)

    return tensor_product(*matrices)


def apply_two_qubit_gate(gate: NDArray[np.complex128],
                         qubit_indices: tuple,
                         n_qubits: int) -> NDArray[np.complex128]:
    """
    Применить двухкубитный гейт к указанным кубитам

    Примечание: для некоторых индексов требуется перестановка кубитов

    Args:
        gate: Двухкубитная матрица 4×4
        qubit_indices: (control, target) или (qubit1, qubit2)
        n_qubits: Общее число кубитов

    Returns:
        Полная матрица гейта
    """
    if gate.shape != (4, 4):
        raise ValueError("Гейт должен быть матрицей 4×4")

    q1, q2 = qubit_indices

    if q1 < 0 or q1 >= n_qubits or q2 < 0 or q2 >= n_qubits:
        raise ValueError(f"Индексы кубитов вне диапазона")

    if q1 == q2:
        raise ValueError("Индексы кубитов должны быть различными")

    # Упрощённая реализация для соседних кубитов
    # Для общего случая нужна более сложная логика с SWAP
    if abs(q1 - q2) != 1:
        raise NotImplementedError("Пока реализованы только соседние кубиты")

    I = np.eye(2, dtype=np.complex128)

    # Определяем позицию двухкубитного гейта
    min_q = min(q1, q2)

    matrices = []
    i = 0
    while i < n_qubits:
        if i == min_q:
            matrices.append(gate)
            i += 2  # Пропускаем следующий кубит
        else:
            matrices.append(I)
            i += 1

    return tensor_product(*matrices)
