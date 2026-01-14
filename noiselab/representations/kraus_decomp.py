"""
Разложение и анализ операторов Крауса
"""

import numpy as np
from typing import List, Tuple
from numpy.typing import NDArray


def kraus_decomposition(choi_matrix: NDArray[np.complex128],
                       n_qubits: int) -> List[NDArray[np.complex128]]:
    """
    Разложение Choi matrix в операторы Крауса

    J = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|
    Kᵢ = √λᵢ reshape(|vᵢ⟩, (d, d))

    Args:
        choi_matrix: Choi matrix размера d²×d²
        n_qubits: Число кубитов

    Returns:
        Список операторов Крауса
    """
    dim = 2 ** n_qubits

    # Диагонализация
    eigenvalues, eigenvectors = np.linalg.eigh(choi_matrix)

    # Оставляем только положительные собственные значения
    threshold = 1e-12
    positive_indices = eigenvalues > threshold

    eigenvalues = eigenvalues[positive_indices]
    eigenvectors = eigenvectors[:, positive_indices]

    # Формируем операторы Крауса
    kraus_operators = []
    for i in range(len(eigenvalues)):
        sqrt_eigenvalue = np.sqrt(eigenvalues[i])
        vec = eigenvectors[:, i]
        K = sqrt_eigenvalue * vec.reshape(dim, dim)
        kraus_operators.append(K)

    return kraus_operators


def minimize_kraus_rank(kraus_operators: List[NDArray[np.complex128]],
                       threshold: float = 1e-10) -> List[NDArray[np.complex128]]:
    """
    Минимизировать число операторов Крауса

    Находит представление с минимальным числом операторов
    через диагонализацию Choi matrix

    Args:
        kraus_operators: Исходные операторы Крауса
        threshold: Порог для отсечения малых собственных значений

    Returns:
        Минимальный набор операторов Крауса
    """
    if len(kraus_operators) == 0:
        return []

    dim = kraus_operators[0].shape[0]
    n_qubits = int(np.log2(dim))

    # Строим Choi matrix
    choi_dim = dim ** 2
    choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

    for K in kraus_operators:
        vec_K = K.reshape(-1, 1)
        choi += vec_K @ vec_K.conj().T

    # Диагонализация
    eigenvalues, eigenvectors = np.linalg.eigh(choi)

    # Оставляем значимые собственные значения
    significant_indices = eigenvalues > threshold

    eigenvalues = eigenvalues[significant_indices]
    eigenvectors = eigenvectors[:, significant_indices]

    # Новые операторы Крауса
    minimal_kraus = []
    for i in range(len(eigenvalues)):
        sqrt_eigenvalue = np.sqrt(eigenvalues[i])
        vec = eigenvectors[:, i]
        K = sqrt_eigenvalue * vec.reshape(dim, dim)
        minimal_kraus.append(K)

    return minimal_kraus


def analyze_kraus_structure(kraus_operators: List[NDArray[np.complex128]]) -> dict:
    """
    Анализ структуры операторов Крауса

    Определяет характерные свойства:
    - Ранг каждого оператора
    - Относительные веса (из нормы)
    - Унитарность каждого оператора
    - Паули-разложение (для 1 кубита)

    Args:
        kraus_operators: Список операторов Крауса

    Returns:
        Словарь с результатами анализа
    """
    if len(kraus_operators) == 0:
        return {"error": "Empty operator list"}

    dim = kraus_operators[0].shape[0]
    n_qubits = int(np.log2(dim))

    analysis = {
        "n_operators": len(kraus_operators),
        "n_qubits": n_qubits,
        "operators_info": []
    }

    # Анализируем каждый оператор
    for i, K in enumerate(kraus_operators):
        # Норма K†K
        weight = np.trace(K.conj().T @ K).real

        # Ранг
        rank = np.linalg.matrix_rank(K)

        # Проверка унитарности (с точностью до скаляра)
        K_normalized = K / np.sqrt(weight) if weight > 1e-10 else K
        is_unitary = np.allclose(
            K_normalized.conj().T @ K_normalized,
            np.eye(dim),
            atol=1e-8
        )

        op_info = {
            "index": i,
            "weight": weight,
            "rank": rank,
            "is_unitary": is_unitary
        }

        # Для однокубитного случая: разложение по Паули
        if n_qubits == 1:
            pauli_decomp = _pauli_decomposition_1q(K)
            op_info["pauli_decomposition"] = pauli_decomp

        analysis["operators_info"].append(op_info)

    # Общая проверка: Σᵢ Kᵢ†Kᵢ = I
    sum_kraus = sum(K.conj().T @ K for K in kraus_operators)
    identity = np.eye(dim, dtype=np.complex128)

    trace_preserving_error = np.linalg.norm(sum_kraus - identity)
    analysis["trace_preserving_error"] = trace_preserving_error
    analysis["is_trace_preserving"] = trace_preserving_error < 1e-8

    return analysis


def _pauli_decomposition_1q(operator: NDArray[np.complex128]) -> dict:
    """
    Разложить однокубитный оператор по базису Паули

    A = a₀I + a₁X + a₂Y + a₃Z

    где aᵢ = Tr(A σᵢ) / 2

    Args:
        operator: Матрица 2×2

    Returns:
        Словарь коэффициентов {I, X, Y, Z}
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    paulis = {"I": I, "X": X, "Y": Y, "Z": Z}

    decomposition = {}
    for name, pauli in paulis.items():
        coeff = np.trace(operator @ pauli) / 2
        decomposition[name] = complex(coeff)

    return decomposition


def compare_kraus_representations(kraus1: List[NDArray[np.complex128]],
                                  kraus2: List[NDArray[np.complex128]]) -> float:
    """
    Сравнить два набора операторов Крауса

    Операторы Крауса не уникальны (унитарная свобода)
    Сравниваем через Choi matrices

    Args:
        kraus1: Первый набор операторов
        kraus2: Второй набор операторов

    Returns:
        Frobenius distance между Choi matrices
    """
    if len(kraus1) == 0 or len(kraus2) == 0:
        return float('inf')

    dim = kraus1[0].shape[0]
    choi_dim = dim ** 2

    # Строим Choi matrices
    choi1 = np.zeros((choi_dim, choi_dim), dtype=np.complex128)
    for K in kraus1:
        vec_K = K.reshape(-1, 1)
        choi1 += vec_K @ vec_K.conj().T

    choi2 = np.zeros((choi_dim, choi_dim), dtype=np.complex128)
    for K in kraus2:
        vec_K = K.reshape(-1, 1)
        choi2 += vec_K @ vec_K.conj().T

    # Frobenius distance
    distance = np.linalg.norm(choi1 - choi2, ord='fro')

    return distance
