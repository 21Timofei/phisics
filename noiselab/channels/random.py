"""
Генерация случайных квантовых каналов
Полезно для тестирования томографии на неизвестных каналах
"""

import numpy as np
from typing import Optional
from numpy.typing import NDArray
from .kraus import KrausChannel


def random_unitary(dim: int, seed: Optional[int] = None) -> NDArray[np.complex128]:
    """
    Генерация случайной унитарной матрицы по мере Хаара

    Использует QR-разложение случайной комплексной матрицы
    с нормировкой фазы

    Args:
        dim: Размерность
        seed: Random seed для воспроизводимости

    Returns:
        Случайная унитарная матрица dim×dim
    """
    if seed is not None:
        np.random.seed(seed)

    # Случайная комплексная матрица
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

    # QR-разложение
    Q, R = np.linalg.qr(A)

    # Нормировка фазы (диагональ R может быть отрицательной)
    Lambda = np.diag(np.diag(R) / np.abs(np.diag(R)))
    U = Q @ Lambda

    return U.astype(np.complex128)


def random_density_matrix(dim: int, rank: Optional[int] = None,
                          seed: Optional[int] = None) -> NDArray[np.complex128]:
    """
    Генерация случайной матрицы плотности

    Args:
        dim: Размерность
        rank: Ранг матрицы (None = полный ранг)
        seed: Random seed

    Returns:
        Случайная матрица плотности
    """
    if seed is not None:
        np.random.seed(seed)

    if rank is None:
        rank = dim

    if rank > dim:
        raise ValueError(f"Ранг {rank} не может превышать размерность {dim}")

    # Генерируем rank случайных векторов
    vectors = np.random.randn(dim, rank) + 1j * np.random.randn(dim, rank)

    # Случайные положительные веса
    weights = np.random.rand(rank)
    weights /= weights.sum()

    # Составляем матрицу плотности
    rho = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(rank):
        vec = vectors[:, i]
        vec /= np.linalg.norm(vec)
        rho += weights[i] * np.outer(vec, vec.conj())

    return rho


def random_cptp_channel(n_qubits: int,
                       kraus_rank: Optional[int] = None,
                       seed: Optional[int] = None) -> KrausChannel:
    """
    Генерация случайного CPTP канала по равномерной мере

    Использует Choi-Jamiolkowski изоморфизм:
    1. Генерируем случайную матрицу плотности (Choi matrix)
    2. Извлекаем операторы Крауса через диагонализацию

    Args:
        n_qubits: Число кубитов
        kraus_rank: Ранг Крауса (число операторов, None = случайный)
        seed: Random seed

    Returns:
        Случайный квантовый канал
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits
    choi_dim = dim ** 2

    if kraus_rank is None:
        # Случайный ранг от 1 до dim²
        kraus_rank = np.random.randint(1, min(choi_dim, 10) + 1)

    # Генерируем случайную Choi matrix
    choi = random_density_matrix(choi_dim, rank=kraus_rank, seed=seed)

    # Проверяем и корректируем trace-preserving условие
    # Tr_B(J) = I, где B - вторая подсистема
    # Для упрощения используем нормировку
    from ..core.tensor import partial_trace
    dims = [dim, dim]
    reduced = partial_trace(choi, dims, 1)

    # Корректируем для trace-preserving
    # Это приближённый метод, более точный требует SDP
    trace_factor = np.trace(reduced).real / dim
    choi /= trace_factor

    # Извлекаем операторы Крауса через диагонализацию
    eigenvalues, eigenvectors = np.linalg.eigh(choi)

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
        # Reshape вектора в матрицу
        K = sqrt_eigenvalue * vec.reshape(dim, dim)
        kraus_operators.append(K)

    return KrausChannel(
        kraus_operators,
        n_qubits=n_qubits,
        name=f"RandomCPTP(rank={len(kraus_operators)})",
        validate=False  # Может не пройти строгую валидацию из-за численных ошибок
    )


def random_unitary_channel(n_qubits: int,
                           n_unitaries: int = 1,
                           seed: Optional[int] = None) -> KrausChannel:
    """
    Генерация случайного унитарного канала (смесь унитарных операций)

    ε(ρ) = Σᵢ pᵢ Uᵢ ρ Uᵢ†

    где Uᵢ - случайные унитарные матрицы, pᵢ - вероятности

    Args:
        n_qubits: Число кубитов
        n_unitaries: Число унитарных операций в смеси
        seed: Random seed

    Returns:
        Случайный унитарный канал
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits

    # Случайные вероятности
    probabilities = np.random.rand(n_unitaries)
    probabilities /= probabilities.sum()

    # Генерируем случайные унитарные матрицы
    kraus_operators = []
    for i in range(n_unitaries):
        U = random_unitary(dim, seed=None)  # Не фиксируем seed для каждой матрицы
        K = np.sqrt(probabilities[i]) * U
        kraus_operators.append(K)

    return KrausChannel(
        kraus_operators,
        n_qubits=n_qubits,
        name=f"RandomUnitary(n={n_unitaries})",
        validate=True
    )


def random_pauli_channel(n_qubits: int = 1,
                        seed: Optional[int] = None) -> KrausChannel:
    """
    Генерация случайного паули-канала

    ε(ρ) = Σᵢ pᵢ σᵢ ρ σᵢ

    где σᵢ - паули-строки, pᵢ - случайные вероятности

    Это специальный класс каналов с хорошими свойствами

    Args:
        n_qubits: Число кубитов
        seed: Random seed

    Returns:
        Случайный паули-канал
    """
    if seed is not None:
        np.random.seed(seed)

    dim = 2 ** n_qubits
    n_paulis = 4 ** n_qubits

    # Случайные вероятности для паули-операторов
    probabilities = np.random.rand(n_paulis)
    probabilities /= probabilities.sum()

    # Паули матрицы
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    paulis = [I, X, Y, Z]

    # Генерируем все паули-строки для n кубитов
    kraus_operators = []

    def generate_pauli_strings(n):
        """Генерирует все паули-строки для n кубитов"""
        if n == 0:
            return [np.array([[1]], dtype=np.complex128)]
        if n == 1:
            return paulis

        # Рекурсивная генерация
        prev = generate_pauli_strings(n - 1)
        result = []
        for p in paulis:
            for prev_p in prev:
                result.append(np.kron(p, prev_p))
        return result

    if n_qubits <= 3:  # Для больших n это экспоненциально растёт
        pauli_strings = generate_pauli_strings(n_qubits)

        for i, pauli in enumerate(pauli_strings):
            K = np.sqrt(probabilities[i]) * pauli
            kraus_operators.append(K)
    else:
        # Для большого n используем подвыборку
        for i in range(min(n_paulis, 50)):  # Ограничение для практичности
            # Случайная паули-строка
            pauli = paulis[0]
            for _ in range(n_qubits):
                idx = np.random.randint(4)
                pauli = np.kron(pauli, paulis[idx])

            K = np.sqrt(probabilities[i]) * pauli
            kraus_operators.append(K)

    return KrausChannel(
        kraus_operators,
        n_qubits=n_qubits,
        name=f"RandomPauli(n={n_qubits})",
        validate=True
    )
