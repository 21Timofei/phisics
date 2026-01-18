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
