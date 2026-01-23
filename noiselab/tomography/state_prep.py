"""
Подготовка входных состояний для квантовой томографии
"""

import numpy as np
from typing import List
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import QuantumState, DensityMatrix



def get_pauli_basis() -> List[NDArray[np.complex128]]:
    """
    Получить базис Паули для 1 кубита

    Паули-строки образуют ортогональный базис для операторов
    Tr(σᵢ σⱼ) = 2 δᵢⱼ

    Используется для разложения состояний и каналов

    Returns:
        Список паули-матриц (4 оператора: I, X, Y, Z)
    """
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    return [I, X, Y, Z]


def decompose_in_pauli_basis(operator: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    Разложить оператор в базисе Паули

    A = Σᵢ aᵢ σᵢ

    где aᵢ = Tr(A σᵢ) / 2

    Args:
        operator: Оператор размера 2 × 2

    Returns:
        Вектор коэффициентов разложения (длина 4)
    """
    pauli_basis = get_pauli_basis()

    coefficients = np.zeros(4, dtype=np.complex128)

    for i, pauli in enumerate(pauli_basis):
        # aᵢ = Tr(A σᵢ) / 2
        coefficients[i] = np.trace(operator @ pauli) / 2

    return coefficients