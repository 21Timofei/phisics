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


def prepare_tomography_states(n_qubits: int) -> List[DensityMatrix]:
    """
    Подготовка полного набора входных состояний для QPT

    Для n кубитов нужен базис из d² = 4^n состояний

    Для 1 кубита используем 6 состояний (избыточный набор):
    |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩

    Для n кубитов: все тензорные произведения базисных состояний

    Args:
        n_qubits: Число кубитов

    Returns:
        Список матриц плотности для томографии
    """
    if n_qubits == 1:
        return _prepare_single_qubit_states()
    else:
        return _prepare_multi_qubit_states(n_qubits)


def _prepare_single_qubit_states() -> List[DensityMatrix]:
    """
    Базисные состояния для томографии одного кубита

    Минимальный набор: 4 состояния образующих базис
    Избыточный набор: 6 состояний (лучше для устойчивости)

    Returns:
        Список из 6 состояний
    """
    # Вычислительный базис
    ket_0 = np.array([1, 0], dtype=np.complex128)
    ket_1 = np.array([0, 1], dtype=np.complex128)

    # X-базис (суперпозиции)
    ket_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)   # (|0⟩+|1⟩)/√2
    ket_minus = np.array([1, -1], dtype=np.complex128) / np.sqrt(2)  # (|0⟩-|1⟩)/√2

    # Y-базис (комплексные суперпозиции)
    ket_plus_i = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)   # (|0⟩+i|1⟩)/√2
    ket_minus_i = np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)  # (|0⟩-i|1⟩)/√2

    states = [ket_0, ket_1, ket_plus, ket_minus, ket_plus_i, ket_minus_i]

    # Преобразуем в матрицы плотности
    rho_list = []
    for ket in states:
        state = QuantumState(ket, normalize=False)
        rho_list.append(state.to_density_matrix())

    return rho_list


def _prepare_multi_qubit_states(n_qubits: int) -> List[DensityMatrix]:
    """
    Базисные состояния для томографии n кубитов

    Используем тензорные произведения однокубитных базисных состояний
    Для эффективности используем минимальный базис: |0⟩, |1⟩, |+⟩, |+i⟩

    Всего: 4^n состояний

    Args:
        n_qubits: Число кубитов

    Returns:
        Список матриц плотности
    """
    # Базисные векторы для одного кубита
    single_qubit_basis = [
        np.array([1, 0], dtype=np.complex128),              # |0⟩
        np.array([0, 1], dtype=np.complex128),              # |1⟩
        np.array([1, 1], dtype=np.complex128) / np.sqrt(2), # |+⟩
        np.array([1, 1j], dtype=np.complex128) / np.sqrt(2) # |+i⟩
    ]

    # Генерируем все комбинации
    from itertools import product

    multi_qubit_states = []

    for indices in product(range(4), repeat=n_qubits):
        # Тензорное произведение базисных состояний
        state_vector = single_qubit_basis[indices[0]]
        for i in range(1, n_qubits):
            state_vector = np.kron(state_vector, single_qubit_basis[indices[i]])

        # Создаём матрицу плотности
        state = QuantumState(state_vector, normalize=False)
        multi_qubit_states.append(state.to_density_matrix())

    return multi_qubit_states


def get_pauli_basis(n_qubits: int) -> List[NDArray[np.complex128]]:
    """
    Получить базис Паули для n кубитов

    Паули-строки образуют ортогональный базис для операторов
    Tr(σᵢ σⱼ) = 2^n δᵢⱼ

    Используется для разложения состояний и каналов

    Args:
        n_qubits: Число кубитов

    Returns:
        Список паули-матриц (4^n операторов)
    """
    # Одно кубитные паули матрицы
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    single_paulis = [I, X, Y, Z]

    if n_qubits == 1:
        return single_paulis

    # Генерируем паули-строки рекурсивно
    from itertools import product

    pauli_basis = []
    for indices in product(range(4), repeat=n_qubits):
        pauli_string = single_paulis[indices[0]]
        for i in range(1, n_qubits):
            pauli_string = np.kron(pauli_string, single_paulis[indices[i]])
        pauli_basis.append(pauli_string)

    return pauli_basis


def decompose_in_pauli_basis(operator: NDArray[np.complex128],
                            n_qubits: int) -> NDArray[np.complex128]:
    """
    Разложить оператор в базисе Паули

    A = Σᵢ aᵢ σᵢ

    где aᵢ = Tr(A σᵢ) / 2^n

    Args:
        operator: Оператор размера 2^n × 2^n
        n_qubits: Число кубитов

    Returns:
        Вектор коэффициентов разложения (длина 4^n)
    """
    pauli_basis = get_pauli_basis(n_qubits)
    dim = 2 ** n_qubits

    coefficients = np.zeros(len(pauli_basis), dtype=np.complex128)

    for i, pauli in enumerate(pauli_basis):
        # aᵢ = Tr(A σᵢ) / 2^n
        coefficients[i] = np.trace(operator @ pauli) / dim

    return coefficients


def reconstruct_from_pauli(coefficients: NDArray[np.complex128],
                          n_qubits: int) -> NDArray[np.complex128]:
    """
    Восстановить оператор из коэффициентов разложения Паули

    A = Σᵢ aᵢ σᵢ

    Args:
        coefficients: Коэффициенты разложения
        n_qubits: Число кубитов

    Returns:
        Оператор размера 2^n × 2^n
    """
    pauli_basis = get_pauli_basis(n_qubits)

    if len(coefficients) != len(pauli_basis):
        raise ValueError(f"Неправильное число коэффициентов: {len(coefficients)} != {len(pauli_basis)}")

    dim = 2 ** n_qubits
    operator = np.zeros((dim, dim), dtype=np.complex128)

    for coeff, pauli in zip(coefficients, pauli_basis):
        operator += coeff * pauli

    return operator


def fidelity_of_states(rho1: DensityMatrix, rho2: DensityMatrix) -> float:
    """
    Верность между двумя состояниями
    Обёртка для удобства
    """
    return rho1.fidelity(rho2)
