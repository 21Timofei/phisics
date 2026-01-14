"""
Модуль квантовых измерений
Реализация проективных измерений и симуляции шума
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from numpy.typing import NDArray
from .states import QuantumState, DensityMatrix
from .gates import PauliGates


class Measurement:
    """
    Квантовое измерение как набор проекторов (POVM)
    {Πᵢ}, где Σᵢ Πᵢ = I

    Для проективного измерения: Πᵢ = |i⟩⟨i|
    Вероятность результата i: P(i) = Tr(Πᵢ ρ)
    Пост-измерительное состояние: ρ' = Πᵢ ρ Πᵢ / P(i)
    """

    def __init__(self, projectors: List[NDArray[np.complex128]],
                 labels: Optional[List[str]] = None):
        """
        Args:
            projectors: Список проекторов {Πᵢ}
            labels: Метки для результатов измерений
        """
        self.projectors = [np.array(p, dtype=np.complex128) for p in projectors]

        # Проверка полноты: Σᵢ Πᵢ = I
        sum_proj = sum(self.projectors)
        dim = len(sum_proj)
        identity = np.eye(dim, dtype=np.complex128)

        if not np.allclose(sum_proj, identity, atol=1e-10):
            raise ValueError("Проекторы не образуют полное множество: Σᵢ Πᵢ ≠ I")

        # Проверка, что проекторы эрмитовы и идемпотентны
        for i, proj in enumerate(self.projectors):
            if not np.allclose(proj, proj.conj().T, atol=1e-10):
                raise ValueError(f"Проектор {i} не эрмитов")
            if not np.allclose(proj @ proj, proj, atol=1e-10):
                raise ValueError(f"Проектор {i} не идемпотентен: Π² ≠ Π")

        self.n_outcomes = len(projectors)
        self.labels = labels if labels else [str(i) for i in range(self.n_outcomes)]

    def probabilities(self, rho: DensityMatrix) -> NDArray[np.float64]:
        """
        Вычислить вероятности всех результатов: P(i) = Tr(Πᵢ ρ)
        """
        probs = np.array([
            np.trace(proj @ rho.matrix).real
            for proj in self.projectors
        ])

        # Численная стабильность
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()  # Ренормализация

        return probs

    def measure(self, rho: DensityMatrix, shots: int = 1) -> Dict[str, int]:
        """
        Симуляция измерения с конечным числом shots

        Args:
            rho: Квантовое состояние
            shots: Число измерений

        Returns:
            Словарь {label: count} с подсчётом результатов
        """
        probs = self.probabilities(rho)
        outcomes = np.random.choice(self.n_outcomes, size=shots, p=probs)

        counts = {label: 0 for label in self.labels}
        for outcome in outcomes:
            counts[self.labels[outcome]] += 1

        return counts

    def expectation_value(self, rho: DensityMatrix,
                         eigenvalues: Optional[NDArray[np.float64]] = None) -> float:
        """
        Среднее значение наблюдаемой: ⟨O⟩ = Σᵢ λᵢ P(i)

        Args:
            rho: Квантовое состояние
            eigenvalues: Собственные значения наблюдаемой (по умолчанию: 0, 1, 2, ...)
        """
        if eigenvalues is None:
            eigenvalues = np.arange(self.n_outcomes, dtype=np.float64)

        probs = self.probabilities(rho)
        return np.dot(eigenvalues, probs)

    def post_measurement_state(self, rho: DensityMatrix, outcome: int) -> DensityMatrix:
        """
        Состояние после измерения с результатом outcome:
        ρ' = Πᵢ ρ Πᵢ / P(i)
        """
        proj = self.projectors[outcome]
        new_rho = proj @ rho.matrix @ proj

        prob = np.trace(new_rho).real
        if prob < 1e-15:
            raise ValueError(f"Результат {outcome} имеет нулевую вероятность")

        new_rho /= prob

        return DensityMatrix(new_rho, validate=False)

    @classmethod
    def computational_basis(cls, n_qubits: int) -> 'Measurement':
        """
        Измерение в вычислительном базисе {|0⟩, |1⟩, ..., |2^n-1⟩}
        """
        dim = 2 ** n_qubits
        projectors = []
        labels = []

        for i in range(dim):
            # Проектор |i⟩⟨i|
            proj = np.zeros((dim, dim), dtype=np.complex128)
            proj[i, i] = 1.0
            projectors.append(proj)

            # Метка в двоичном виде
            binary = format(i, f'0{n_qubits}b')
            labels.append(binary)

        return cls(projectors, labels)


class PauliMeasurement:
    """
    Измерение в базисе Паули (X, Y, Z)
    Используется в квантовой томографии
    """

    def __init__(self, basis: str, qubit_index: int = 0, n_qubits: int = 1):
        """
        Args:
            basis: 'X', 'Y' или 'Z'
            qubit_index: Индекс кубита для измерения
            n_qubits: Общее число кубитов
        """
        self.basis = basis.upper()
        self.qubit_index = qubit_index
        self.n_qubits = n_qubits

        if self.basis not in ['X', 'Y', 'Z']:
            raise ValueError(f"Неизвестный базис: {basis}")

        # Собственные векторы и значения паули-матриц
        if self.basis == 'X':
            # X: собственные векторы |+⟩ (λ=+1) и |-⟩ (λ=-1)
            self.eigenvectors = [
                np.array([1, 1], dtype=np.complex128) / np.sqrt(2),   # |+⟩
                np.array([1, -1], dtype=np.complex128) / np.sqrt(2)   # |-⟩
            ]
            self.eigenvalues = np.array([1, -1])
            self.labels = ['+', '-']

        elif self.basis == 'Y':
            # Y: собственные векторы |+i⟩ (λ=+1) и |-i⟩ (λ=-1)
            self.eigenvectors = [
                np.array([1, 1j], dtype=np.complex128) / np.sqrt(2),  # |+i⟩
                np.array([1, -1j], dtype=np.complex128) / np.sqrt(2)  # |-i⟩
            ]
            self.eigenvalues = np.array([1, -1])
            self.labels = ['+i', '-i']

        else:  # Z
            # Z: собственные векторы |0⟩ (λ=+1) и |1⟩ (λ=-1)
            self.eigenvectors = [
                np.array([1, 0], dtype=np.complex128),  # |0⟩
                np.array([0, 1], dtype=np.complex128)   # |1⟩
            ]
            self.eigenvalues = np.array([1, -1])
            self.labels = ['0', '1']

        # Проекторы для однокубитного случая
        self._create_projectors()

    def _create_projectors(self):
        """Создать проекторы для измерения"""
        single_qubit_projectors = [
            np.outer(vec, vec.conj())
            for vec in self.eigenvectors
        ]

        if self.n_qubits == 1:
            self.projectors = single_qubit_projectors
        else:
            # Для многокубитного случая: расширяем проекторы тензорным произведением
            I = np.eye(2, dtype=np.complex128)

            # Проекторы для других кубитов (суммируем по всем базисам)
            other_identity = np.eye(2 ** (self.n_qubits - 1), dtype=np.complex128)

            self.projectors = []
            for proj in single_qubit_projectors:
                if self.qubit_index == 0:
                    # Проектор действует на первый кубит
                    full_proj = np.kron(proj, other_identity)
                elif self.qubit_index == self.n_qubits - 1:
                    # Проектор действует на последний кубит
                    full_proj = np.kron(other_identity, proj)
                else:
                    # Проектор действует на промежуточный кубит
                    left_dim = 2 ** self.qubit_index
                    right_dim = 2 ** (self.n_qubits - self.qubit_index - 1)
                    I_left = np.eye(left_dim, dtype=np.complex128)
                    I_right = np.eye(right_dim, dtype=np.complex128)
                    full_proj = np.kron(np.kron(I_left, proj), I_right)

                self.projectors.append(full_proj)

    def measure(self, rho: DensityMatrix, shots: int = 1) -> Dict[str, int]:
        """Провести измерение в базисе Паули"""
        probs = self._compute_probabilities(rho)

        # Симуляция измерений
        outcomes = np.random.choice(2, size=shots, p=probs)

        counts = {self.labels[0]: 0, self.labels[1]: 0}
        for outcome in outcomes:
            counts[self.labels[outcome]] += 1

        return counts

    def _compute_probabilities(self, rho: DensityMatrix) -> NDArray[np.float64]:
        """Вычислить вероятности двух результатов"""
        probs = np.array([
            np.trace(proj @ rho.matrix).real
            for proj in self.projectors
        ])

        # Численная стабильность
        probs = np.clip(probs, 0, 1)
        probs /= probs.sum()

        return probs

    def expectation_value(self, rho: DensityMatrix) -> float:
        """
        Среднее значение паули-оператора: ⟨σ⟩ = P₊ - P₋
        где P₊ и P₋ - вероятности результатов +1 и -1
        """
        probs = self._compute_probabilities(rho)
        return probs[0] - probs[1]  # (+1) * P₊ + (-1) * P₋

    @staticmethod
    def get_pauli_matrix(basis: str) -> NDArray[np.complex128]:
        """Получить матрицу Паули для данного базиса"""
        if basis.upper() == 'X':
            return PauliGates.X
        elif basis.upper() == 'Y':
            return PauliGates.Y
        elif basis.upper() == 'Z':
            return PauliGates.Z
        elif basis.upper() == 'I':
            return PauliGates.I
        else:
            raise ValueError(f"Неизвестный базис: {basis}")


class MeasurementWithNoise:
    """
    Измерение с учётом SPAM (State Preparation and Measurement) ошибок

    Модель ошибки считывания:
    P_measured(i) = (1-εᵢ) P_true(i) + Σⱼ≠ᵢ εⱼᵢ P_true(j)

    Для однокубитного случая:
    P(measure 0 | true 0) = 1 - ε₀
    P(measure 1 | true 0) = ε₀
    P(measure 0 | true 1) = ε₁
    P(measure 1 | true 1) = 1 - ε₁
    """

    def __init__(self, ideal_measurement: Measurement,
                 readout_error: float = 0.0,
                 confusion_matrix: Optional[NDArray[np.float64]] = None):
        """
        Args:
            ideal_measurement: Идеальное измерение без шума
            readout_error: Вероятность ошибки считывания (упрощённая модель)
            confusion_matrix: Матрица переходов P(measured|true) размера n×n
        """
        self.ideal_measurement = ideal_measurement
        self.readout_error = readout_error

        if confusion_matrix is not None:
            self.confusion_matrix = confusion_matrix
        else:
            # Создаём простую симметричную матрицу ошибок
            n = ideal_measurement.n_outcomes
            self.confusion_matrix = np.eye(n) * (1 - readout_error)
            self.confusion_matrix += (readout_error / (n - 1)) * (1 - np.eye(n))

        # Проверка стохастичности
        if not np.allclose(self.confusion_matrix.sum(axis=0), 1.0, atol=1e-10):
            raise ValueError("Столбцы матрицы confusion должны суммироваться в 1")

    def measure(self, rho: DensityMatrix, shots: int = 1) -> Dict[str, int]:
        """Измерение с шумом"""
        # Истинные вероятности
        true_probs = self.ideal_measurement.probabilities(rho)

        # Измеренные вероятности после применения ошибок
        measured_probs = self.confusion_matrix @ true_probs
        measured_probs = np.clip(measured_probs, 0, 1)
        measured_probs /= measured_probs.sum()

        # Симуляция измерений
        outcomes = np.random.choice(
            self.ideal_measurement.n_outcomes,
            size=shots,
            p=measured_probs
        )

        counts = {label: 0 for label in self.ideal_measurement.labels}
        for outcome in outcomes:
            counts[self.ideal_measurement.labels[outcome]] += 1

        return counts

    def mitigate_readout_error(self, noisy_counts: Dict[str, int]) -> Dict[str, int]:
        """
        Митигация (исправление) ошибок считывания
        Решаем: measured = M @ true → true = M⁻¹ @ measured
        """
        total_shots = sum(noisy_counts.values())

        # Вектор измеренных частот
        measured_probs = np.array([
            noisy_counts.get(label, 0) / total_shots
            for label in self.ideal_measurement.labels
        ])

        # Обратная задача: истинные вероятности
        try:
            inv_confusion = np.linalg.inv(self.confusion_matrix)
            true_probs = inv_confusion @ measured_probs

            # Проекция на симплекс (физические вероятности)
            true_probs = np.clip(true_probs, 0, 1)
            true_probs /= true_probs.sum()

            # Конвертируем обратно в counts
            corrected_counts = {
                label: int(prob * total_shots)
                for label, prob in zip(self.ideal_measurement.labels, true_probs)
            }

            return corrected_counts

        except np.linalg.LinAlgError:
            # Если матрица вырождена, возвращаем исходные данные
            return noisy_counts
