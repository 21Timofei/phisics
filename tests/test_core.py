"""
Тесты для core модулей: states, gates, measurements
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noiselab.core.states import QuantumState, DensityMatrix
from noiselab.core.gates import PauliGates, RotationGates, TwoQubitGates
from noiselab.core.measurements import PauliMeasurement


class TestQuantumState:
    """Тесты для QuantumState"""

    def test_normalization(self):
        """Тест нормализации состояния"""
        state = QuantumState([1, 1], normalize=True)
        norm = np.linalg.norm(state.statevector)
        assert np.isclose(norm, 1.0), "Состояние должно быть нормировано"

    def test_zero_state(self):
        """Тест создания |0⟩"""
        state = QuantumState.zero_state(1)
        assert np.isclose(state.probability(0), 1.0)
        assert np.isclose(state.probability(1), 0.0)

    def test_plus_state(self):
        """Тест создания |+⟩"""
        state = QuantumState.plus_state(1)
        # |+⟩ должен давать 50/50 вероятность для |0⟩ и |1⟩
        assert np.isclose(state.probability(0), 0.5, atol=0.01)
        assert np.isclose(state.probability(1), 0.5, atol=0.01)

    def test_measurement_statistics(self):
        """Тест статистики измерений"""
        state = QuantumState([1, 0], normalize=False)
        outcomes = state.measure(shots=1000)
        # Все результаты должны быть 0 для |0⟩
        assert np.all(outcomes == 0)


class TestDensityMatrix:
    """Тесты для DensityMatrix"""

    def test_pure_state(self):
        """Тест чистого состояния"""
        state = QuantumState([1, 0], normalize=False)
        rho = state.to_density_matrix()

        assert rho.is_pure(), "Состояние должно быть чистым"
        assert np.isclose(rho.purity(), 1.0), "Чистота = 1"

    def test_maximally_mixed(self):
        """Тест максимально смешанного состояния"""
        rho = DensityMatrix.maximally_mixed(1)

        expected_purity = 0.5  # Для 1 кубита
        assert np.isclose(rho.purity(), expected_purity, atol=0.01)
        assert not rho.is_pure()

    def test_fidelity_identical(self):
        """Тест fidelity идентичных состояний"""
        rho1 = DensityMatrix.maximally_mixed(1)
        rho2 = DensityMatrix.maximally_mixed(1)

        F = rho1.fidelity(rho2)
        assert np.isclose(F, 1.0), "Fidelity идентичных состояний = 1"

    def test_trace_distance(self):
        """Тест trace distance"""
        state1 = QuantumState([1, 0], normalize=False)
        state2 = QuantumState([0, 1], normalize=False)

        rho1 = state1.to_density_matrix()
        rho2 = state2.to_density_matrix()

        D = rho1.trace_distance(rho2)
        assert np.isclose(D, 1.0, atol=0.01), "Ортогональные состояния: D=1"


class TestPauliGates:
    """Тесты для паули-гейтов"""

    def test_pauli_x(self):
        """Тест X gate: X|0⟩ = |1⟩"""
        X = PauliGates.pauli_x()
        state = QuantumState.zero_state(1)

        new_state = X.apply(state)

        assert np.isclose(new_state.probability(1), 1.0)
        assert np.isclose(new_state.probability(0), 0.0)

    def test_hadamard(self):
        """Тест H gate: H|0⟩ = |+⟩"""
        H = PauliGates.hadamard()
        state = QuantumState.zero_state(1)

        new_state = H.apply(state)

        # |+⟩ = (|0⟩ + |1⟩)/√2
        assert np.isclose(new_state.probability(0), 0.5, atol=0.01)
        assert np.isclose(new_state.probability(1), 0.5, atol=0.01)

    def test_pauli_z(self):
        """Тест Z gate: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩"""
        Z = PauliGates.pauli_z()
        state = QuantumState([1, 1], normalize=True)

        new_state = Z.apply(state)

        # Проверяем относительную фазу
        assert np.isclose(np.abs(new_state.statevector[0]), 1/np.sqrt(2))
        assert np.isclose(np.abs(new_state.statevector[1]), 1/np.sqrt(2))

    def test_gate_composition(self):
        """Тест композиции гейтов: HXH = Z"""
        H = PauliGates.hadamard()
        X = PauliGates.pauli_x()
        Z = PauliGates.pauli_z()

        HXH = H @ X @ H

        # Проверяем на |0⟩
        state = QuantumState.zero_state(1)
        result1 = HXH.apply(state)
        result2 = Z.apply(state)

        # Матрицы должны быть близки (с точностью до глобальной фазы)
        assert np.allclose(np.abs(result1.statevector), np.abs(result2.statevector))


class TestRotationGates:
    """Тесты для вращений"""

    def test_rx_pi(self):
        """Тест Rx(π) = -iX"""
        Rx = RotationGates.rx(np.pi)
        X = PauliGates.pauli_x()

        # Rx(π) должен быть эквивалентен X (с точностью до глобальной фазы)
        state = QuantumState.zero_state(1)

        result1 = Rx.apply(state)
        result2 = X.apply(state)

        # Вероятности должны совпадать
        for i in range(2):
            assert np.isclose(result1.probability(i), result2.probability(i))

    def test_ry_pi_half(self):
        """Тест Ry(π/2): |0⟩ → (|0⟩ + |1⟩)/√2"""
        Ry = RotationGates.ry(np.pi / 2)
        state = QuantumState.zero_state(1)

        new_state = Ry.apply(state)

        assert np.isclose(new_state.probability(0), 0.5, atol=0.01)
        assert np.isclose(new_state.probability(1), 0.5, atol=0.01)


class TestTwoQubitGates:
    """Тесты для двухкубитных гейтов"""

    def test_cnot(self):
        """Тест CNOT: |10⟩ → |11⟩"""
        CNOT = TwoQubitGates.cnot(control=0, target=1)

        # |10⟩ = |1⟩ ⊗ |0⟩
        state = QuantumState.computational_basis_state(2, 2)  # |10⟩

        new_state = CNOT.apply(state)

        # Должно получиться |11⟩
        assert np.isclose(new_state.probability(3), 1.0)  # |11⟩ = index 3

    def test_cnot_preserves_00(self):
        """Тест CNOT: |00⟩ → |00⟩"""
        CNOT = TwoQubitGates.cnot()
        state = QuantumState.zero_state(2)  # |00⟩

        new_state = CNOT.apply(state)

        assert np.isclose(new_state.probability(0), 1.0)


class TestPauliMeasurement:
    """Тесты для измерений"""

    def test_z_measurement(self):
        """Тест измерения в Z базисе"""
        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        measurement = PauliMeasurement('Z', qubit_index=0, n_qubits=1)

        counts = measurement.measure(state, shots=100)

        # Должны получить только '0'
        assert counts['0'] > 90  # С учётом статистических флуктуаций

    def test_x_measurement(self):
        """Тест измерения в X базисе"""
        # |+⟩ - собственное состояние X с собственным значением +1
        state = QuantumState([1, 1], normalize=True).to_density_matrix()
        measurement = PauliMeasurement('X', qubit_index=0, n_qubits=1)

        counts = measurement.measure(state, shots=100)

        # Должны получить в основном '+'
        assert counts['+'] > 90


def test_full_pipeline():
    """Интеграционный тест: создание состояния, применение гейта, измерение"""
    # |0⟩
    state = QuantumState.zero_state(1)

    # Применяем H
    H = PauliGates.hadamard()
    state = H.apply(state)

    # Измеряем
    outcomes = state.measure(shots=1000)

    # Должно быть примерно 50/50
    freq_0 = np.sum(outcomes == 0) / 1000
    freq_1 = np.sum(outcomes == 1) / 1000

    assert 0.45 < freq_0 < 0.55
    assert 0.45 < freq_1 < 0.55


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
