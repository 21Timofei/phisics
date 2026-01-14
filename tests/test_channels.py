"""
Тесты для модуля channels
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noiselab.channels.noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    BitFlipChannel
)
from noiselab.channels.kraus import KrausChannel
from noiselab.core.states import QuantumState


class TestDepolarizingChannel:
    """Тесты для деполяризующего канала"""

    def test_zero_depolarization(self):
        """Тест p=0: канал = identity"""
        channel = DepolarizingChannel(p=0.0, n_qubits=1)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # Состояние не должно измениться
        assert np.allclose(output.matrix, state.matrix, atol=1e-10)

    def test_cptp_validation(self):
        """Тест CPTP условий"""
        channel = DepolarizingChannel(p=0.1, n_qubits=1)

        assert channel.validate_cptp(), "Канал должен быть CPTP"

    def test_depolarization_effect(self):
        """Тест деполяризации: чистота уменьшается"""
        channel = DepolarizingChannel(p=0.3, n_qubits=1)

        # Чистое состояние
        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        initial_purity = state.purity()

        # После канала
        output = channel.apply(state)
        final_purity = output.purity()

        # Чистота должна уменьшиться
        assert final_purity < initial_purity


class TestAmplitudeDampingChannel:
    """Тесты для amplitude damping"""

    def test_ground_state_stable(self):
        """Тест: |0⟩ стабильно под amplitude damping"""
        channel = AmplitudeDampingChannel(gamma=0.5)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # |0⟩ не должно измениться
        assert np.allclose(output.matrix, state.matrix, atol=1e-10)

    def test_excited_state_decay(self):
        """Тест: |1⟩ затухает к |0⟩"""
        channel = AmplitudeDampingChannel(gamma=1.0)  # Полное затухание

        state = QuantumState([0, 1], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # Должно получиться |0⟩
        assert np.isclose(output.probability(0), 1.0, atol=0.01)
        assert np.isclose(output.probability(1), 0.0, atol=0.01)

    def test_partial_decay(self):
        """Тест частичного затухания"""
        gamma = 0.3
        channel = AmplitudeDampingChannel(gamma=gamma)

        state = QuantumState([0, 1], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # P(|1⟩) должна быть (1-γ)
        expected_prob_1 = 1 - gamma
        assert np.isclose(output.probability(1), expected_prob_1, atol=0.01)


class TestPhaseDampingChannel:
    """Тесты для phase damping"""

    def test_preserves_diagonal(self):
        """Тест: phase damping сохраняет диагональ"""
        channel = PhaseDampingChannel(lambda_=0.3)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # Диагональ не должна измениться
        assert np.allclose(np.diag(output.matrix), np.diag(state.matrix))

    def test_destroys_coherence(self):
        """Тест: разрушает когерентность"""
        channel = PhaseDampingChannel(lambda_=0.5)

        # |+⟩ имеет недиагональные элементы
        state = QuantumState([1, 1], normalize=True).to_density_matrix()

        initial_coherence = np.abs(state.matrix[0, 1])

        output = channel.apply(state)
        final_coherence = np.abs(output.matrix[0, 1])

        # Когерентность должна уменьшиться
        assert final_coherence < initial_coherence


class TestBitFlipChannel:
    """Тесты для bit flip"""

    def test_no_flip(self):
        """Тест p=0: без переворота"""
        channel = BitFlipChannel(p=0.0)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        output = channel.apply(state)

        assert np.allclose(output.matrix, state.matrix)

    def test_certain_flip(self):
        """Тест p=1: гарантированный переворот"""
        channel = BitFlipChannel(p=1.0)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()
        output = channel.apply(state)

        # |0⟩ → |1⟩
        assert np.isclose(output.probability(1), 1.0)


class TestChannelComposition:
    """Тесты композиции каналов"""

    def test_sequential_application(self):
        """Тест последовательного применения каналов"""
        channel1 = BitFlipChannel(p=0.1)
        channel2 = DepolarizingChannel(p=0.1, n_qubits=1)

        # Композиция
        composed = channel1.compose(channel2)

        # Должен быть валидным CPTP каналом
        assert composed.validate_cptp()

    def test_channel_power(self):
        """Тест степени канала: ε∘ε∘...∘ε"""
        channel = DepolarizingChannel(p=0.1, n_qubits=1)

        # Применяем 3 раза
        composed = channel.compose(channel).compose(channel)

        state = QuantumState([1, 0], normalize=False).to_density_matrix()

        # Прямое применение
        result1 = composed.apply(state)

        # Последовательное применение
        result2 = channel.apply(channel.apply(channel.apply(state)))

        assert np.allclose(result1.matrix, result2.matrix, atol=1e-10)


class TestRandomChannels:
    """Тесты для случайных каналов"""

    def test_random_cptp_is_valid(self):
        """Тест: случайный CPTP канал валидный"""
        from noiselab.channels.random import random_cptp_channel

        channel = random_cptp_channel(n_qubits=1, seed=42)

        # Проверяем хотя бы базовые свойства
        assert channel.n_qubits == 1
        assert len(channel.get_kraus_operators()) > 0

    def test_random_unitary_is_cptp(self):
        """Тест: случайный унитарный канал CPTP"""
        from noiselab.channels.random import random_unitary_channel

        channel = random_unitary_channel(n_qubits=1, n_unitaries=3, seed=42)

        assert channel.validate_cptp()


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
