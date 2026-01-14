"""Модуль квантовых каналов и шумовых моделей"""
from .base import QuantumChannel
from .kraus import KrausChannel
from .noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    GeneralizedAmplitudeDamping,
    ThermalRelaxationChannel
)
from .two_qubit_noise import (
    TwoQubitDepolarizing,
    CorrelatedNoise,
    CrosstalkChannel
)
from .random import random_cptp_channel, random_unitary_channel

__all__ = [
    'QuantumChannel',
    'KrausChannel',
    'DepolarizingChannel',
    'AmplitudeDampingChannel',
    'PhaseDampingChannel',
    'BitFlipChannel',
    'PhaseFlipChannel',
    'GeneralizedAmplitudeDamping',
    'ThermalRelaxationChannel',
    'TwoQubitDepolarizing',
    'CorrelatedNoise',
    'CrosstalkChannel',
    'random_cptp_channel',
    'random_unitary_channel'
]
