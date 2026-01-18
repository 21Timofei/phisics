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
    CrosstalkChannel,
    GeneralCorrelatedNoise
)
from .random import random_cptp_channel

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
    'CrosstalkChannel',
    'GeneralCorrelatedNoise',
    'random_cptp_channel'
]
