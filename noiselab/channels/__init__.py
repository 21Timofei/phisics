"""Модуль квантовых каналов и шумовых моделей (1 кубит)"""
from .base import QuantumChannel
from .kraus import KrausChannel
from .noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,
    BitFlipChannel,
    PhaseFlipChannel,
    GeneralizedAmplitudeDamping
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
    'random_cptp_channel'
]
