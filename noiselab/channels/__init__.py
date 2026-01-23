"""Модуль квантовых каналов и шумовых моделей (1 кубит)"""
from .base import QuantumChannel
from .kraus import KrausChannel
from .noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel,

)

__all__ = [
    'QuantumChannel',
    'KrausChannel',
    'DepolarizingChannel',
    'AmplitudeDampingChannel',
    'PhaseDampingChannel',
]
