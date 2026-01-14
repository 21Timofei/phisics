"""Модуль метрик качества и близости каналов"""
from .fidelity import (
    process_fidelity,
    process_fidelity_choi,
    average_gate_fidelity,
    entanglement_fidelity
)
from .distance import (
    diamond_distance,
    trace_distance_channels,
    frobenius_distance
)
from .validation import (
    validate_cptp,
    check_physicality,
    estimate_error_rates
)

__all__ = [
    'process_fidelity',
    'process_fidelity_choi',
    'average_gate_fidelity',
    'entanglement_fidelity',
    'diamond_distance',
    'trace_distance_channels',
    'frobenius_distance',
    'validate_cptp',
    'check_physicality',
    'estimate_error_rates'
]
