"""
Валидация и проверка физичности квантовых каналов
"""

import numpy as np
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def validate_cptp(kraus_operators: List[NDArray[np.complex128]],
                  tol: float = 1e-10) -> Dict[str, bool]:
    """
    Валидация CPTP условий для набора операторов Крауса

    Проверяет:
    1. Trace-preserving: Σᵢ Kᵢ†Kᵢ = I
    2. Completely positive: через Choi matrix ≥ 0

    Args:
        kraus_operators: Список операторов Крауса
        tol: Допустимая погрешность

    Returns:
        Словарь с результатами проверки
    """
    if len(kraus_operators) == 0:
        return {"error": "Empty operator list"}

    dim = kraus_operators[0].shape[0]

    # 1. Проверка trace-preserving
    sum_kraus = sum(K.conj().T @ K for K in kraus_operators)
    identity = np.eye(dim, dtype=np.complex128)

    tp_error = np.linalg.norm(sum_kraus - identity)
    is_trace_preserving = tp_error < tol

    # 2. Проверка completely positive через Choi matrix
    choi_dim = dim ** 2
    choi = np.zeros((choi_dim, choi_dim), dtype=np.complex128)

    for K in kraus_operators:
        vec_K = K.reshape(-1, 1)
        choi += vec_K @ vec_K.conj().T

    eigenvalues = np.linalg.eigvalsh(choi)
    min_eigenvalue = eigenvalues.min()
    is_completely_positive = min_eigenvalue >= -tol

    return {
        "is_trace_preserving": is_trace_preserving,
        "tp_error": tp_error,
        "is_completely_positive": is_completely_positive,
        "min_choi_eigenvalue": min_eigenvalue,
        "is_cptp": is_trace_preserving and is_completely_positive,
        "n_operators": len(kraus_operators)
    }


def check_physicality(density_matrix: NDArray[np.complex128],
                     tol: float = 1e-10) -> Dict[str, bool]:
    """
    Проверка физичности матрицы плотности

    Требования:
    1. Эрмитова: ρ† = ρ
    2. Положительно полуопределённая: ρ ≥ 0
    3. Нормированная: Tr(ρ) = 1

    Args:
        density_matrix: Матрица плотности
        tol: Допустимая погрешность

    Returns:
        Словарь с результатами проверки
    """
    # 1. Эрмитовость
    hermitian_error = np.linalg.norm(density_matrix - density_matrix.conj().T)
    is_hermitian = hermitian_error < tol

    # 2. Положительная полуопределённость
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    min_eigenvalue = eigenvalues.min()
    is_positive = min_eigenvalue >= -tol

    # 3. Нормировка
    trace = np.trace(density_matrix).real
    trace_error = abs(trace - 1.0)
    is_normalized = trace_error < tol

    # 4. Чистота (дополнительная информация)
    purity = np.trace(density_matrix @ density_matrix).real
    is_pure = np.isclose(purity, 1.0, atol=tol)

    return {
        "is_hermitian": is_hermitian,
        "hermitian_error": hermitian_error,
        "is_positive": is_positive,
        "min_eigenvalue": min_eigenvalue,
        "is_normalized": is_normalized,
        "trace": trace,
        "trace_error": trace_error,
        "is_physical": is_hermitian and is_positive and is_normalized,
        "purity": purity,
        "is_pure": is_pure
    }


def estimate_error_rates(reconstructed_channel,
                        ideal_channel,
                        error_model: str = 'depolarizing') -> Dict[str, float]:
    """
    Оценка параметров ошибок из реконструированного канала (1 кубит)

    Использует прямой анализ Choi matrix для более точной оценки параметров
    вместо подгонки через сравнение fidelity.

    Args:
        reconstructed_channel: Реконструированный канал из томографии
        ideal_channel: Идеальный канал (без шума)
        error_model: Тип модели шума ('depolarizing', 'amplitude_damping', etc.)

    Returns:
        Словарь с оценёнными параметрами
    """
    from ..metrics.fidelity import process_fidelity
    from ..channels.noise_models import (
        DepolarizingChannel,
        AmplitudeDampingChannel
    )

    if error_model == 'depolarizing':
        # МЕТОД: Прямое сравнение с теоретическими каналами через diamond distance
        # Используем более точную метрику - diamond distance или trace distance

        try:
            # Для 1-кубитного случая используем прямое сравнение
            # Применяем оба канала к набору тестовых состояний и сравниваем выходы

            # Генерируем тестовые состояния
            test_states = []
            for theta in np.linspace(0, np.pi, 5):
                for phi in np.linspace(0, 2*np.pi, 5):
                    # Состояние на сфере Блоха
                    psi = np.array([np.cos(theta/2),
                                   np.exp(1j*phi)*np.sin(theta/2)],
                                  dtype=np.complex128)
                    from ..core.states import QuantumState, DensityMatrix
                    test_states.append(QuantumState(psi).to_density_matrix())

            # Для каждого значения p вычисляем среднее расстояние
            best_p = 0.0
            min_distance = float('inf')

            for p in np.linspace(0, 0.5, 100):
                try:
                    noise = DepolarizingChannel(p)
                    noisy_channel = ideal_channel.compose(noise)

                    # Вычисляем среднее расстояние между выходами
                    total_distance = 0.0
                    for rho in test_states:
                        rho_recon = reconstructed_channel.apply(rho)
                        rho_test = noisy_channel.apply(rho)

                        # Trace distance: D(ρ₁, ρ₂) = ½ Tr|ρ₁ - ρ₂|
                        diff = rho_recon.matrix - rho_test.matrix
                        eigenvalues = np.linalg.eigvalsh(diff)
                        distance = 0.5 * np.sum(np.abs(eigenvalues))
                        total_distance += distance

                    avg_distance = total_distance / len(test_states)

                    if avg_distance < min_distance:
                        min_distance = avg_distance
                        best_p = p
                except:
                    continue

            estimated_p = best_p

            # Уточняем в окрестности
            if best_p > 0:
                p_min = max(0, best_p - 0.05)
                p_max = min(0.75, best_p + 0.05)

                for p in np.linspace(p_min, p_max, 50):
                    try:
                        noise = DepolarizingChannel(p)
                        noisy_channel = ideal_channel.compose(noise)

                        total_distance = 0.0
                        for rho in test_states:
                            rho_recon = reconstructed_channel.apply(rho)
                            rho_test = noisy_channel.apply(rho)

                            diff = rho_recon.matrix - rho_test.matrix
                            eigenvalues = np.linalg.eigvalsh(diff)
                            distance = 0.5 * np.sum(np.abs(eigenvalues))
                            total_distance += distance

                        avg_distance = total_distance / len(test_states)

                        if avg_distance < min_distance:
                            min_distance = avg_distance
                            estimated_p = p
                    except:
                        continue

        except Exception as e:
            # Fallback: подгонка через fidelity
            estimated_p = 0.0
            best_fidelity = 0.0

            for p in np.linspace(0, 0.5, 100):
                try:
                    noise = DepolarizingChannel(p)
                    noisy_channel = ideal_channel.compose(noise)
                    F = process_fidelity(reconstructed_channel, noisy_channel)

                    if F > best_fidelity:
                        best_fidelity = F
                        estimated_p = p
                except:
                    continue

        # Проверяем точность через подгонку
        try:
            noise = DepolarizingChannel(estimated_p)
            noisy_channel = ideal_channel.compose(noise)
            fit_fidelity = process_fidelity(reconstructed_channel, noisy_channel)
        except:
            fit_fidelity = 0.0

        return {
            "model": "depolarizing",
            "parameter": estimated_p,
            "fit_fidelity": fit_fidelity
        }

    elif error_model == 'amplitude_damping':
        # Сканируем параметр γ
        best_gamma = 0.0
        best_fidelity = 0.0

        for gamma in np.linspace(0, 1.0, 50):
            try:
                noise = AmplitudeDampingChannel(gamma)
                noisy_channel = ideal_channel.compose(noise)

                F = process_fidelity(reconstructed_channel, noisy_channel)

                if F > best_fidelity:
                    best_fidelity = F
                    best_gamma = gamma
            except:
                continue

        return {
            "model": "amplitude_damping",
            "parameter": best_gamma,
            "fit_fidelity": best_fidelity
        }

    else:
        return {
            "error": f"Модель {error_model} не реализована для 1 кубита"
        }


def analyze_tomography_quality(qpt_result) -> Dict:
    """
    Анализ качества томографической реконструкции

    Проверяет:
    - CPTP условия реконструированного канала
    - Process fidelity с истинным каналом
    - Ранг Крауса
    - Собственные значения Choi matrix

    Args:
        qpt_result: Результат QPT (QPTResult объект)

    Returns:
        Словарь с метриками качества
    """
    channel = qpt_result.reconstructed_channel

    # CPTP валидация
    kraus_ops = channel.get_kraus_operators()
    cptp_check = validate_cptp(kraus_ops)

    # Choi matrix анализ
    choi = channel.get_choi_matrix()
    eigenvalues = np.linalg.eigvalsh(choi)

    # Ранг
    rank = np.sum(eigenvalues > 1e-10)

    # Спектр
    spectrum_entropy = -np.sum(
        eigenvalues[eigenvalues > 1e-10] *
        np.log(eigenvalues[eigenvalues > 1e-10] + 1e-15)
    )

    analysis = {
        "is_cptp": cptp_check["is_cptp"],
        "tp_error": cptp_check["tp_error"],
        "min_choi_eigenvalue": cptp_check["min_choi_eigenvalue"],
        "kraus_rank": rank,
        "n_kraus_operators": len(kraus_ops),
        "process_fidelity": qpt_result.process_fidelity,
        "spectrum_entropy": spectrum_entropy,
        "reconstruction_method": qpt_result.reconstruction_method,
        "shots": qpt_result.shots_per_measurement
    }

    return analysis


def statistical_analysis_multiple_runs(qpt_results: List) -> Dict:
    """
    Статистический анализ множественных прогонов томографии

    Вычисляет:
    - Средние значения и дисперсии метрик
    - Распределение fidelity
    - Стабильность реконструкции

    Args:
        qpt_results: Список QPTResult объектов

    Returns:
        Словарь со статистикой
    """
    fidelities = [r.process_fidelity for r in qpt_results if r.process_fidelity is not None]
    ranks = [r.reconstructed_channel.kraus_rank() for r in qpt_results]

    if not fidelities:
        return {"error": "Нет данных для анализа"}

    # Анализ CPTP условий
    cptp_violations = []
    for result in qpt_results:
        kraus_ops = result.reconstructed_channel.get_kraus_operators()
        cptp = validate_cptp(kraus_ops)
        cptp_violations.append(cptp["tp_error"])

    analysis = {
        "n_runs": len(qpt_results),
        "fidelity": {
            "mean": np.mean(fidelities),
            "std": np.std(fidelities),
            "min": np.min(fidelities),
            "max": np.max(fidelities),
            "median": np.median(fidelities)
        },
        "kraus_rank": {
            "mean": np.mean(ranks),
            "std": np.std(ranks),
            "mode": int(np.bincount(ranks).argmax())
        },
        "cptp_error": {
            "mean": np.mean(cptp_violations),
            "max": np.max(cptp_violations)
        }
    }

    return analysis
