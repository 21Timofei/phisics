"""
Демонстрация квантовой процессной томографии
Пример использования NoiseLab++ для полного цикла QPT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from noiselab.channels.noise_models import DepolarizingChannel, AmplitudeDampingChannel
from noiselab.tomography.qpt import QuantumProcessTomography
from noiselab.metrics.validation import analyze_tomography_quality, estimate_error_rates
from noiselab.channels.kraus import KrausChannel
from noiselab.core.gates import PauliGates


def demo_single_qubit_depolarizing():
    """
    Демонстрация томографии деполяризующего канала на 1 кубите
    """
    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ 1: Томография деполяризующего канала (1 кубит)")
    print("=" * 70)

    # 1. Создаём неизвестный канал с заданным параметром
    true_parameter = 0.1
    unknown_channel = DepolarizingChannel(true_parameter, n_qubits=1)

    print(f"\n✓ Создан неизвестный канал: Depolarizing(p={true_parameter})")
    print(f"  Ранг Крауса: {unknown_channel.kraus_rank()}")

    # 2. Инициализируем QPT
    qpt = QuantumProcessTomography(n_qubits=1, shots=1000)

    # 3. Проводим томографию
    print("\n📊 Запуск томографии...")
    result = qpt.run_tomography(
        unknown_channel,
        reconstruction_method='LSQ',
        add_measurement_noise=False
    )

    # 4. Анализ качества
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ ТОМОГРАФИИ")
    print("="*70)

    quality = analyze_tomography_quality(result)

    print(f"\n✓ Process Fidelity: {result.process_fidelity:.6f}")
    print(f"✓ CPTP валидация: {'PASSED' if quality['is_cptp'] else 'FAILED'}")
    print(f"✓ TP error: {quality['tp_error']:.2e}")
    print(f"✓ Реконструированный ранг: {quality['kraus_rank']}")
    print(f"✓ Число операторов Крауса: {quality['n_kraus_operators']}")

    # 5. Оценка параметра
    print("\n" + "-"*70)
    print("ОЦЕНКА ПАРАМЕТРОВ ШУМА")
    print("-"*70)

    identity_channel = KrausChannel.from_unitary(np.eye(2), name="Identity")
    estimated = estimate_error_rates(
        result.reconstructed_channel,
        identity_channel,
        error_model='depolarizing'
    )

    print(f"\n✓ Истинный параметр p: {true_parameter:.6f}")
    print(f"✓ Оценённый параметр p: {estimated['parameter']:.6f}")
    print(f"✓ Ошибка оценки: {abs(estimated['parameter'] - true_parameter):.6f}")
    print(f"✓ Fit fidelity: {estimated['fit_fidelity']:.6f}")

    return result


def demo_amplitude_damping():
    """
    Демонстрация томографии amplitude damping канала
    """
    print("\n\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 2: Томография Amplitude Damping канала")
    print("=" * 70)

    # Создаём канал с затуханием
    true_gamma = 0.3
    unknown_channel = AmplitudeDampingChannel(true_gamma)

    print(f"\n✓ Создан канал: AmplitudeDamping(γ={true_gamma})")

    # QPT
    qpt = QuantumProcessTomography(n_qubits=1, shots=2000)

    print("\n📊 Запуск томографии...")
    result = qpt.run_tomography(unknown_channel, reconstruction_method='LSQ')

    print(f"\n✓ Process Fidelity: {result.process_fidelity:.6f}")

    # Оценка параметра
    identity_channel = KrausChannel.from_unitary(np.eye(2), name="Identity")
    estimated = estimate_error_rates(
        result.reconstructed_channel,
        identity_channel,
        error_model='amplitude_damping'
    )

    print(f"\n✓ Истинный γ: {true_gamma:.6f}")
    print(f"✓ Оценённый γ: {estimated['parameter']:.6f}")
    print(f"✓ Ошибка: {abs(estimated['parameter'] - true_gamma):.6f}")

    return result


def demo_statistical_analysis():
    """
    Демонстрация статистического анализа (N прогонов томографии)
    """
    print("\n\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 3: Статистический анализ (10 прогонов)")
    print("=" * 70)

    unknown_channel = DepolarizingChannel(0.15, n_qubits=1)
    qpt = QuantumProcessTomography(n_qubits=1, shots=1000)

    print("\n📊 Запуск 10 независимых томографий...")
    results = qpt.run_multiple_tomographies(
        unknown_channel,
        n_runs=10,
        reconstruction_method='LSQ'
    )

    # Статистический анализ
    from noiselab.metrics.validation import statistical_analysis_multiple_runs

    stats = statistical_analysis_multiple_runs(results)

    print("\n" + "="*70)
    print("СТАТИСТИКА")
    print("="*70)

    print(f"\n✓ Средняя fidelity: {stats['fidelity']['mean']:.6f} ± {stats['fidelity']['std']:.6f}")
    print(f"✓ Мин/Макс: {stats['fidelity']['min']:.6f} / {stats['fidelity']['max']:.6f}")
    print(f"✓ Медиана: {stats['fidelity']['median']:.6f}")
    print(f"\n✓ Средний ранг Крауса: {stats['kraus_rank']['mean']:.2f}")
    print(f"✓ Наиболее частый ранг: {stats['kraus_rank']['mode']}")

    return results


def demo_two_qubit_tomography():
    """
    Демонстрация 2-кубитной томографии
    """
    print("\n\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ 4: Двухкубитная томография")
    print("=" * 70)

    from noiselab.channels.two_qubit_noise import TwoQubitDepolarizing

    # Создаём 2-кубитный канал
    unknown_channel = TwoQubitDepolarizing(p=0.1)

    print(f"\n✓ Создан 2-кубитный канал")
    print(f"  Ранг Крауса: {unknown_channel.kraus_rank()}")

    # QPT для 2 кубитов
    # Внимание: это требует больше времени (4^2 = 16 входных состояний, 3^2 = 9 измерений)
    qpt = QuantumProcessTomography(n_qubits=2, shots=500)

    print(f"\n📊 Запуск 2-кубитной томографии...")
    print(f"  Число входных состояний: {len(qpt.input_states)}")
    print(f"  Число измерительных базисов: {len(qpt.measurement_bases)}")
    print(f"  Всего измерений: {len(qpt.input_states) * len(qpt.measurement_bases) * 500}")

    result = qpt.run_tomography(unknown_channel, reconstruction_method='LSQ')

    print(f"\n✓ Process Fidelity: {result.process_fidelity:.6f}")

    quality = analyze_tomography_quality(result)
    print(f"✓ CPTP: {'PASSED' if quality['is_cptp'] else 'FAILED'}")
    print(f"✓ Реконструированный ранг: {quality['kraus_rank']}")

    return result


def main():
    """
    Главная функция демонстрации
    """
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 15 + "NoiseLab++ DEMO: Квантовая томография" + " " * 15 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # Запускаем демонстрации
    try:
        result1 = demo_single_qubit_depolarizing()
        result2 = demo_amplitude_damping()
        result3 = demo_statistical_analysis()
        result4 = demo_two_qubit_tomography()

        print("\n\n" + "=" * 70)
        print("✓ ВСЕ ДЕМОНСТРАЦИИ ВЫПОЛНЕНЫ УСПЕШНО!")
        print("=" * 70)
        print("\nДемонстрация показала:")
        print("  • Томографию различных типов шумовых каналов")
        print("  • Оценку параметров шума с высокой точностью")
        print("  • Статистический анализ множественных прогонов")
        print("  • Масштабирование на 2-кубитные системы")

    except Exception as e:
        print(f"\n❌ Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
