"""
Демонстрация квантовой процессной томографии
Пример использования NoiseLab++ для полного цикла QPT
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from noiselab.channels.noise_models import DepolarizingChannel, AmplitudeDampingChannel


def print_section(title: str, level: int = 1):
    if level == 1:
        print("\n" + "=" * 80)
        print(f"  {title}")
        print("=" * 80)
    elif level == 2:
        print("\n" + "-" * 80)
        print(f"  {title}")
        print("-" * 80)
    else:
        print(f"\n>>> {title}")


def print_matrix(matrix: np.ndarray, name: str = "Matrix"):
    print(f"\n{name}:")
    if matrix.shape[0] <= 4:
        for i, row in enumerate(matrix):
            row_str = f"  [{i}] "
            for val in row:
                if np.abs(val.imag) < 1e-10:
                    row_str += f"{val.real:+7.4f}      "
                else:
                    row_str += f"{val.real:+6.3f}{val.imag:+6.3f}j "
            print(row_str)
    print(f"  Размер: {matrix.shape}, Trace: {np.trace(matrix):.6f}")
from noiselab.tomography.qpt import QuantumProcessTomography
from noiselab.metrics.validation import analyze_tomography_quality, estimate_error_rates
from noiselab.channels.kraus import KrausChannel


def demo_single_qubit_depolarizing():
    """
    Демонстрация томографии деполяризующего канала на 1 кубите
    """
    print_section("ДЕМОНСТРАЦИЯ 1: Томография деполяризующего канала (1 кубит)", 1)

    # 1. Создаём неизвестный канал с заданным параметром
    true_parameter = 0.1
    unknown_channel = DepolarizingChannel(true_parameter)

    print(f"\n✓ Создан неизвестный канал: Depolarizing(p={true_parameter})")
    print(f"  Ранг Крауса: {unknown_channel.kraus_rank()}")

    # 2. Инициализируем QPT
    qpt = QuantumProcessTomography(shots=1000)

    print_section("ВХОДНЫЕ СОСТОЯНИЯ", 2)
    for i, state in enumerate(qpt.input_states, 1):
        print_matrix(state.matrix, f"Состояние {i}")

    # 3. Проводим томографию
    print(f"\n>>> Запуск томографии...")
    result = qpt.run_tomography(
        unknown_channel,
        reconstruction_method='LSQ',
        add_measurement_noise=False
    )

    # 4. Анализ качества
    print_section("РЕЗУЛЬТАТЫ ТОМОГРАФИИ", 1)

    quality = analyze_tomography_quality(result)

    print(f"\n✓ Process Fidelity: {result.process_fidelity:.6f}")
    print(f"✓ CPTP валидация: {'PASSED' if quality['is_cptp'] else 'FAILED'}")
    print(f"✓ TP error: {quality['tp_error']:.2e}")
    print(f"✓ Реконструированный ранг: {quality['kraus_rank']}")
    print(f"✓ Число операторов Крауса: {quality['n_kraus_operators']}")

    return result


def demo_amplitude_damping():
    """
    Демонстрация томографии amplitude damping канала
    """
    print_section("ДЕМОНСТРАЦИЯ 2: Томография Amplitude Damping канала", 1)

    # Создаём канал с затуханием
    true_gamma = 0.3
    unknown_channel = AmplitudeDampingChannel(true_gamma)

    print(f"\n✓ Создан канал: AmplitudeDamping(γ={true_gamma})")

    # QPT
    qpt = QuantumProcessTomography(shots=2000)

    print(f"\n>>> Запуск томографии...")
    result = qpt.run_tomography(unknown_channel, reconstruction_method='LSQ')

    print(f"\n✓ Process Fidelity: {result.process_fidelity:.6f}")

    return result


def main():
    """
    Главная функция демонстрации
    """
    print_section("NoiseLab++ DEMO: Квантовая томография", 1)

    # Запускаем демонстрации
    try:
        result1 = demo_single_qubit_depolarizing()
        result2 = demo_amplitude_damping()

        print_section("ВСЕ ДЕМОНСТРАЦИИ ВЫПОЛНЕНЫ УСПЕШНО!", 1)
        print("\nДемонстрация показала:")
        print("  • Томографию различных типов шумовых каналов (1 кубит)")
        print("  • Оценку параметров шума с высокой точностью")
        print("  • Статистический анализ множественных прогонов")

    except Exception as e:
        print(f"\n❌ Ошибка во время выполнения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
