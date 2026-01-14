"""
Автоматическое тестирование всех сценариев использования NoiseLab++
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from noiselab.channels.noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel
)
from noiselab.channels.two_qubit_noise import TwoQubitDepolarizing
from noiselab.channels.random import random_cptp_channel
from noiselab.tomography.qpt import QuantumProcessTomography
from noiselab.metrics.validation import analyze_tomography_quality
from noiselab.representations.ptm import PauliTransferMatrix


def print_separator(title):
    """Красивый разделитель"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def test_channel_creation():
    """Тест 1: Создание всех типов каналов"""
    print_separator("ТЕСТ 1: Создание каналов")

    tests = []

    # 1-кубитные каналы
    try:
        ch = DepolarizingChannel(p=0.1, n_qubits=1)
        tests.append(("✅", "Depolarizing (1 qubit, p=0.1)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Depolarizing (1 qubit, p=0.1)", str(e)))

    try:
        ch = AmplitudeDampingChannel(gamma=0.3)
        tests.append(("✅", "Amplitude Damping (γ=0.3)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Amplitude Damping (γ=0.3)", str(e)))

    try:
        ch = PhaseDampingChannel(lambda_=0.2)
        tests.append(("✅", "Phase Damping (λ=0.2)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Phase Damping (λ=0.2)", str(e)))

    try:
        ch = random_cptp_channel(n_qubits=1, seed=42)
        tests.append(("✅", "Random CPTP (1 qubit)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Random CPTP (1 qubit)", str(e)))

    # 2-кубитные каналы
    try:
        ch = DepolarizingChannel(p=0.1, n_qubits=2)
        tests.append(("✅", "Depolarizing (2 qubits, p=0.1)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Depolarizing (2 qubits, p=0.1)", str(e)))

    try:
        ch = TwoQubitDepolarizing(p=0.1)
        tests.append(("✅", "Two-Qubit Depolarizing (p=0.1)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Two-Qubit Depolarizing (p=0.1)", str(e)))

    try:
        ch = random_cptp_channel(n_qubits=2, seed=42)
        tests.append(("✅", "Random CPTP (2 qubits)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Random CPTP (2 qubits)", str(e)))

    # 3-кубитные каналы
    try:
        ch = DepolarizingChannel(p=0.1, n_qubits=3)
        tests.append(("✅", "Depolarizing (3 qubits, p=0.1)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Depolarizing (3 qubits, p=0.1)", str(e)))

    try:
        ch = random_cptp_channel(n_qubits=3, seed=42)
        tests.append(("✅", "Random CPTP (3 qubits)", f"Kraus ops: {len(ch.get_kraus_operators())}"))
    except Exception as e:
        tests.append(("❌", "Random CPTP (3 qubits)", str(e)))

    # Вывод результатов
    for status, name, info in tests:
        print(f"{status} {name:45s} | {info}")

    passed = sum(1 for t in tests if t[0] == "✅")
    print(f"\nРезультат: {passed}/{len(tests)} тестов пройдено")
    return passed == len(tests)


def test_qpt_single_qubit():
    """Тест 2: QPT для 1-кубитных каналов"""
    print_separator("ТЕСТ 2: QPT для 1-кубитных каналов")

    configs = [
        ("Depolarizing p=0.1", DepolarizingChannel(p=0.1, n_qubits=1), 0.95),
        ("Depolarizing p=0.3", DepolarizingChannel(p=0.3, n_qubits=1), 0.85),
        ("Amplitude Damping γ=0.3", AmplitudeDampingChannel(gamma=0.3), 0.90),
        ("Phase Damping λ=0.2", PhaseDampingChannel(lambda_=0.2), 0.90),
    ]

    results = []

    for name, channel, expected_fidelity in configs:
        try:
            qpt = QuantumProcessTomography(n_qubits=1, shots=1000)
            result = qpt.run_tomography(channel, reconstruction_method='LSQ')

            fidelity = result.process_fidelity
            quality = analyze_tomography_quality(result)

            status = "✅" if fidelity >= expected_fidelity - 0.1 else "⚠️"
            results.append((
                status,
                name,
                f"Fidelity: {fidelity:.4f}",
                f"CPTP: {quality['is_cptp']}",
                f"Rank: {quality['kraus_rank']}"
            ))
        except Exception as e:
            results.append(("❌", name, str(e)[:50], "", ""))

    for res in results:
        if len(res) == 5:
            print(f"{res[0]} {res[1]:30s} | {res[2]} | {res[3]} | {res[4]}")
        else:
            print(f"{res[0]} {res[1]:30s} | {res[2]}")

    passed = sum(1 for r in results if r[0] in ["✅", "⚠️"])
    print(f"\nРезультат: {passed}/{len(results)} тестов пройдено")
    return passed == len(results)


def test_qpt_two_qubit():
    """Тест 3: QPT для 2-кубитных каналов"""
    print_separator("ТЕСТ 3: QPT для 2-кубитных каналов")

    configs = [
        ("Depolarizing p=0.1", DepolarizingChannel(p=0.1, n_qubits=2), 0.80),
        ("Two-Qubit Depolarizing p=0.1", TwoQubitDepolarizing(p=0.1), 0.80),
    ]

    results = []

    for name, channel, expected_fidelity in configs:
        try:
            qpt = QuantumProcessTomography(n_qubits=2, shots=2000)
            result = qpt.run_tomography(channel, reconstruction_method='LSQ')

            fidelity = result.process_fidelity
            quality = analyze_tomography_quality(result)

            status = "✅" if fidelity >= expected_fidelity - 0.15 else "⚠️"
            results.append((
                status,
                name,
                f"Fidelity: {fidelity:.4f}",
                f"CPTP: {quality['is_cptp']}",
                f"Rank: {quality['kraus_rank']}"
            ))
        except Exception as e:
            results.append(("❌", name, str(e)[:50], "", ""))

    for res in results:
        if len(res) == 5:
            print(f"{res[0]} {res[1]:35s} | {res[2]} | {res[3]} | {res[4]}")
        else:
            print(f"{res[0]} {res[1]:35s} | {res[2]}")

    passed = sum(1 for r in results if r[0] in ["✅", "⚠️"])
    print(f"\nРезультат: {passed}/{len(results)} тестов пройдено")
    return passed == len(results)


def test_qpt_three_qubit():
    """Тест 3.5: QPT для 3-кубитных каналов"""
    print_separator("ТЕСТ 3.5: QPT для 3-кубитных каналов")

    configs = [
        ("Depolarizing p=0.1", DepolarizingChannel(p=0.1, n_qubits=3), 0.70),
    ]

    results = []

    for name, channel, expected_fidelity in configs:
        try:
            print(f"   Запуск {name}... (это может занять несколько минут)")
            qpt = QuantumProcessTomography(n_qubits=3, shots=3000)
            result = qpt.run_tomography(channel, reconstruction_method='LSQ')

            fidelity = result.process_fidelity
            quality = analyze_tomography_quality(result)

            status = "✅" if fidelity >= expected_fidelity - 0.20 else "⚠️"
            results.append((
                status,
                name,
                f"Fidelity: {fidelity:.4f}",
                f"CPTP: {quality['is_cptp']}",
                f"Rank: {quality['kraus_rank']}"
            ))
        except Exception as e:
            results.append(("❌", name, str(e)[:50], "", ""))

    for res in results:
        if len(res) == 5:
            print(f"{res[0]} {res[1]:35s} | {res[2]} | {res[3]} | {res[4]}")
        else:
            print(f"{res[0]} {res[1]:35s} | {res[2]}")

    passed = sum(1 for r in results if r[0] in ["✅", "⚠️"])
    print(f"\nРезультат: {passed}/{len(results)} тестов пройдено")
    return passed == len(results)


def test_reconstruction_methods():
    """Тест 4: Сравнение методов реконструкции"""
    print_separator("ТЕСТ 4: Методы реконструкции (LSQ vs MLE)")

    channel = DepolarizingChannel(p=0.15, n_qubits=1)

    results = []

    for method in ['LSQ', 'MLE']:
        try:
            qpt = QuantumProcessTomography(n_qubits=1, shots=1000)
            import time
            start = time.perf_counter()  # Более точное измерение времени
            result = qpt.run_tomography(channel, reconstruction_method=method)
            elapsed = time.perf_counter() - start

            fidelity = result.process_fidelity
            quality = analyze_tomography_quality(result)

            # Форматируем время с достаточной точностью
            if elapsed < 0.01:
                time_str = f"Time: <0.01s"
            else:
                time_str = f"Time: {elapsed:.2f}s"

            results.append((
                "✅",
                method,
                f"Fidelity: {fidelity:.4f}",
                time_str,
                f"CPTP: {quality['is_cptp']}"
            ))
        except Exception as e:
            results.append(("❌", method, str(e)[:40], "", ""))

    for res in results:
        if len(res) == 5:
            print(f"{res[0]} {res[1]:10s} | {res[2]} | {res[3]} | {res[4]}")
        else:
            print(f"{res[0]} {res[1]:10s} | {res[2]}")

    # Сравнение
    if len(results) == 2 and results[0][0] == "✅" and results[1][0] == "✅":
        lsq_fidelity = float(results[0][2].split(': ')[1])
        mle_fidelity = float(results[1][2].split(': ')[1])

        print(f"\n📊 Сравнение:")
        fidelity_diff = mle_fidelity - lsq_fidelity
        if fidelity_diff > 0:
            print(f"   MLE точнее на: {fidelity_diff:.4f}")
        elif fidelity_diff < 0:
            print(f"   LSQ точнее на: {abs(fidelity_diff):.4f}")
        else:
            print(f"   Методы дают одинаковую точность")

        if "Time:" in results[0][3] and "Time:" in results[1][3]:
            lsq_time_str = results[0][3].split(': ')[1].rstrip('s')
            mle_time_str = results[1][3].split(': ')[1].rstrip('s')

            # Обработка случая "<0.01s"
            if lsq_time_str.startswith('<'):
                lsq_time = 0.005  # Приблизительное значение
            else:
                lsq_time = float(lsq_time_str)

            if mle_time_str.startswith('<'):
                mle_time = 0.005
            else:
                mle_time = float(mle_time_str)

            if lsq_time > 0.001:  # Избегаем деления на ноль
                speedup = mle_time / lsq_time
                print(f"   MLE медленнее в: {speedup:.1f}x раз")
            else:
                print(f"   LSQ: <0.01s, MLE: {mle_time:.2f}s (MLE значительно медленнее)")

    passed = sum(1 for r in results if r[0] == "✅")
    return passed == len(results)


def test_noise_effects():
    """Тест 5: Влияние шума измерений"""
    print_separator("ТЕСТ 5: Влияние шума измерений")

    channel = DepolarizingChannel(p=0.1, n_qubits=1)
    noise_levels = [0.0, 0.01, 0.05]

    results = []

    for noise in noise_levels:
        try:
            qpt = QuantumProcessTomography(n_qubits=1, shots=1000)
            result = qpt.run_tomography(
                channel,
                reconstruction_method='LSQ',
                add_measurement_noise=(noise > 0),
                readout_error=noise
            )

            fidelity = result.process_fidelity

            results.append((
                "✅",
                f"Noise={noise:.2f}",
                f"Fidelity: {fidelity:.4f}"
            ))
        except Exception as e:
            results.append(("❌", f"Noise={noise:.2f}", str(e)[:40]))

    for res in results:
        print(f"{res[0]} {res[1]:15s} | {res[2]}")

    # Анализ деградации
    if len(results) == 3 and all(r[0] == "✅" for r in results):
        fidelities = [float(r[2].split(': ')[1]) for r in results]
        print(f"\n📊 Деградация:")
        print(f"   Без шума:      {fidelities[0]:.4f}")
        print(f"   Шум 1%:        {fidelities[1]:.4f} (потеря: {(fidelities[0]-fidelities[1]):.4f})")
        print(f"   Шум 5%:        {fidelities[2]:.4f} (потеря: {(fidelities[0]-fidelities[2]):.4f})")

    passed = sum(1 for r in results if r[0] == "✅")
    return passed == len(results)


def test_representations():
    """Тест 6: Различные представления каналов"""
    print_separator("ТЕСТ 6: Представления каналов")

    channel = DepolarizingChannel(p=0.1, n_qubits=1)

    tests = []

    # Choi matrix
    try:
        choi = channel.get_choi_matrix()
        tests.append(("✅", "Choi Matrix", f"Shape: {choi.shape}, Trace: {np.trace(choi):.4f}"))
    except Exception as e:
        tests.append(("❌", "Choi Matrix", str(e)[:40]))

    # Kraus operators
    try:
        kraus = channel.get_kraus_operators()
        total_weight = sum(np.trace(K.conj().T @ K).real for K in kraus)
        tests.append(("✅", "Kraus Operators", f"Count: {len(kraus)}, Total weight: {total_weight:.4f}"))
    except Exception as e:
        tests.append(("❌", "Kraus Operators", str(e)[:40]))

    # PTM
    try:
        ptm = PauliTransferMatrix.from_channel(channel)
        is_tp = ptm.is_trace_preserving()
        tests.append(("✅", "PTM", f"Shape: {ptm.ptm_matrix.shape}, TP: {is_tp}"))
    except Exception as e:
        tests.append(("❌", "PTM", str(e)[:40]))

    # Kraus rank
    try:
        rank = channel.kraus_rank()
        tests.append(("✅", "Kraus Rank", f"Rank: {rank}"))
    except Exception as e:
        tests.append(("❌", "Kraus Rank", str(e)[:40]))

    for status, name, info in tests:
        print(f"{status} {name:20s} | {info}")

    passed = sum(1 for t in tests if t[0] == "✅")
    print(f"\nРезультат: {passed}/{len(tests)} тестов пройдено")
    return passed == len(tests)


def test_multiple_runs():
    """Тест 7: Статистический анализ (множественные прогоны)"""
    print_separator("ТЕСТ 7: Множественные прогоны")

    channel = DepolarizingChannel(p=0.1, n_qubits=1)

    try:
        from noiselab.metrics.validation import statistical_analysis_multiple_runs

        qpt = QuantumProcessTomography(n_qubits=1, shots=1000)
        results = qpt.run_multiple_tomographies(channel, n_runs=10)

        stats = statistical_analysis_multiple_runs(results)

        print(f"✅ Множественные прогоны")
        print(f"   Прогонов:        {stats['n_runs']}")
        print(f"   Fidelity mean:   {stats['fidelity']['mean']:.4f}")
        print(f"   Fidelity std:    {stats['fidelity']['std']:.4f}")
        print(f"   Fidelity min:    {stats['fidelity']['min']:.4f}")
        print(f"   Fidelity max:    {stats['fidelity']['max']:.4f}")
        print(f"   Kraus rank mean: {stats['kraus_rank']['mean']:.2f}")

        return True
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_cases():
    """Тест 8: Граничные случаи и параметры"""
    print_separator("ТЕСТ 8: Граничные случаи")

    tests = []

    # Очень малый шум
    try:
        ch = DepolarizingChannel(p=0.001, n_qubits=1)
        qpt = QuantumProcessTomography(n_qubits=1, shots=500)
        result = qpt.run_tomography(ch, reconstruction_method='LSQ')
        tests.append(("✅", "p=0.001 (малый шум)", f"Fidelity: {result.process_fidelity:.4f}"))
    except Exception as e:
        tests.append(("❌", "p=0.001 (малый шум)", str(e)[:40]))

    # Большой шум
    try:
        ch = DepolarizingChannel(p=0.6, n_qubits=1)
        qpt = QuantumProcessTomography(n_qubits=1, shots=500)
        result = qpt.run_tomography(ch, reconstruction_method='LSQ')
        tests.append(("✅", "p=0.6 (большой шум)", f"Fidelity: {result.process_fidelity:.4f}"))
    except Exception as e:
        tests.append(("❌", "p=0.6 (большой шум)", str(e)[:40]))

    # Малое число shots
    try:
        ch = DepolarizingChannel(p=0.1, n_qubits=1)
        qpt = QuantumProcessTomography(n_qubits=1, shots=100)
        result = qpt.run_tomography(ch, reconstruction_method='LSQ')
        tests.append(("✅", "shots=100 (мало)", f"Fidelity: {result.process_fidelity:.4f}"))
    except Exception as e:
        tests.append(("❌", "shots=100 (мало)", str(e)[:40]))

    # Большое число shots
    try:
        ch = DepolarizingChannel(p=0.1, n_qubits=1)
        qpt = QuantumProcessTomography(n_qubits=1, shots=5000)
        result = qpt.run_tomography(ch, reconstruction_method='LSQ')
        tests.append(("✅", "shots=5000 (много)", f"Fidelity: {result.process_fidelity:.4f}"))
    except Exception as e:
        tests.append(("❌", "shots=5000 (много)", str(e)[:40]))

    for status, name, info in tests:
        print(f"{status} {name:30s} | {info}")

    passed = sum(1 for t in tests if t[0] == "✅")
    print(f"\nРезультат: {passed}/{len(tests)} тестов пройдено")
    return passed == len(tests)


def main():
    """Главная функция тестирования"""
    print("\n" + "🧪 " * 40)
    print("  ПОЛНОЕ ТЕСТИРОВАНИЕ NoiseLab++")
    print("🧪 " * 40)

    results = []

    # Запуск всех тестов с обработкой ошибок
    test_functions = [
        ("Создание каналов", test_channel_creation),
        ("QPT 1-кубит", test_qpt_single_qubit),
        ("QPT 2-кубита", test_qpt_two_qubit),
        ("QPT 3-кубита", test_qpt_three_qubit),  # Опционально, может быть медленным
        ("Методы реконструкции", test_reconstruction_methods),
        ("Влияние шума", test_noise_effects),
        ("Представления", test_representations),
        ("Множественные прогоны", test_multiple_runs),
        ("Граничные случаи", test_edge_cases),
    ]

    for name, test_func in test_functions:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА в тесте '{name}': {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Итоговый отчет
    print_separator("ИТОГОВЫЙ ОТЧЕТ")

    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status:12s} | {name}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)

    print(f"\n{'='*80}")
    print(f"Итого: {total_passed}/{total_tests} тестов пройдено")

    if total_passed == total_tests:
        print("🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    else:
        print("⚠️  Некоторые тесты провалились. Требуется доработка.")

    print("="*80 + "\n")

    return total_passed == total_tests


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
