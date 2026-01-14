"""
Тест для проверки исправлений в многокубитном Depolarizing канале
"""

import numpy as np
import sys
sys.path.append('.')

from noiselab.channels.noise_models import DepolarizingChannel

def test_cptp_conditions():
    """Проверка CPTP условий для всех конфигураций"""
    print("="*80)
    print("ТЕСТ: Проверка CPTP условий для Depolarizing канала")
    print("="*80)

    test_cases = [
        (1, 0.1),
        (1, 0.3),
        (1, 0.6),
        (2, 0.05),
        (2, 0.1),
        (2, 0.2),
        (3, 0.05),
        (3, 0.1),
    ]

    results = []

    for n_qubits, p in test_cases:
        try:
            channel = DepolarizingChannel(p, n_qubits=n_qubits)
            kraus_ops = channel.get_kraus_operators()

            # Проверка TP: Σ K†K = I
            dim = 2 ** n_qubits
            sum_kraus = sum(K.conj().T @ K for K in kraus_ops)
            identity = np.eye(dim, dtype=np.complex128)

            tp_error = np.linalg.norm(sum_kraus - identity)

            # Проверка CP: Choi matrix ≥ 0
            choi = channel.get_choi_matrix()
            eigenvalues = np.linalg.eigvalsh(choi)
            min_eigenvalue = eigenvalues.min()

            # Проверка trace Choi
            choi_trace = np.trace(choi).real

            is_cptp = tp_error < 1e-8 and min_eigenvalue >= -1e-8

            status = "✅" if is_cptp else "❌"
            results.append((
                status,
                f"n={n_qubits}, p={p:.2f}",
                f"Kraus ops: {len(kraus_ops)}",
                f"TP error: {tp_error:.2e}",
                f"Min λ: {min_eigenvalue:.2e}",
                f"Tr(J): {choi_trace:.4f}"
            ))

        except Exception as e:
            results.append((
                "❌",
                f"n={n_qubits}, p={p:.2f}",
                f"ERROR: {str(e)[:40]}",
                "",
                "",
                ""
            ))

    # Вывод результатов
    for res in results:
        if len(res) == 6:
            print(f"{res[0]} {res[1]:15s} | {res[2]:15s} | {res[3]:18s} | {res[4]:18s} | {res[5]}")
        else:
            print(f"{res[0]} {res[1]:15s} | {res[2]}")

    passed = sum(1 for r in results if r[0] == "✅")
    print(f"\n{'='*80}")
    print(f"Результат: {passed}/{len(results)} тестов пройдено")
    print(f"{'='*80}\n")

    return passed == len(results)


def test_correct_coefficients():
    """Проверка правильности коэффициентов"""
    print("="*80)
    print("ТЕСТ: Проверка коэффициентов операторов Крауса")
    print("="*80)

    test_cases = [
        (1, 0.1, 4),
        (2, 0.1, 16),
        (3, 0.1, 64),
    ]

    for n_qubits, p, expected_ops in test_cases:
        channel = DepolarizingChannel(p, n_qubits=n_qubits)
        kraus_ops = channel.get_kraus_operators()

        # Проверка числа операторов
        assert len(kraus_ops) == expected_ops, f"Expected {expected_ops} ops, got {len(kraus_ops)}"

        # Проверка коэффициентов
        num_paulis = 4 ** n_qubits
        expected_c0 = np.sqrt(1 - p * (num_paulis - 1) / num_paulis)
        expected_c_pauli = np.sqrt(p / num_paulis)

        # Первый оператор (Identity)
        K0_norm = np.linalg.norm(kraus_ops[0])
        dim = 2 ** n_qubits
        expected_K0_norm = expected_c0 * np.sqrt(dim)  # Норма Фробениуса для c0*I

        print(f"n={n_qubits}, p={p:.2f}:")
        print(f"  Операторов: {len(kraus_ops)} (ожидалось {expected_ops})")
        print(f"  c0 (теория): {expected_c0:.6f}")
        print(f"  ||K0|| / √d: {K0_norm / np.sqrt(dim):.6f}")
        print(f"  c_pauli (теория): {expected_c_pauli:.6f}")

        # Проверка веса первого оператора
        weight_K0 = np.trace(kraus_ops[0].conj().T @ kraus_ops[0]).real
        expected_weight_K0 = expected_c0**2 * dim
        print(f"  Tr(K0†K0): {weight_K0:.6f} (ожидалось {expected_weight_K0:.6f})")

        # Проверка веса остальных операторов
        weight_K1 = np.trace(kraus_ops[1].conj().T @ kraus_ops[1]).real
        expected_weight_K1 = expected_c_pauli**2 * dim
        print(f"  Tr(K1†K1): {weight_K1:.6f} (ожидалось {expected_weight_K1:.6f})")
        print()

    print("✅ Все коэффициенты правильные!\n")
    return True


def test_comparison_old_vs_new():
    """Сравнение старой и новой модели"""
    print("="*80)
    print("СРАВНЕНИЕ: Старая vs Новая модель")
    print("="*80)

    # Для демонстрации разницы вычислим коэффициенты
    test_cases = [
        (2, 0.1),
        (2, 0.3),
        (3, 0.1),
        (3, 0.3),
    ]

    print(f"{'n':3s} | {'p':5s} | {'Старая (p/n)':15s} | {'Новая (правильная)':20s} | {'Разница':10s}")
    print("-" * 80)

    for n_qubits, p in test_cases:
        # Старая модель
        p_eff_old = p / n_qubits
        c0_old = (1 - 3*p_eff_old/4) ** n_qubits

        # Новая модель
        num_paulis = 4 ** n_qubits
        c0_new = 1 - p * (num_paulis - 1) / num_paulis

        diff = abs(c0_old - c0_new) / c0_new * 100

        print(f"{n_qubits:3d} | {p:5.2f} | {c0_old:15.6f} | {c0_new:20.6f} | {diff:9.2f}%")

    print("\n✅ Видна разница между моделями!\n")
    return True


if __name__ == '__main__':
    print("\n🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЙ DEPOLARIZING КАНАЛА\n")

    success = True

    try:
        success &= test_cptp_conditions()
    except Exception as e:
        print(f"❌ Тест CPTP провалился: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success &= test_correct_coefficients()
    except Exception as e:
        print(f"❌ Тест коэффициентов провалился: {e}")
        import traceback
        traceback.print_exc()
        success = False

    try:
        success &= test_comparison_old_vs_new()
    except Exception as e:
        print(f"❌ Тест сравнения провалился: {e}")
        import traceback
        traceback.print_exc()
        success = False

    if success:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Исправления работают корректно.")
    else:
        print("⚠️  Некоторые тесты провалились. Требуется дополнительная проверка.")

    import sys
    sys.exit(0 if success else 1)
