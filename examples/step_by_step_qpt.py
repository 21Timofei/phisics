"""
ПОШАГОВАЯ ДЕМОНСТРАЦИЯ КВАНТОВОЙ ПРОЦЕССНОЙ ТОМОГРАФИИ
Показывает каждый шаг с детальным выводом и объяснениями
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from noiselab.channels.noise_models import DepolarizingChannel
from noiselab.core.states import DensityMatrix
from noiselab.core.measurements import PauliMeasurement
from noiselab.metrics.fidelity import process_fidelity_choi


def print_section(title: str, level: int = 1):
    """Красивый заголовок"""
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
    """Вывод матрицы"""
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


def step_by_step_demo():
    """
    Пошаговая демонстрация QPT с детальным выводом
    """

    print_section("ПОШАГОВАЯ ДЕМОНСТРАЦИЯ КВАНТОВОЙ ПРОЦЕССНОЙ ТОМОГРАФИИ", 1)

    # ============================================================================
    # ШАГ 0: Создание канала
    # ============================================================================
    print_section("ШАГ 0: Создание неизвестного квантового канала", 1)

    p = 0.15
    channel = DepolarizingChannel(p)

    print(f"\n✓ Создан деполяризующий канал: p = {p}")
    print(f"\nФизический смысл:")
    print(f"  • С вероятностью (1-p) = {1-p:.3f} → состояние не меняется")
    print(f"  • С вероятностью p = {p:.3f} → применяется случайная паули-ошибка")

    print(f"\nОператоры Крауса:")
    for i, K in enumerate(channel.kraus_operators):
        norm = np.linalg.norm(K)
        print(f"  K_{i}: норма = {norm:.6f}")

    choi_true = channel.get_choi_matrix()
    print_matrix(choi_true, "Истинная Choi matrix")

    # ============================================================================
    # ШАГ 1: Подготовка входных состояний
    # ============================================================================
    print_section("ШАГ 1: Подготовка входных состояний", 1)

    print("\nДля томографии нужен информационно-полный набор состояний")
    print("Используем 4 базисных состояния (минимальный набор для 1 кубита):")

    # Создаем 4 базисных состояния
    input_states = [
        DensityMatrix(np.array([[1, 0], [0, 0]], dtype=np.complex128)),  # |0⟩⟨0|
        DensityMatrix(np.array([[0, 0], [0, 1]], dtype=np.complex128)),  # |1⟩⟨1|
        DensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)),  # |+⟩⟨+|
        DensityMatrix(np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)),  # |+i⟩⟨+i|
    ]

    state_names = ["|0⟩⟨0|", "|1⟩⟨1|", "|+⟩⟨+|", "|+i⟩⟨+i|"]

    for i, (state, name) in enumerate(zip(input_states, state_names)):
        print(f"\n{i+1}. Состояние {name}:")
        print_matrix(state.matrix, f"  ρ_{i+1}")
        purity = np.trace(state.matrix @ state.matrix).real
        print(f"  Чистота: {purity:.6f}")

    # ============================================================================
    # ШАГ 2: Применение канала
    # ============================================================================
    print_section("ШАГ 2: Применение канала к входным состояниям", 1)

    print("\nПрименяем канал ε к каждому входному состоянию: ρ_out = ε(ρ_in)")

    output_states = []
    for i, (rho_in, name) in enumerate(zip(input_states, state_names)):
        print(f"\n{'─' * 80}")
        print(f"Преобразование {i+1}: {name}")

        rho_out = channel.apply(rho_in)
        output_states.append(rho_out)

        print_matrix(rho_in.matrix, "  ВХОД (ρ_in)")
        print_matrix(rho_out.matrix, "  ВЫХОД (ρ_out)")

        # Анализ
        fidelity = np.trace(rho_in.matrix @ rho_out.matrix).real
        purity_in = np.trace(rho_in.matrix @ rho_in.matrix).real
        purity_out = np.trace(rho_out.matrix @ rho_out.matrix).real

        print(f"\n  Анализ:")
        print(f"    Fidelity: {fidelity:.6f}")
        print(f"    Чистота: {purity_in:.6f} → {purity_out:.6f}")
        print(f"    Потеря чистоты: {purity_in - purity_out:.6f}")

    # ============================================================================
    # ШАГ 3: Измерения
    # ============================================================================
    print_section("ШАГ 3: Измерения выходных состояний", 1)

    print("\nПроводим измерения в паули-базисах: X, Y, Z")
    shots = 2000
    print(f"Число измерений на базис: {shots}")

    measurement_data = []

    # Показываем детально только первое состояние
    print(f"\n{'═' * 80}")
    print(f"ДЕТАЛЬНО: Измерения для состояния |0⟩⟨0|")
    print(f"{'═' * 80}")

    rho_out = output_states[0]
    state_measurements = {}

    for basis in ['X', 'Y', 'Z']:
        print(f"\n>>> Измерение в базисе {basis}")

        measurement = PauliMeasurement(basis, qubit_index=0, n_qubits=1)
        counts = measurement.measure(rho_out, shots=shots)
        state_measurements[basis] = counts

        # Вывод результатов
        total = sum(counts.values())
        print(f"  Результаты:")
        for outcome, count in sorted(counts.items()):
            prob = count / total
            print(f"    {outcome}: {count:4d} раз ({prob:.4f})")

        # Expectation value
        n_plus = counts.get('0', 0) + counts.get('+', 0) + counts.get('+i', 0)
        n_minus = counts.get('1', 0) + counts.get('-', 0) + counts.get('-i', 0)
        exp_val = (n_plus - n_minus) / total
        print(f"  ⟨σ_{basis}⟩ = {exp_val:+.6f}")

    measurement_data.append(state_measurements)

    # Остальные состояния без деталей
    print(f"\n{'─' * 80}")
    print("Измеряем остальные 3 состояния...")

    for rho_out in output_states[1:]:
        state_measurements = {}
        for basis in ['X', 'Y', 'Z']:
            measurement = PauliMeasurement(basis, qubit_index=0, n_qubits=1)
            counts = measurement.measure(rho_out, shots=shots)
            state_measurements[basis] = counts
        measurement_data.append(state_measurements)

    print(f"✓ Всего проведено {len(output_states) * 3} серий измерений")
    print(f"✓ Общее число квантовых измерений: {len(output_states) * 3 * shots}")

    # ============================================================================
    # ШАГ 4: Реконструкция состояний
    # ============================================================================
    print_section("ШАГ 4: Реконструкция выходных состояний из измерений", 1)

    print("\nИспользуем формулу:")
    print("  ρ = (I + ⟨X⟩·X + ⟨Y⟩·Y + ⟨Z⟩·Z) / 2")

    reconstructed_states = []

    # Показываем детально первое состояние
    print(f"\n{'═' * 80}")
    print(f"ДЕТАЛЬНО: Реконструкция состояния |0⟩⟨0|")
    print(f"{'═' * 80}")

    measurements = measurement_data[0]

    # Вычисляем expectation values
    exp_values = {}
    for basis in ['X', 'Y', 'Z']:
        counts = measurements[basis]
        total = sum(counts.values())
        n_plus = counts.get('0', 0) + counts.get('+', 0) + counts.get('+i', 0)
        n_minus = counts.get('1', 0) + counts.get('-', 0) + counts.get('-i', 0)
        exp_values[basis] = (n_plus - n_minus) / total

    print(f"\nШаг 4.1: Вычислены expectation values:")
    print(f"  ⟨X⟩ = {exp_values['X']:+.6f}")
    print(f"  ⟨Y⟩ = {exp_values['Y']:+.6f}")
    print(f"  ⟨Z⟩ = {exp_values['Z']:+.6f}")

    # Паули матрицы
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)

    print(f"\nШаг 4.2: Строим матрицу плотности:")
    print(f"  ρ = (I + {exp_values['X']:+.3f}·X + {exp_values['Y']:+.3f}·Y + {exp_values['Z']:+.3f}·Z) / 2")

    rho_raw = (I + exp_values['X']*X + exp_values['Y']*Y + exp_values['Z']*Z) / 2
    print_matrix(rho_raw, "  ρ (сырая)")

    # Проверяем физичность
    eigenvalues = np.linalg.eigvalsh(rho_raw)
    print(f"\nШаг 4.3: Проверка физичности:")
    print(f"  Собственные значения: {[f'{ev:.6f}' for ev in eigenvalues]}")

    if np.any(eigenvalues < -1e-10):
        print(f"  ⚠ Есть отрицательные! Обрезаем их до 0")
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvectors = np.linalg.eigh(rho_raw)[1]
        rho_physical = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    else:
        print(f"  ✓ Все положительные")
        rho_physical = rho_raw

    # Нормализация
    trace = np.trace(rho_physical).real
    print(f"\nШаг 4.4: Нормализация:")
    print(f"  Trace = {trace:.6f} (должно быть 1.0)")

    if abs(trace - 1.0) > 1e-6:
        print(f"  → Нормализуем: ρ = ρ / {trace:.6f}")
        rho_physical = rho_physical / trace
    else:
        print(f"  ✓ Уже нормализовано")

    print_matrix(rho_physical, "  ρ (финальная)")

    # Сравнение с истинным
    rho_true = output_states[0].matrix
    fidelity = np.trace(rho_true @ rho_physical).real
    print(f"\nСравнение с истинным состоянием:")
    print(f"  Fidelity: {fidelity:.6f}")

    reconstructed_states.append(DensityMatrix(rho_physical))

    # Остальные состояния без деталей
    print(f"\n{'─' * 80}")
    print("Реконструируем остальные 3 состояния...")

    for measurements in measurement_data[1:]:
        exp_values = {}
        for basis in ['X', 'Y', 'Z']:
            counts = measurements[basis]
            total = sum(counts.values())
            n_plus = counts.get('0', 0) + counts.get('+', 0) + counts.get('+i', 0)
            n_minus = counts.get('1', 0) + counts.get('-', 0) + counts.get('-i', 0)
            exp_values[basis] = (n_plus - n_minus) / total

        rho = (I + exp_values['X']*X + exp_values['Y']*Y + exp_values['Z']*Z) / 2
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.maximum(eigenvalues, 0)
        trace = np.trace(rho).real
        if trace > 1e-10:
            rho = rho / trace

        reconstructed_states.append(DensityMatrix(rho))

    print(f"✓ Реконструировано {len(reconstructed_states)} состояний")

    # ============================================================================
    # ШАГ 5: Построение Choi matrix
    # ============================================================================
    print_section("ШАГ 5: Построение Choi matrix канала", 1)

    print("\nChoi matrix строится из соответствий: входные состояния → выходные состояния")
    print("Используем метод линейной инверсии (LSQ)")

    from noiselab.tomography.reconstruction import LinearInversion

    lin_inv = LinearInversion(n_qubits=1)
    choi_raw = lin_inv.reconstruct_choi(input_states, reconstructed_states)

    print(f"\nШаг 5.1: Построена сырая Choi matrix")
    print_matrix(choi_raw, "χ (сырая)")

    # Анализ
    eigenvalues_raw = np.linalg.eigvalsh(choi_raw)
    trace_raw = np.trace(choi_raw).real

    print(f"\nШаг 5.2: Анализ сырой матрицы:")
    print(f"  Собственные значения: {[f'{ev:.6f}' for ev in eigenvalues_raw]}")
    print(f"  Минимальное: {np.min(eigenvalues_raw):.6e}")
    print(f"  Отрицательных: {np.sum(eigenvalues_raw < -1e-10)}")
    print(f"  Trace: {trace_raw:.6f} (должно быть 2.0)")

    # Проверка Trace Preservation
    choi_reshaped = choi_raw.reshape(2, 2, 2, 2)
    partial_trace_raw = np.trace(choi_reshaped.transpose(0, 2, 1, 3).reshape(4, 4))
    tp_error_raw = np.abs(partial_trace_raw - 2.0)

    print(f"\nШаг 5.3: Проверка CPTP:")
    print(f"  Complete Positivity (CP):")
    if np.all(eigenvalues_raw > -1e-10):
        print(f"    ✓ Выполнено (все собственные значения ≥ 0)")
    else:
        print(f"    ✗ Нарушено (есть отрицательные собственные значения)")

    print(f"  Trace Preservation (TP):")
    print(f"    Tr_2(χ) = {partial_trace_raw:.6f} (должно быть 2.0)")
    print(f"    TP error = {tp_error_raw:.6e}")
    if tp_error_raw < 1e-6:
        print(f"    ✓ Выполнено")
    else:
        print(f"    ✗ Нарушено")

    # ============================================================================
    # ШАГ 6: Проекция на CPTP
    # ============================================================================
    print_section("ШАГ 6: Проекция на множество CPTP каналов", 1)

    print("\nЧтобы гарантировать физичность, применяем проекцию на CPTP")
    print("CPTP = Completely Positive + Trace Preserving")

    needs_projection = (np.any(eigenvalues_raw < -1e-10) or tp_error_raw > 1e-6)

    if needs_projection:
        print(f"\n⚠ Матрица НЕ физична - проекция НЕОБХОДИМА")
        if np.any(eigenvalues_raw < -1e-10):
            print(f"  Причина 1: Отрицательные собственные значения")
        if tp_error_raw > 1e-6:
            print(f"  Причина 2: Нарушено Trace Preservation (TP error = {tp_error_raw:.2e})")
    else:
        print(f"\n✓ Матрица уже физична, но применяем проекцию для гарантии")

    print(f"\nШаг 6.1: Применяем проекцию...")
    choi_cptp = lin_inv._project_to_cptp(choi_raw)

    print_matrix(choi_cptp, "χ (после проекции)")

    # Анализ после проекции
    eigenvalues_cptp = np.linalg.eigvalsh(choi_cptp)
    trace_cptp = np.trace(choi_cptp).real

    choi_cptp_reshaped = choi_cptp.reshape(2, 2, 2, 2)
    partial_trace_cptp = np.trace(choi_cptp_reshaped.transpose(0, 2, 1, 3).reshape(4, 4))
    tp_error_cptp = np.abs(partial_trace_cptp - 2.0)

    print(f"\nШаг 6.2: Анализ после проекции:")
    print(f"  Собственные значения: {[f'{ev:.6f}' for ev in eigenvalues_cptp]}")
    print(f"  Минимальное: {np.min(eigenvalues_cptp):.6e}")
    print(f"  Отрицательных: {np.sum(eigenvalues_cptp < -1e-10)}")
    print(f"  Trace: {trace_cptp:.6f}")
    print(f"  Tr_2(χ) = {partial_trace_cptp:.6f}")
    print(f"  TP error = {tp_error_cptp:.6e}")

    print(f"\nШаг 6.3: Изменения после проекции:")
    matrix_change = np.linalg.norm(choi_cptp - choi_raw, ord='fro')
    eigenvalue_change = np.linalg.norm(eigenvalues_cptp - eigenvalues_raw)

    print(f"  Frobenius distance: {matrix_change:.6e}")
    print(f"  Изменение собственных значений: {eigenvalue_change:.6e}")
    print(f"  Изменение TP error: {tp_error_raw:.6e} → {tp_error_cptp:.6e}")

    if matrix_change < 1e-10:
        print(f"  → Матрица практически не изменилась (уже была физична)")
    elif matrix_change < 1e-3:
        print(f"  → Небольшие изменения (хорошие измерения)")
    else:
        print(f"  → Значительные изменения (зашумленные измерения)")

    # ============================================================================
    # ШАГ 7: Извлечение операторов Крауса
    # ============================================================================
    print_section("ШАГ 7: Извлечение операторов Крауса", 1)

    print("\nИз Choi matrix извлекаем операторы Крауса через спектральное разложение")
    print("χ = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|  →  Kᵢ = √λᵢ · reshape(|vᵢ⟩, (2,2))")

    threshold = 1e-10
    positive_indices = eigenvalues_cptp > threshold
    eigenvalues_positive = eigenvalues_cptp[positive_indices]

    print(f"\nШаг 7.1: Отбираем значимые собственные значения:")
    print(f"  Порог: {threshold:.2e}")
    print(f"  Значимых: {len(eigenvalues_positive)} из {len(eigenvalues_cptp)}")
    print(f"  Ранг Крауса: {len(eigenvalues_positive)}")

    eigenvectors = np.linalg.eigh(choi_cptp)[1]
    kraus_operators = []

    print(f"\nШаг 7.2: Строим операторы Крауса:")

    for i in range(len(eigenvalues_positive)):
        sqrt_eigenvalue = np.sqrt(eigenvalues_positive[i])
        vec = eigenvectors[:, positive_indices][:, i]
        K = sqrt_eigenvalue * vec.reshape(2, 2)
        kraus_operators.append(K)

        print(f"\n  K_{i} (вес λ_{i} = {eigenvalues_positive[i]:.6f}):")
        print_matrix(K, f"    K_{i}")

        # Проверка
        norm_sq = np.trace(K.conj().T @ K).real
        print(f"    Tr(K†K) = {norm_sq:.6f}")

    # Проверка полноты
    completeness = sum(K.conj().T @ K for K in kraus_operators)
    print(f"\nШаг 7.3: Проверка полноты (Σᵢ K†ᵢKᵢ = I):")
    print_matrix(completeness, "  Σᵢ K†ᵢKᵢ")

    completeness_error = np.linalg.norm(completeness - np.eye(2))
    print(f"  Ошибка: {completeness_error:.6e}")
    if completeness_error < 1e-6:
        print(f"  ✓ Условие полноты выполнено")
    else:
        print(f"  ⚠ Условие полноты нарушено")

    # ============================================================================
    # ШАГ 8: Анализ качества
    # ============================================================================
    print_section("ШАГ 8: Анализ качества реконструкции", 1)

    print("\nСравниваем реконструированный канал с истинным")

    # Process Fidelity
    fidelity = process_fidelity_choi(choi_true, choi_cptp)

    print(f"\n1. Process Fidelity:")
    print(f"   F = Tr(χ_true · χ_reconstructed) = {fidelity:.8f}")
    print(f"   Интерпретация: {fidelity*100:.4f}% перекрытие")

    if fidelity > 0.99:
        print(f"   ✓ Отличная реконструкция!")
    elif fidelity > 0.95:
        print(f"   ✓ Хорошая реконструкция")
    else:
        print(f"   ⚠ Требуется улучшение")

    # Frobenius distance
    frobenius_dist = np.linalg.norm(choi_true - choi_cptp, ord='fro')
    print(f"\n2. Frobenius distance:")
    print(f"   ||χ_true - χ_reconstructed||_F = {frobenius_dist:.8f}")

    # Сравнение параметров
    print(f"\n3. Оценка параметра шума:")
    print(f"   Истинный p: {p:.6f}")

    if len(kraus_operators) > 0:
        K0_coeff = np.abs(kraus_operators[0][0, 0])
        estimated_p = 4 * (1 - K0_coeff**2) / 3
        error = abs(estimated_p - p)
        rel_error = error / p * 100

        print(f"   Оценённый p: {estimated_p:.6f}")
        print(f"   Абсолютная ошибка: {error:.6f}")
        print(f"   Относительная ошибка: {rel_error:.2f}%")

    # ============================================================================
    # ФИНАЛЬНЫЙ ОТЧЕТ
    # ============================================================================
    print_section("ФИНАЛЬНЫЙ ОТЧЕТ", 1)

    print("\n" + "=" * 80)
    print("  СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    print(f"\n1. ИССЛЕДУЕМЫЙ КАНАЛ:")
    print(f"   Тип: Деполяризующий")
    print(f"   Параметр: p = {p}")

    print(f"\n2. ТОМОГРАФИЯ:")
    print(f"   Входных состояний: {len(input_states)}")
    print(f"   Shots на измерение: {shots}")
    print(f"   Всего измерений: {len(input_states) * 3 * shots}")

    print(f"\n3. РЕКОНСТРУКЦИЯ:")
    print(f"   Метод: Linear Inversion (LSQ)")
    print(f"   Ранг Крауса: {len(kraus_operators)}")
    print(f"   CPTP проекция: {'Применена' if needs_projection else 'Не требовалась'}")

    print(f"\n4. КАЧЕСТВО:")
    print(f"   Process Fidelity: {fidelity:.8f}")
    print(f"   Frobenius distance: {frobenius_dist:.8f}")
    print(f"   Оценка параметра: {estimated_p:.6f} (истинный: {p:.6f})")

    print(f"\n5. CPTP ПРОВЕРКА:")
    print(f"   Complete Positivity: {'✓ PASSED' if np.all(eigenvalues_cptp > -1e-10) else '✗ FAILED'}")
    print(f"   Trace Preservation: {'✓ PASSED' if tp_error_cptp < 1e-6 else '✗ FAILED'}")
    print(f"   Полнота Крауса: {'✓ PASSED' if completeness_error < 1e-6 else '✗ FAILED'}")

    print("\n" + "=" * 80)
    print("  ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    step_by_step_demo()
