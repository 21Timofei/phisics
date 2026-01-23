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
    print_section("ПОШАГОВАЯ ДЕМОНСТРАЦИЯ КВАНТОВОЙ ПРОЦЕССНОЙ ТОМОГРАФИИ", 1)
    print_section("ШАГ 0: Создание неизвестного квантового канала", 1)
    p = 0.15
    channel = DepolarizingChannel(p)

    print(f"\n✓ Создан деполяризующий канал: p = {p}")

    print(f"\nОператоры Крауса:")
    for i, K in enumerate(channel.kraus_operators):
        norm = np.linalg.norm(K)
        print(f"  K_{i}: норма = {norm:.6f}")

    choi_true = channel.get_choi_matrix()
    print_matrix(choi_true, "Истинная Choi matrix")

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

        measurement = PauliMeasurement(basis)
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

    # Измеряем остальные 3 состояния без детального вывода
    for rho_out in output_states[1:]:
        state_measurements = {}
        for basis in ['X', 'Y', 'Z']:
            measurement = PauliMeasurement(basis)
            counts = measurement.measure(rho_out, shots=shots)
            state_measurements[basis] = counts
        measurement_data.append(state_measurements)

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
    print(f"\nШаг 4.3: Нормализация:")
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

    print(f"Реконструировано {len(reconstructed_states)} состояний")


    print_section("ШАГ 5: Построение Choi matrix канала", 1)

    print("\nChoi matrix строится из соответствий: входные состояния → выходные состояния")

    from noiselab.tomography.reconstruction import LinearInversion

    lin_inv = LinearInversion()
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

    choi_reconstructed = choi_raw

    print_section("ШАГ 6: Извлечение операторов Крауса", 1)

    print("\nИз Choi matrix извлекаем операторы Крауса")
    print("χ = Σᵢ λᵢ |vᵢ⟩⟨vᵢ|  →  Kᵢ = √λᵢ · reshape(|vᵢ⟩, (2,2))")

    from noiselab.representations.kraus_decomp import kraus_decomposition

    kraus_operators = kraus_decomposition(choi_reconstructed)

    print(f"\nПолучено операторов Крауса: {len(kraus_operators)}")
    print(f"Ранг Крауса: {len(kraus_operators)}")

    print(f"\nОператоры Крауса:")
    print("-" * 60)

    for i, k in enumerate(kraus_operators):
        print(f"\nK_{i}:")
        print_matrix(k, f"  K_{i}")


    print_section("ШАГ 7: Анализ качества реконструкции", 1)

    print("\nСравниваем реконструированный канал с истинным")

    # Process Fidelity
    fidelity = process_fidelity_choi(choi_true, choi_reconstructed)

    print(f"\n1. Process Fidelity:")
    print(f"   F = Tr(χ_true · χ_reconstructed) = {fidelity:.8f}")
    print(f"   Интерпретация: {fidelity*100:.4f}% перекрытие")

if __name__ == "__main__":
    step_by_step_demo()
