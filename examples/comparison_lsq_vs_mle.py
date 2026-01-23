"""
СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ: LSQ vs MLE
Показывает разницу между методами, которые гарантируют и не гарантируют CPTP
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from noiselab.channels.noise_models import DepolarizingChannel

from noiselab.tomography.qpt import QuantumProcessTomography
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


def check_cptp(choi: np.ndarray, name: str = "Choi matrix"):
    """Проверка CPTP свойств"""
    print(f"\nПроверка CPTP для {name}:")

    # Complete Positivity
    eigenvalues = np.linalg.eigvalsh(choi)
    min_eigenvalue = np.min(eigenvalues)
    n_negative = np.sum(eigenvalues < -1e-10)

    print(f"\n1. Complete Positivity (CP):")
    print(f"   Собственные значения: {[f'{ev:.6f}' for ev in eigenvalues]}")
    print(f"   Минимальное: {min_eigenvalue:.6e}")
    print(f"   Отрицательных: {n_negative}")

    if n_negative == 0:
        print(f"   ✓ PASSED - все собственные значения ≥ 0")
        cp_passed = True
    else:
        print(f"   ✗ FAILED - есть отрицательные собственные значения")
        cp_passed = False

    # Trace Preservation - ПРАВИЛЬНАЯ проверка
    # Для Choi matrix: Tr_2(J) = I (единичная матрица)
    # Правильная формула: (Tr_2(J))_{ik} = Σ_j J_{ij,kj}
    dim = int(np.sqrt(choi.shape[0]))

    # Вычисляем Tr_2(J) правильно
    reduced = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(dim):
        for k in range(dim):
            for j in range(dim):
                reduced[i, k] += choi[i*dim + j, k*dim + j]

    identity = np.eye(dim, dtype=np.complex128)
    tp_error = np.linalg.norm(reduced - identity)

    print(f"\n2. Trace Preservation (TP):")
    print(f"   Tr_2(χ) должно быть единичной матрицей I")
    print(f"   ||Tr_2(χ) - I|| = {tp_error:.6e}")

    if tp_error < 1e-6:
        print(f"   ✓ PASSED - след сохраняется")
        tp_passed = True
    else:
        print(f"   ✗ FAILED - след не сохраняется")
        tp_passed = False

    # Общий результат
    print(f"\n3. Общий результат:")
    if cp_passed and tp_passed:
        print(f"   ✓✓ CPTP PASSED - канал физичен")
    else:
        print(f"   ✗✗ CPTP FAILED - канал НЕ физичен")
        if not cp_passed:
            print(f"      → Нарушена Complete Positivity")
        if not tp_passed:
            print(f"      → Нарушена Trace Preservation")

    return cp_passed and tp_passed


def compare_methods():
    """
    Сравнение LSQ и MLE методов
    """

    print_section("СРАВНЕНИЕ МЕТОДОВ РЕКОНСТРУКЦИИ: LSQ vs MLE", 1)

    print("\nЦель: Показать разницу между методами, которые:")
    print("  • LSQ (Linear Inversion) - НЕ гарантирует CPTP")
    print("  • MLE (Maximum Likelihood) - ГАРАНТИРУЕТ CPTP")

    # ============================================================================
    # Подготовка
    # ============================================================================
    print_section("ПОДГОТОВКА ЭКСПЕРИМЕНТА", 1)

    p = 0.15
    channel = DepolarizingChannel(p)
    choi_true = channel.get_choi_matrix()

    print(f"\n✓ Создан деполяризующий канал: p = {p}")
    print_matrix(choi_true, "Истинная Choi matrix")

    # Проверяем истинный канал
    check_cptp(choi_true, "истинного канала")

    # ============================================================================
    # МЕТОД 1: LSQ (Linear Inversion)
    # ============================================================================
    print_section("МЕТОД 1: LSQ (Linear Inversion)", 1)

    print("\nОписание метода:")
    print("  • Решает систему линейных уравнений методом наименьших квадратов")
    print("  • Минимизирует ||Ax - b||²")
    print("  • НЕ учитывает ограничения CPTP")
    print("  • Быстрый, но может дать нефизичный результат")

    print("\nЗапуск томографии с LSQ...")
    qpt_lsq = QuantumProcessTomography(shots=2000)

    result_lsq = qpt_lsq.run_tomography(
        channel,
        reconstruction_method='LSQ',
        add_measurement_noise=False
    )

    choi_lsq = result_lsq.choi_matrix

    print("\n" + "─" * 80)
    print("РЕЗУЛЬТАТ LSQ:")
    print("─" * 80)

    print_matrix(choi_lsq, "Choi matrix (LSQ)")

    # Проверка CPTP
    lsq_is_cptp = check_cptp(choi_lsq, "LSQ реконструкции")

    # Process Fidelity
    fidelity_lsq = process_fidelity_choi(choi_true, choi_lsq)
    print(f"\nProcess Fidelity (LSQ): {fidelity_lsq:.8f}")

    # ============================================================================
    # МЕТОД 2: MLE (Maximum Likelihood Estimation)
    # ============================================================================
    print_section("МЕТОД 2: MLE (Maximum Likelihood Estimation)", 1)

    print("\nОписание метода:")
    print("  • Максимизирует функцию правдоподобия")
    print("  • Решает задачу оптимизации с ограничениями CPTP")
    print("  • ГАРАНТИРУЕТ физичность результата")
    print("  • Медленнее, но всегда дает физичный канал")

    print("\nЗапуск томографии с MLE...")
    qpt_mle = QuantumProcessTomography(shots=2000)

    try:
        result_mle = qpt_mle.run_tomography(
            channel,
            reconstruction_method='MLE',
            add_measurement_noise=False
        )

        choi_mle = result_mle.choi_matrix

        print("\n" + "─" * 80)
        print("РЕЗУЛЬТАТ MLE:")
        print("─" * 80)

        print_matrix(choi_mle, "Choi matrix (MLE)")

        # Проверка CPTP
        mle_is_cptp = check_cptp(choi_mle, "MLE реконструкции")

        # Process Fidelity
        fidelity_mle = process_fidelity_choi(choi_true, choi_mle)
        print(f"\nProcess Fidelity (MLE): {fidelity_mle:.8f}")

        mle_available = True

    except Exception as e:
        print(f"\n⚠ MLE метод недоступен или не работает:")
        print(f"   {str(e)}")
        print(f"\n   Это нормально - MLE требует дополнительных библиотек (CVXPY)")
        mle_available = False
        mle_is_cptp = None
        fidelity_mle = None

    # ============================================================================
    # СРАВНЕНИЕ
    # ============================================================================
    print_section("СРАВНЕНИЕ РЕЗУЛЬТАТОВ", 1)

    print(f"\n{'Метрика':<40} {'LSQ':>20} {'MLE':>20}")
    print("─" * 80)

    # Process Fidelity
    print(f"{'Process Fidelity':<40} {fidelity_lsq:>20.8f} ", end="")
    if mle_available:
        print(f"{fidelity_mle:>20.8f}")
    else:
        print(f"{'N/A':>20}")

    # CPTP проверка
    print(f"{'CPTP статус':<40} ", end="")
    print(f"{'✗ FAILED' if not lsq_is_cptp else '✓ PASSED':>20} ", end="")
    if mle_available:
        print(f"{'✗ FAILED' if not mle_is_cptp else '✓ PASSED':>20}")
    else:
        print(f"{'N/A':>20}")

    # Complete Positivity
    eigenvalues_lsq = np.linalg.eigvalsh(choi_lsq)
    min_ev_lsq = np.min(eigenvalues_lsq)
    n_neg_lsq = np.sum(eigenvalues_lsq < -1e-10)

    print(f"{'Мин. собственное значение':<40} {min_ev_lsq:>20.6e} ", end="")
    if mle_available:
        eigenvalues_mle = np.linalg.eigvalsh(choi_mle)
        min_ev_mle = np.min(eigenvalues_mle)
        print(f"{min_ev_mle:>20.6e}")
    else:
        print(f"{'N/A':>20}")

    print(f"{'Отрицательных собственных значений':<40} {n_neg_lsq:>20d} ", end="")
    if mle_available:
        n_neg_mle = np.sum(eigenvalues_mle < -1e-10)
        print(f"{n_neg_mle:>20d}")
    else:
        print(f"{'N/A':>20}")

    # Trace Preservation
    dim = 2
    choi_lsq_reshaped = choi_lsq.reshape(dim, dim, dim, dim)
    partial_trace_lsq = np.trace(choi_lsq_reshaped.transpose(0, 2, 1, 3).reshape(dim*dim, dim*dim))
    tp_error_lsq = np.abs(partial_trace_lsq - dim)

    print(f"{'TP error':<40} {tp_error_lsq:>20.6e} ", end="")
    if mle_available:
        choi_mle_reshaped = choi_mle.reshape(dim, dim, dim, dim)
        partial_trace_mle = np.trace(choi_mle_reshaped.transpose(0, 2, 1, 3).reshape(dim*dim, dim*dim))
        tp_error_mle = np.abs(partial_trace_mle - dim)
        print(f"{tp_error_mle:>20.6e}")
    else:
        print(f"{'N/A':>20}")

    # ============================================================================
    # ВЫВОДЫ
    # ============================================================================
    print_section("ВЫВОДЫ", 1)

    print("\n1. LSQ (Linear Inversion):")
    print(f"   Process Fidelity: {fidelity_lsq:.6f}")
    if lsq_is_cptp:
        print(f"   ✓ CPTP: PASSED")
        print(f"   → В данном случае LSQ дал физичный результат")
        print(f"   → Это возможно при хороших измерениях")
    else:
        print(f"   ✗ CPTP: FAILED")
        if n_neg_lsq > 0:
            print(f"   → Нарушена Complete Positivity ({n_neg_lsq} отрицательных собственных значений)")
        if tp_error_lsq > 1e-6:
            print(f"   → Нарушена Trace Preservation (TP error = {tp_error_lsq:.2e})")
        print(f"   → LSQ НЕ гарантирует физичность!")

    if mle_available:
        print(f"\n2. MLE (Maximum Likelihood):")
        print(f"   Process Fidelity: {fidelity_mle:.6f}")
        if mle_is_cptp:
            print(f"   ✓ CPTP: PASSED")
            print(f"   → MLE гарантирует физичность результата")
            print(f"   → Все ограничения CPTP выполнены")
        else:
            print(f"   ⚠ CPTP: FAILED (неожиданно!)")
            print(f"   → Возможна проблема в реализации")

        print(f"\n3. Сравнение:")
        fidelity_diff = abs(fidelity_lsq - fidelity_mle)
        print(f"   Разница в fidelity: {fidelity_diff:.6e}")

        if fidelity_diff < 1e-3:
            print(f"   → Оба метода дают похожие результаты")
            print(f"   → При хороших измерениях разница минимальна")
        else:
            print(f"   → LSQ дает лучшую fidelity, но нарушает CPTP")
            print(f"   → MLE дает физичный результат, но fidelity ниже")
            print(f"   → Причина: MLE использует упрощенную реконструкцию без CVXPY")
    else:
        print(f"\n2. MLE (Maximum Likelihood):")
        print(f"   ⚠ Метод недоступен")
        print(f"   → Требуется установка CVXPY:")
        print(f"      pip install cvxpy")
        print(f"   → MLE гарантирует CPTP, но работает медленнее")

    print(f"\n4. Рекомендации:")
    print(f"   • Для обучения: используйте LSQ (быстро, высокая fidelity)")
    print(f"   • Для исследований: используйте MLE (гарантирует CPTP)")
    print(f"   • LSQ: высокая точность, но может нарушать TP")
    print(f"   • MLE: гарантирует физичность, но требует CVXPY для лучшей точности")

    # ============================================================================
    # ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: С ШУМОМ
    # ============================================================================
    if mle_available:
        print_section("ДОПОЛНИТЕЛЬНЫЙ ТЕСТ: С ЗАШУМЛЕННЫМИ ИЗМЕРЕНИЯМИ", 1)

        print("\nТеперь добавим шум измерений и сравним методы:")
        print("  • Shots: 500 (меньше)")
        print("  • Readout error: 0.02 (2% ошибка)")

        print("\nLSQ с шумом...")
        qpt_lsq_noisy = QuantumProcessTomography(shots=500)
        result_lsq_noisy = qpt_lsq_noisy.run_tomography(
            channel,
            reconstruction_method='LSQ',
            add_measurement_noise=True,
            readout_error=0.02
        )

        choi_lsq_noisy = result_lsq_noisy.choi_matrix
        lsq_noisy_is_cptp = check_cptp(choi_lsq_noisy, "LSQ с шумом")
        fidelity_lsq_noisy = process_fidelity_choi(choi_true, choi_lsq_noisy)

        print("\nMLE с шумом...")
        qpt_mle_noisy = QuantumProcessTomography(shots=500)
        result_mle_noisy = qpt_mle_noisy.run_tomography(
            channel,
            reconstruction_method='MLE',
            add_measurement_noise=True,
            readout_error=0.02
        )

        choi_mle_noisy = result_mle_noisy.choi_matrix
        mle_noisy_is_cptp = check_cptp(choi_mle_noisy, "MLE с шумом")
        fidelity_mle_noisy = process_fidelity_choi(choi_true, choi_mle_noisy)

        print("\n" + "─" * 80)
        print("СРАВНЕНИЕ С ШУМОМ:")
        print("─" * 80)

        print(f"\n{'Метрика':<40} {'LSQ':>20} {'MLE':>20}")
        print("─" * 80)
        print(f"{'Process Fidelity':<40} {fidelity_lsq_noisy:>20.8f} {fidelity_mle_noisy:>20.8f}")
        print(f"{'CPTP статус':<40} ", end="")
        print(f"{'✗ FAILED' if not lsq_noisy_is_cptp else '✓ PASSED':>20} ", end="")
        print(f"{'✗ FAILED' if not mle_noisy_is_cptp else '✓ PASSED':>20}")

        print(f"\nВыводы с шумом:")
        if not lsq_noisy_is_cptp and mle_noisy_is_cptp:
            print(f"  ✓ MLE сохраняет CPTP даже с шумом")
            print(f"  ✗ LSQ теряет физичность при зашумленных данных")
            print(f"  → MLE предпочтительнее для реальных экспериментов")
        elif lsq_noisy_is_cptp and mle_noisy_is_cptp:
            print(f"  ✓ Оба метода сохраняют CPTP")
            print(f"  → Уровень шума недостаточен для проявления проблемы LSQ")

    print("\n" + "=" * 80)
    print("  СРАВНЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    compare_methods()
