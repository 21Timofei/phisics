"""
Квантовая процессная томография (Quantum Process Tomography)

Главный модуль для проведения полного цикла томографии:
1. Подготовка входных состояний
2. Применение неизвестного канала
3. Измерения в различных базисах
4. Реконструкция канала
5. Анализ результатов
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from numpy.typing import NDArray
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.states import DensityMatrix
from core.measurements import PauliMeasurement
from channels.base import QuantumChannel
from channels.kraus import KrausChannel
from .reconstruction import LinearInversion, MaximumLikelihood


@dataclass
class QPTResult:
    """
    Результат квантовой процессной томографии
    """
    reconstructed_channel: KrausChannel
    input_states: List[DensityMatrix]
    measurement_data: List[Dict[str, Dict[str, int]]]
    choi_matrix: Optional[NDArray[np.complex128]] = None
    process_fidelity: Optional[float] = None
    reconstruction_method: str = "LSQ"
    shots_per_measurement: int = 1000

    def __repr__(self) -> str:
        return (f"QPTResult(n_qubits={self.reconstructed_channel.n_qubits}, "
                f"method={self.reconstruction_method}, "
                f"fidelity={self.process_fidelity:.4f if self.process_fidelity else 'N/A'})")


class QuantumProcessTomography:
    """
    Класс для проведения квантовой процессной томографии

    Полный цикл QPT:
    1. Подготовка набора входных состояний (tomographically complete set)
    2. Применение неизвестного канала к каждому состоянию
    3. Измерения выходных состояний во всех паули-базисах
    4. Реконструкция канала (LSQ или MLE)
    5. Валидация и анализ
    """

    def __init__(self, shots: int = 1000):
        """
        Args:
            shots: Число измерений для каждой настройки
        """

        self.n_qubits = 1
        self.dim = 2
        self.shots = shots

        # Подготовка входных состояний
        # ПРАВИЛЬНЫЙ ПОДХОД: используем базисные операторы {|i⟩⟨j|}
        # вместо произвольных состояний для точной реконструкции Choi matrix
        self.input_states = self._generate_basis_operators()

        # Паули-базисы для измерений
        self.measurement_bases = self._prepare_measurement_bases()

        print(f"QPT инициализирована для 1 кубита")
        print(f"Входных состояний: {len(self.input_states)}")
        print(f"Измерительных базисов: {len(self.measurement_bases)}")
        print(f"Shots на измерение: {shots}")

    def _generate_basis_operators(self) -> List[DensityMatrix]:
        """
        Генерация информационно-полного набора состояний для QPT

        Используем обобщённые состояния Паули, которые образуют полный базис
        в пространстве операторов.

        Для 1 кубита используем 6 состояний:
        - |0⟩, |1⟩ (вычислительный базис)
        - |+⟩, |-⟩ (базис X)
        - |+i⟩, |-i⟩ (базис Y)

        Returns:
            Список состояний как DensityMatrix
        """
        basis_states = []

        # Вычислительный базис: |0⟩, |1⟩
        basis_states.append(DensityMatrix(np.array([[1, 0], [0, 0]], dtype=np.complex128)))
        basis_states.append(DensityMatrix(np.array([[0, 0], [0, 1]], dtype=np.complex128)))

        # X-базис: |+⟩, |-⟩
        basis_states.append(DensityMatrix(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)))
        basis_states.append(DensityMatrix(np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)))

        # Y-базис: |+i⟩, |-i⟩
        basis_states.append(DensityMatrix(np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)))
        basis_states.append(DensityMatrix(np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)))

        # Примечание: 6 состояний избыточны, но обеспечивают лучшую статистику

        return basis_states

    def _prepare_measurement_bases(self) -> List[str]:
        """
        Подготовить список измерительных базисов

        Для 1 кубита: 3 паули-измерения (X, Y, Z)

        Returns:
            Список строк ['X', 'Y', 'Z']
        """
        return ['X', 'Y', 'Z']

    def run_tomography(self,
                      unknown_channel: QuantumChannel,
                      reconstruction_method: str = 'LSQ',
                      add_measurement_noise: bool = False,
                      readout_error: float = 0.0) -> QPTResult:
        """
        Провести полную томографию неизвестного канала

        Args:
            unknown_channel: Неизвестный канал для диагностики
            reconstruction_method: 'LSQ' или 'MLE'
            add_measurement_noise: Добавлять ли шум измерений
            readout_error: Вероятность ошибки считывания

        Returns:
            QPTResult с результатами томографии
        """
        print(f"\n=== Начало томографии ===")
        print(f"Исследуемый канал: {unknown_channel.name}")
        print(f"Метод реконструкции: {reconstruction_method}")

        # Шаг 1: Применяем канал ко всем входным состояниям
        output_states = []
        for i, rho_in in enumerate(self.input_states):
            rho_out = unknown_channel.apply(rho_in)
            output_states.append(rho_out)

        print(f"[OK] Применён канал к {len(output_states)} входным состояниям")

        # Шаг 2: Измеряем выходные состояния во всех базисах
        measurement_data = []

        for state_idx, rho_out in enumerate(output_states):
            state_measurements = {}

            for basis_string in self.measurement_bases:
                # Проводим измерения в данном базисе
                counts = self._measure_in_basis(
                    rho_out,
                    basis_string,
                    add_noise=add_measurement_noise,
                    readout_error=readout_error
                )
                state_measurements[basis_string] = counts

            measurement_data.append(state_measurements)

        print(f"[OK] Проведено {len(measurement_data) * len(self.measurement_bases)} измерений")

        # Шаг 3: Реконструкция канала
        if reconstruction_method == 'LSQ':
            # Реконструируем состояния из измерений (state tomography)
            reconstructed_output_states = self._reconstruct_states_from_measurements(measurement_data)
            reconstructed = self._reconstruct_lsq(reconstructed_output_states)
        elif reconstruction_method == 'MLE':
            reconstructed = self._reconstruct_mle(measurement_data)
        else:
            raise ValueError(f"Неизвестный метод: {reconstruction_method}")

        print(f"[OK] Канал реконструирован (ранг Крауса: {reconstructed.kraus_rank()})")

        # Шаг 4: Вычисляем метрики качества
        choi_true = unknown_channel.get_choi_matrix()
        choi_reconstructed = reconstructed.get_choi_matrix()

        from ..metrics.fidelity import process_fidelity_choi
        fidelity = process_fidelity_choi(choi_true, choi_reconstructed)

        print(f"[OK] Process fidelity: {fidelity:.6f}")

        # Формируем результат
        result = QPTResult(
            reconstructed_channel=reconstructed,
            input_states=self.input_states,
            measurement_data=measurement_data,
            choi_matrix=choi_reconstructed,
            process_fidelity=fidelity,
            reconstruction_method=reconstruction_method,
            shots_per_measurement=self.shots
        )

        print(f"=== Томография завершена ===\n")

        return result

    def _measure_in_basis(self,
                         rho: DensityMatrix,
                         basis_string: str,
                         add_noise: bool = False,
                         readout_error: float = 0.0) -> Dict[str, int]:
        """
        Измерить состояние в указанном паули-базисе

        Args:
            rho: Квантовое состояние
            basis_string: Строка базиса ('X', 'Y' или 'Z')
            add_noise: Добавлять ли шум
            readout_error: Вероятность ошибки считывания

        Returns:
            Словарь с результатами измерений {outcome: count}
        """
        # Однокубитный случай
        measurement = PauliMeasurement(basis_string)
        counts = measurement.measure(rho, shots=self.shots)

        # Добавляем шум измерений если нужно
        if add_noise and readout_error > 0:
            counts = self._add_readout_noise(counts, readout_error)

        return counts


    def _add_readout_noise(self,
                          counts: Dict[str, int],
                          error_rate: float) -> Dict[str, int]:
        """
        Добавить шум считывания к результатам измерений

        Упрощённая модель: каждый бит переворачивается с вероятностью error_rate
        """
        total = sum(counts.values())
        noisy_outcomes = []

        for outcome, count in counts.items():
            for _ in range(count):
                # Применяем шум к каждому биту
                noisy_outcome = ''
                for bit in outcome:
                    if np.random.rand() < error_rate:
                        # Переворачиваем бит
                        noisy_outcome += '1' if bit == '0' else '0'
                    else:
                        noisy_outcome += bit
                noisy_outcomes.append(noisy_outcome)

        # Подсчитываем зашумлённые результаты
        noisy_counts = {}
        for outcome in noisy_outcomes:
            noisy_counts[outcome] = noisy_counts.get(outcome, 0) + 1

        return noisy_counts

    def _reconstruct_states_from_measurements(self, measurement_data: List[Dict]) -> List[DensityMatrix]:
        """
        Реконструировать выходные состояния из данных измерений (state tomography)

        Использует линейную инверсию в базисе Паули для реконструкции
        матриц плотности из expectation values паули-операторов.

        Args:
            measurement_data: Список словарей с результатами измерений для каждого состояния

        Returns:
            Список реконструированных матриц плотности
        """
        reconstructed_states = []

        for state_measurements in measurement_data:
            # Реконструкция через линейную инверсию в базисе Паули
            # ρ = (I + Σᵢ ⟨σᵢ⟩ σᵢ) / 2 для 1 кубита
            # где ⟨σᵢ⟩ - expectation values паули-операторов

            # Извлекаем expectation values из измерений
            exp_x = self._expectation_from_counts(state_measurements.get('X', {}))
            exp_y = self._expectation_from_counts(state_measurements.get('Y', {}))
            exp_z = self._expectation_from_counts(state_measurements.get('Z', {}))

            # Реконструируем матрицу плотности
            # Паули матрицы
            X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

            rho_matrix = (np.eye(2) + exp_x * X + exp_y * Y + exp_z * Z) / 2

            # Проверяем физичность и корректируем если нужно
            # 1. Эрмитовость (уже выполнена по конструкции)
            # 2. Положительность
            eigenvalues, eigenvectors = np.linalg.eigh(rho_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Обрезаем отрицательные
            rho_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T

            # 3. Нормализация Tr(ρ) = 1
            trace = np.trace(rho_matrix).real
            if trace > 1e-10:
                rho_matrix = rho_matrix / trace

            reconstructed_states.append(DensityMatrix(rho_matrix))

        return reconstructed_states

    def _expectation_from_counts(self, counts: Dict[str, int]) -> float:
        """
        Вычислить expectation value из результатов измерений

        Для паули-измерения: ⟨σ⟩ = (N₊ - N₋) / (N₊ + N₋)
        где N₊, N₋ - число измерений с собственными значениями +1 и -1

        Args:
            counts: Словарь с результатами измерений {outcome: count}

        Returns:
            Expectation value ∈ [-1, 1]
        """
        if not counts:
            return 0.0

        # Обрабатываем разные форматы результатов:
        # Z-измерение: '0' (|0⟩, eigenvalue +1), '1' (|1⟩, eigenvalue -1)
        # X-измерение: '+' (|+⟩, eigenvalue +1), '-' (|-⟩, eigenvalue -1)
        # Y-измерение: '+i' (|+i⟩, eigenvalue +1), '-i' (|-i⟩, eigenvalue -1)

        n_plus = counts.get('0', 0) + counts.get('+', 0) + counts.get('+i', 0)
        n_minus = counts.get('1', 0) + counts.get('-', 0) + counts.get('-i', 0)
        total = n_plus + n_minus

        if total == 0:
            return 0.0

        return (n_plus - n_minus) / total

    def _reconstruct_lsq(self, output_states: List[DensityMatrix]) -> KrausChannel:
        """
        Реконструкция методом наименьших квадратов
        """
        lin_inv = LinearInversion()
        choi = lin_inv.reconstruct_choi(self.input_states, output_states)
        choi = lin_inv._project_to_cptp(choi)

        # Извлекаем операторы Крауса
        eigenvalues, eigenvectors = np.linalg.eigh(choi)

        threshold = 1e-10
        positive_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:, positive_indices]

        kraus_operators = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            kraus_operators.append(K)

        return KrausChannel(
            kraus_operators,
            n_qubits=1,
            name="LSQ_reconstructed",
            validate=False
        )

    def _reconstruct_mle(self, measurement_data: List[Dict[str, Dict[str, int]]]) -> KrausChannel:
        """
        Реконструкция методом максимального правдоподобия
        """
        mle = MaximumLikelihood()

        # Преобразуем данные в нужный формат
        # Для упрощения используем измерения в Z-базисе
        simplified_data = []

        for state_measurements in measurement_data:
            if 'Z' in state_measurements:
                simplified_data.append(state_measurements['Z'])
            else:
                # Используем первый доступный базис
                first_basis = list(state_measurements.keys())[0]
                simplified_data.append(state_measurements[first_basis])

        return mle.reconstruct(self.input_states, simplified_data, method='choi')

    def run_multiple_tomographies(self,
                                 unknown_channel: QuantumChannel,
                                 n_runs: int = 10,
                                 reconstruction_method: str = 'LSQ') -> List[QPTResult]:
        """
        Провести N независимых томографий для статистического анализа

        Args:
            unknown_channel: Канал для исследования
            n_runs: Число прогонов
            reconstruction_method: Метод реконструкции

        Returns:
            Список результатов
        """
        print(f"\n=== Множественная томография: {n_runs} прогонов ===")

        results = []
        for run in range(n_runs):
            print(f"\nПрогон {run + 1}/{n_runs}")
            result = self.run_tomography(unknown_channel, reconstruction_method)
            results.append(result)

        # Статистический анализ
        fidelities = [r.process_fidelity for r in results if r.process_fidelity is not None]

        if fidelities:
            mean_fidelity = np.mean(fidelities)
            std_fidelity = np.std(fidelities)

            print(f"\n=== Статистика ===")
            print(f"Средняя fidelity: {mean_fidelity:.6f} ± {std_fidelity:.6f}")
            print(f"Мин: {np.min(fidelities):.6f}, Макс: {np.max(fidelities):.6f}")

        return results
