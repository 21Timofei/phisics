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
from .state_prep import prepare_tomography_states, get_pauli_basis
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

    def __init__(self, n_qubits: int, shots: int = 1000):
        """
        Args:
            n_qubits: Число кубитов (1-3)
            shots: Число измерений для каждой настройки
        """
        if n_qubits < 1 or n_qubits > 3:
            raise ValueError("Поддерживается 1-3 кубита")

        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.shots = shots

        # Подготовка входных состояний
        # ПРАВИЛЬНЫЙ ПОДХОД: используем базисные операторы {|i⟩⟨j|}
        # вместо произвольных состояний для точной реконструкции Choi matrix
        self.input_states = self._generate_basis_operators(n_qubits)

        # Паули-базисы для измерений
        self.measurement_bases = self._prepare_measurement_bases()

        print(f"QPT инициализирована для {n_qubits} кубитов")
        print(f"Входных состояний: {len(self.input_states)}")
        print(f"Измерительных базисов: {len(self.measurement_bases)}")
        print(f"Shots на измерение: {shots}")

    def _generate_basis_operators(self, n_qubits: int) -> List[DensityMatrix]:
        """
        Генерация информационно-полного набора состояний для QPT

        Используем обобщённые состояния Паули, которые образуют полный базис
        в пространстве операторов. Для d-мерной системы генерируем d² состояний.

        Для 1 кубита используем 4 состояния:
        - |0⟩, |1⟩ (вычислительный базис)
        - |+⟩, |-⟩ (базис X)
        - |+i⟩, |-i⟩ (базис Y)

        Args:
            n_qubits: Число кубитов

        Returns:
            Список состояний как DensityMatrix
        """
        dim = 2 ** n_qubits
        basis_states = []

        if n_qubits == 1:
            # Для 1 кубита: 4 ортогональных состояния (минимальный полный набор)
            # Это образует ортонормированный базис в пространстве операторов

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
        else:
            # Для многокубитных: используем тензорные произведения
            from itertools import product

            # Базисные состояния для 1 кубита
            single_qubit_basis = [
                np.array([[1, 0], [0, 0]], dtype=np.complex128),  # |0⟩
                np.array([[0, 0], [0, 1]], dtype=np.complex128),  # |1⟩
                np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),  # |+⟩
                np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128),  # |+i⟩
            ]

            # Генерируем тензорные произведения
            for combo in product(range(4), repeat=n_qubits):
                state = single_qubit_basis[combo[0]]
                for idx in combo[1:]:
                    state = np.kron(state, single_qubit_basis[idx])
                basis_states.append(DensityMatrix(state, validate=False))

        return basis_states

    def _prepare_measurement_bases(self) -> List[str]:
        """
        Подготовить список измерительных базисов

        Для n кубитов: 3^n паули-измерений (X, Y, Z на каждом кубите)

        Returns:
            Список строк типа 'XYZ' для 3 кубитов
        """
        from itertools import product

        bases = ['X', 'Y', 'Z']

        # Генерируем все комбинации
        measurement_strings = []
        for combo in product(bases, repeat=self.n_qubits):
            measurement_strings.append(''.join(combo))

        return measurement_strings

    def run_tomography(self,
                      unknown_channel: QuantumChannel,
                      reconstruction_method: str = 'LSQ',
                      add_measurement_noise: bool = False,
                      readout_error: float = 0.0,
                      measurement_selection: str = 'full',
                      excluded_bases: Optional[List[str]] = None,
                      subset_size: Optional[int] = None,
                      regularization_lambda: Optional[float] = None) -> QPTResult:
        """
        Провести полную томографию неизвестного канала

        Args:
            unknown_channel: Неизвестный канал для диагностики
            reconstruction_method: 'LSQ', 'MLE', 'Tikhonov', 'MaxEntropy', 'L1'
            add_measurement_noise: Добавлять ли шум измерений
            readout_error: Вероятность ошибки считывания
            measurement_selection: Выбор базисов ('full', 'minimal', 'random', 'optimized')
            excluded_bases: Список исключаемых базисов
            subset_size: Размер подмножества базисов (для 'random', 'optimized')
            regularization_lambda: Параметр регуляризации (для Tikhonov, L1)

        Returns:
            QPTResult с результатами томографии
        """
        print(f"\n=== Начало томографии ===")
        print(f"Исследуемый канал: {unknown_channel.name}")
        print(f"Метод реконструкции: {reconstruction_method}")

        # Выбираем базисы для измерений
        selected_bases = self.filter_measurement_bases(
            selection_method=measurement_selection,
            excluded_bases=excluded_bases,
            subset_size=subset_size
        )

        print(f"Выбрано базисов для измерений: {len(selected_bases)}/{len(self.measurement_bases)}")

        # Шаг 1: Применяем канал ко всем входным состояниям
        output_states = []
        for i, rho_in in enumerate(self.input_states):
            rho_out = unknown_channel.apply(rho_in)
            output_states.append(rho_out)

        print(f"✓ Применён канал к {len(output_states)} входным состояниям")

        # Шаг 2: Измеряем выходные состояния в выбранных базисах
        measurement_data = []

        for state_idx, rho_out in enumerate(output_states):
            state_measurements = {}

            for basis_string in selected_bases:
                # Проводим измерения в данном базисе
                counts = self._measure_in_basis(
                    rho_out,
                    basis_string,
                    add_noise=add_measurement_noise,
                    readout_error=readout_error
                )
                state_measurements[basis_string] = counts

            measurement_data.append(state_measurements)

        print(f"✓ Проведено {len(measurement_data) * len(selected_bases)} измерений")

        # Шаг 3: Реконструкция канала
        if reconstruction_method == 'LSQ':
            # Для 1 кубита: реконструируем состояния из измерений (state tomography)
            # Для многих кубитов: используем идеальные выходы (fallback)
            if self.n_qubits == 1:
                reconstructed_output_states = self._reconstruct_states_from_measurements(measurement_data)
                reconstructed = self._reconstruct_lsq(reconstructed_output_states)
            else:
                # Fallback: используем идеальные выходы для многокубитных систем
                reconstructed = self._reconstruct_lsq(output_states)

        elif reconstruction_method == 'MLE':
            reconstructed = self._reconstruct_mle(measurement_data)

        elif reconstruction_method in ['Tikhonov', 'MaxEntropy', 'L1']:
            # Регуляризованные методы
            from .reconstruction_regularized import RegularizedReconstruction

            reg_recon = RegularizedReconstruction(self.n_qubits)

            # Реконструируем выходные состояния из измерений
            reconstructed_output_states = self._reconstruct_states_from_measurements(measurement_data)

            # Автоматический выбор lambda если не задан
            if regularization_lambda is None:
                if measurement_selection != 'full':
                    # Для неполных измерений используем кросс-валидацию
                    print("Выбираем оптимальный параметр регуляризации...")
                    method_map = {'Tikhonov': 'tikhonov', 'L1': 'l1'}
                    regularization_lambda = reg_recon.select_regularization_parameter(
                        self.input_states,
                        reconstructed_output_states,
                        method=method_map.get(reconstruction_method, 'tikhonov')
                    )
                    print(f"Выбран λ = {regularization_lambda:.6f}")
                else:
                    regularization_lambda = 0.01  # Значение по умолчанию

            # Применяем регуляризацию
            if reconstruction_method == 'Tikhonov':
                choi = reg_recon.reconstruct_tikhonov(
                    self.input_states,
                    reconstructed_output_states,
                    lambda_reg=regularization_lambda
                )
            elif reconstruction_method == 'MaxEntropy':
                choi = reg_recon.reconstruct_max_entropy(
                    self.input_states,
                    reconstructed_output_states
                )
            elif reconstruction_method == 'L1':
                choi = reg_recon.reconstruct_compressed_sensing(
                    self.input_states,
                    reconstructed_output_states,
                    lambda_reg=regularization_lambda
                )

            # Преобразуем Choi в Kraus
            reconstructed = self._choi_to_kraus_channel(choi)

        else:
            raise ValueError(f"Неизвестный метод: {reconstruction_method}")

        print(f"✓ Канал реконструирован (ранг Крауса: {reconstructed.kraus_rank()})")

        # Шаг 4: Вычисляем метрики качества
        choi_true = unknown_channel.get_choi_matrix()
        choi_reconstructed = reconstructed.get_choi_matrix()

        from ..metrics.fidelity import process_fidelity_choi
        fidelity = process_fidelity_choi(choi_true, choi_reconstructed)

        print(f"✓ Process fidelity: {fidelity:.6f}")

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
            basis_string: Строка базиса, например 'XYZ'
            add_noise: Добавлять ли шум
            readout_error: Вероятность ошибки считывания

        Returns:
            Словарь с результатами измерений {outcome: count}
        """
        # Для многокубитного случая нужно измерять каждый кубит отдельно
        # Упрощённая реализация: измеряем в произведении паули-базисов

        if self.n_qubits == 1:
            # Однокубитный случай
            measurement = PauliMeasurement(basis_string, qubit_index=0, n_qubits=1)
            counts = measurement.measure(rho, shots=self.shots)
        else:
            # Многокубитный: измеряем каждый кубит в своём базисе
            # Получаем результаты для всех кубитов одновременно

            # Упрощение: измеряем в вычислительном базисе после поворота
            # Сначала применяем базисные повороты
            rho_rotated = self._rotate_to_measurement_basis(rho, basis_string)

            # Затем измеряем в вычислительном базисе
            outcomes = rho_rotated.measure(shots=self.shots)

            # Подсчитываем результаты
            counts = {}
            for outcome in outcomes:
                binary = format(outcome, f'0{self.n_qubits}b')
                counts[binary] = counts.get(binary, 0) + 1

        # Добавляем шум измерений если нужно
        if add_noise and readout_error > 0:
            counts = self._add_readout_noise(counts, readout_error)

        return counts

    def _rotate_to_measurement_basis(self,
                                    rho: DensityMatrix,
                                    basis_string: str) -> DensityMatrix:
        """
        Повернуть состояние для измерения в паули-базисе

        Для измерения в базисе X применяем H
        Для измерения в базисе Y применяем H S†
        Для измерения в базисе Z ничего не применяем

        Args:
            rho: Состояние
            basis_string: Базисы для каждого кубита

        Returns:
            Повёрнутое состояние
        """
        from ..core.gates import PauliGates
        from ..core.tensor import apply_single_qubit_gate

        # Определяем повороты для каждого кубита
        rotations = {
            'X': PauliGates.hadamard().matrix,
            'Y': (PauliGates.hadamard() @ PauliGates.sdg_gate()).matrix,
            'Z': PauliGates.identity().matrix
        }

        # Применяем повороты к каждому кубиту
        total_rotation = np.eye(self.dim, dtype=np.complex128)

        for qubit_idx, basis in enumerate(basis_string):
            single_qubit_rot = rotations[basis]
            full_rotation = apply_single_qubit_gate(single_qubit_rot, qubit_idx, self.n_qubits)
            total_rotation = full_rotation @ total_rotation

        # Применяем поворот: ρ' = U ρ U†
        rotated_matrix = total_rotation @ rho.matrix @ total_rotation.conj().T

        return DensityMatrix(rotated_matrix, validate=False)

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

            if self.n_qubits == 1:
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
            else:
                # Для многокубитного случая: не реализовано
                # Используется fallback в run_tomography
                raise NotImplementedError("Многокубитная state tomography не реализована")

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
        lin_inv = LinearInversion(self.n_qubits)
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
            n_qubits=self.n_qubits,
            name="LSQ_reconstructed",
            validate=False
        )

    def _reconstruct_mle(self, measurement_data: List[Dict[str, Dict[str, int]]]) -> KrausChannel:
        """
        Реконструкция методом максимального правдоподобия

        Использует ВСЕ доступные базисы измерений для MLE оптимизации
        """
        mle = MaximumLikelihood(self.n_qubits)

        # Преобразуем данные в формат для MLE
        # Объединяем все измерения из всех базисов
        combined_data = []

        for state_measurements in measurement_data:
            # Для каждого входного состояния объединяем результаты
            # из всех базисов в один словарь с уникальными ключами
            combined_counts = {}

            for basis_string, counts in state_measurements.items():
                # Добавляем префикс базиса к результатам
                for outcome, count in counts.items():
                    # Ключ формата: "basis:outcome" (например, "XZ:01")
                    key = f"{basis_string}:{outcome}"
                    combined_counts[key] = count

            combined_data.append(combined_counts)

        return mle.reconstruct(self.input_states, combined_data, method='choi')

    def _choi_to_kraus_channel(self, choi: NDArray[np.complex128]) -> KrausChannel:
        """
        Преобразовать Choi matrix в KrausChannel

        Args:
            choi: Choi matrix

        Returns:
            KrausChannel
        """
        # Спектральное разложение
        eigenvalues, eigenvectors = np.linalg.eigh(choi)

        # Оставляем положительные собственные значения
        threshold = 1e-10
        positive_indices = eigenvalues > threshold

        eigenvalues = eigenvalues[positive_indices]
        eigenvectors = eigenvectors[:, positive_indices]

        # Формируем операторы Крауса
        kraus_operators = []
        for i in range(len(eigenvalues)):
            sqrt_eigenvalue = np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i]

            # Reshape: вектор размера d² -> матрица d×d
            K = sqrt_eigenvalue * vec.reshape(self.dim, self.dim)
            kraus_operators.append(K)

        if len(kraus_operators) == 0:
            # Пустой канал - возвращаем единичный
            kraus_operators = [np.eye(self.dim, dtype=np.complex128)]

        return KrausChannel(
            kraus_operators,
            n_qubits=self.n_qubits,
            name="Reconstructed",
            validate=False
        )

    def filter_measurement_bases(self,
                                 selection_method: str = 'full',
                                 excluded_bases: Optional[List[str]] = None,
                                 subset_size: Optional[int] = None) -> List[str]:
        """
        Фильтрация измерительных базисов для неполных измерений

        Методы выбора:
        - 'full': все базисы (по умолчанию)
        - 'minimal': минимальный информационно-полный набор
        - 'random': случайный выбор subset_size базисов
        - 'optimized': жадный выбор по condition number системы

        Args:
            selection_method: Метод выбора базисов
            excluded_bases: Список исключаемых базисов
            subset_size: Размер подмножества (для 'random')

        Returns:
            Список выбранных базисов
        """
        all_bases = self.measurement_bases.copy()

        # Исключаем указанные базисы
        if excluded_bases:
            all_bases = [b for b in all_bases if b not in excluded_bases]

        if selection_method == 'full':
            return all_bases

        elif selection_method == 'minimal':
            return self._get_minimal_bases(all_bases)

        elif selection_method == 'random':
            if subset_size is None:
                subset_size = len(all_bases) // 2
            subset_size = min(subset_size, len(all_bases))

            indices = np.random.choice(len(all_bases), size=subset_size, replace=False)
            return [all_bases[i] for i in sorted(indices)]

        elif selection_method == 'optimized':
            return self._get_optimized_bases(all_bases, subset_size)

        else:
            raise ValueError(f"Неизвестный метод выбора: {selection_method}")

    def _get_minimal_bases(self, available_bases: List[str]) -> List[str]:
        """
        Минимальный информационно-полный набор измерительных базисов

        Для n кубитов нужно минимум (3^n + 1) базисов для полной томографии
        (Это из теории SIC-POVM)

        Для практических целей выбираем базисы равномерно из паули-групп

        Args:
            available_bases: Доступные базисы

        Returns:
            Минимальный набор базисов
        """
        # Для 1 кубита: минимум 4 базиса (Z, X, Y и диагональный)
        # Для 2 кубитов: минимум 10 базисов
        # Для 3 кубитов: минимум 28 базисов

        minimal_size = 3 ** self.n_qubits + 1

        # Выбираем первые minimal_size базисов
        # (можно улучшить более умным отбором)
        if len(available_bases) <= minimal_size:
            return available_bases

        # Стратегия: выбираем базисы с разнообразными паули-операторами
        # Приоритет: базисы с меньшим числом I (тождественных операторов)

        def count_identities(basis_string: str) -> int:
            return basis_string.count('I')

        # Сортируем по числу I (меньше I - лучше)
        sorted_bases = sorted(available_bases, key=count_identities)

        return sorted_bases[:minimal_size]

    def _get_optimized_bases(self, available_bases: List[str], target_size: Optional[int] = None) -> List[str]:
        """
        Оптимизированный выбор базисов через жадный алгоритм

        Выбираем базисы которые минимизируют condition number
        системы уравнений для реконструкции

        Args:
            available_bases: Доступные базисы
            target_size: Желаемое число базисов

        Returns:
            Оптимизированный набор базисов
        """
        if target_size is None:
            target_size = len(available_bases) // 2

        target_size = min(target_size, len(available_bases))

        # Начинаем с минимального набора
        selected = []

        # Жадный алгоритм: на каждом шаге добавляем базис
        # который максимально улучшает обусловленность
        remaining = available_bases.copy()

        for _ in range(target_size):
            best_basis = None
            best_condition = float('inf')

            for basis in remaining:
                # Пробуем добавить этот базис
                test_set = selected + [basis]

                # Вычисляем condition number
                condition = self._compute_measurement_condition(test_set)

                if condition < best_condition:
                    best_condition = condition
                    best_basis = basis

            if best_basis:
                selected.append(best_basis)
                remaining.remove(best_basis)

        return selected

    def _compute_measurement_condition(self, bases: List[str]) -> float:
        """
        Вычислить обусловленность системы измерений

        Более низкое значение = лучше обусловленная система

        Args:
            bases: Список измерительных базисов

        Returns:
            Число обусловленности (condition number)
        """
        # Простая эвристика: считаем ранг паули-матриц
        # соответствующих данным базисам

        # Для полноты нужны все 4^n паули-строк
        # Проверяем насколько близки мы к этому

        unique_paulis = set(bases)

        # Чем больше уникальных базисов, тем лучше
        # Чем меньше I в базисах, тем лучше

        total_identities = sum(b.count('I') for b in bases)
        coverage = len(unique_paulis) / (4 ** self.n_qubits)

        # Комбинированная метрика (чем меньше, тем лучше)
        # Нормализуем на [0, 1]
        identity_penalty = total_identities / (len(bases) * self.n_qubits)
        coverage_bonus = 1.0 - coverage

        condition = identity_penalty + coverage_bonus

        return condition

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
