"""
NoiseLab++ - Интерактивный веб-интерфейс для квантовой томографии процессов
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import sys
import os

# Добавляем путь к модулю
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noiselab.channels.noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel
)
from noiselab.channels.two_qubit_noise import TwoQubitDepolarizing
from noiselab.channels.random import random_cptp_channel
from noiselab.tomography.qpt import QuantumProcessTomography
from noiselab.metrics.validation import (
    analyze_tomography_quality,
    estimate_error_rates,
    statistical_analysis_multiple_runs
)
from noiselab.channels.kraus import KrausChannel


# Конфигурация страницы
st.set_page_config(
    page_title="NoiseLab++ | Quantum Process Tomography",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        padding: 0.5rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def create_channel(channel_type, params, n_qubits):
    """Создать квантовый канал по типу и параметрам"""
    if channel_type == 'Depolarizing':
        p = params.get('p', 0.1)
        return DepolarizingChannel(p, n_qubits=n_qubits)

    elif channel_type == 'Amplitude Damping' and n_qubits == 1:
        gamma = params.get('gamma', 0.3)
        return AmplitudeDampingChannel(gamma)

    elif channel_type == 'Phase Damping' and n_qubits == 1:
        lambda_ = params.get('lambda', 0.2)
        return PhaseDampingChannel(lambda_)

    elif channel_type == 'Two-Qubit Depolarizing' and n_qubits == 2:
        p = params.get('p', 0.1)

        # Проверяем модель корреляций
        correlation_model = params.get('correlation_model', 'symmetric')

        if correlation_model == 'asymmetric':
            from noiselab.channels.two_qubit_noise import GeneralCorrelatedNoise
            p1 = params.get('p1', 0.05)
            p2 = params.get('p2', 0.05)
            p_corr = params.get('p_corr', 0.01)
            return GeneralCorrelatedNoise.asymmetric_depolarizing(p1, p2, p_corr)

        elif correlation_model == 'general':
            from noiselab.channels.two_qubit_noise import GeneralCorrelatedNoise
            # Используем случайные параметры для демонстрации
            import numpy as np
            error_probs = np.random.dirichlet(np.ones(16)) * p
            return GeneralCorrelatedNoise(error_probabilities=error_probs.reshape(4, 4))

        else:
            # Symmetric (по умолчанию)
            return TwoQubitDepolarizing(p)

    # Для 3 кубитов DepolarizingChannel уже поддерживает n_qubits=3

    elif channel_type == 'Random CPTP':
        seed = params.get('seed', None)
        return random_cptp_channel(n_qubits, seed=seed)

    return None


def plot_choi_matrix(choi_matrix):
    """Визуализация матрицы Чой (действительная и мнимая части)"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Действительная часть', 'Мнимая часть'),
        horizontal_spacing=0.15
    )

    # Действительная часть
    fig.add_trace(
        go.Heatmap(
            z=np.real(choi_matrix),
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(x=0.45, len=0.9),
            showscale=True
        ),
        row=1, col=1
    )

    # Мнимая часть
    fig.add_trace(
        go.Heatmap(
            z=np.imag(choi_matrix),
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(x=1.02, len=0.9),
            showscale=True
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        title_text="Матрица Чой канала",
        title_x=0.5,
        showlegend=False
    )

    return fig


def plot_kraus_operators(kraus_ops):
    """Гистограмма весов операторов Крауса"""
    weights = []
    for K in kraus_ops:
        weight = np.trace(K.conj().T @ K).real
        weights.append(weight)

    fig = go.Figure(data=[
        go.Bar(
            x=list(range(len(weights))),
            y=weights,
            marker=dict(
                color=weights,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Вес")
            ),
            text=[f'{w:.4f}' for w in weights],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Веса операторов Крауса",
        xaxis_title="Индекс оператора",
        yaxis_title="Вес (Tr[K†K])",
        height=400,
        showlegend=False
    )

    return fig


def plot_ptm_matrix(channel):
    """Визуализация PTM (Pauli Transfer Matrix)"""
    try:
        from noiselab.representations.ptm import PauliTransferMatrix

        # Создаем PTM из канала
        ptm_obj = PauliTransferMatrix.from_channel(channel)
        ptm = ptm_obj.ptm_matrix

        fig = go.Figure(data=go.Heatmap(
            z=ptm,
            colorscale='RdBu',
            zmid=0,
            colorbar=dict(title="Значение")
        ))

        fig.update_layout(
            title="Pauli Transfer Matrix (PTM)",
            xaxis_title="Выходной базис Паули",
            yaxis_title="Входной базис Паули",
            height=500
        )

        return fig
    except Exception as e:
        st.error(f"Ошибка при построении PTM: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def plot_bloch_sphere_trajectory(channel, n_points=20):
    """3D представление действия канала через траектории на сфере Блоха (только для 1 кубита)"""
    try:
        if channel.n_qubits != 1:
            st.warning("Визуализация сферы Блоха доступна только для 1-кубитных каналов")
            return None

        from noiselab.core.states import DensityMatrix

        # Генерируем точки на сфере Блоха
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2*np.pi, n_points)

        # Сфера Блоха
        x_sphere = np.outer(np.sin(theta), np.cos(phi))
        y_sphere = np.outer(np.sin(theta), np.sin(phi))
        z_sphere = np.outer(np.cos(theta), np.ones(n_points))

        fig = go.Figure()

        # Добавляем сферу
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.2,
            colorscale='Blues',
            showscale=False,
            name='Сфера Блоха'
        ))

        # Тестовые состояния и их эволюция
        test_states = [
            (np.array([[1], [0]], dtype=complex), 'red', '|0⟩'),
            (np.array([[0], [1]], dtype=complex), 'blue', '|1⟩'),
            (np.array([[1], [1]], dtype=complex)/np.sqrt(2), 'green', '|+⟩'),
            (np.array([[1], [-1]], dtype=complex)/np.sqrt(2), 'orange', '|-⟩'),
        ]

        for state, color, label in test_states:
            # Начальное состояние
            rho_in_matrix = state @ state.conj().T
            rho_in = DensityMatrix(rho_in_matrix, validate=False)

            # Применяем канал
            rho_out = channel.apply(rho_in)
            rho_out_matrix = rho_out.matrix if hasattr(rho_out, 'matrix') else rho_out

            # Вычисляем координаты Блоха
            # r = Tr(ρ σ) для σ = σx, σy, σz
            sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
            sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
            sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

            x_in = np.real(np.trace(rho_in_matrix @ sigma_x))
            y_in = np.real(np.trace(rho_in_matrix @ sigma_y))
            z_in = np.real(np.trace(rho_in_matrix @ sigma_z))

            x_out = np.real(np.trace(rho_out_matrix @ sigma_x))
            y_out = np.real(np.trace(rho_out_matrix @ sigma_y))
            z_out = np.real(np.trace(rho_out_matrix @ sigma_z))

            # Точка входа
            fig.add_trace(go.Scatter3d(
                x=[x_in], y=[y_in], z=[z_in],
                mode='markers',
                marker=dict(size=8, color=color),
                name=f'{label} (вход)',
                showlegend=True
            ))

            # Точка выхода
            fig.add_trace(go.Scatter3d(
                x=[x_out], y=[y_out], z=[z_out],
                mode='markers',
                marker=dict(size=8, color=color, symbol='diamond'),
                name=f'{label} (выход)',
                showlegend=True
            ))

            # Траектория
            fig.add_trace(go.Scatter3d(
                x=[x_in, x_out],
                y=[y_in, y_out],
                z=[z_in, z_out],
                mode='lines',
                line=dict(color=color, width=4, dash='dash'),
                showlegend=False
            ))

        fig.update_layout(
            title="Действие канала на сфере Блоха",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode='cube'
            ),
            height=600,
            showlegend=True
        )

        return fig
    except Exception as e:
        st.error(f"Ошибка при построении сферы Блоха: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def main():
    """Главная функция приложения"""

    # Заголовок
    st.markdown('<div class="main-header">⚛️ NoiseLab++ Quantum Process Tomography</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Интерактивная платформа для исследования квантовых каналов и томографии процессов</div>', unsafe_allow_html=True)

    # Боковая панель управления
    with st.sidebar:
        st.header("⚙️ Панель управления")

        # 1. Выбор числа кубитов
        st.subheader("1️⃣ Число кубитов")
        n_qubits = st.selectbox("Выберите число кубитов", [1, 2, 3], index=0)

        if n_qubits == 3:
            st.warning("⚠️ 3-кубитные системы требуют больше ресурсов и времени вычислений")

        # 2. Выбор канала
        st.subheader("2️⃣ Квантовый канал")

        if n_qubits == 1:
            channel_types = ['Depolarizing', 'Amplitude Damping', 'Phase Damping', 'Random CPTP']
        elif n_qubits == 2:
            channel_types = ['Depolarizing', 'Two-Qubit Depolarizing', 'Random CPTP']
        else:  # n_qubits == 3
            channel_types = ['Depolarizing', 'Random CPTP']

        channel_type = st.selectbox("Тип канала", channel_types)

        # Параметры канала
        channel_params = {}

        if channel_type == 'Depolarizing':
            max_p = 0.75 if n_qubits == 1 else (0.5 if n_qubits == 2 else 0.3)
            p = st.slider("Параметр p (вероятность деполяризации)", 0.0, max_p, 0.1, 0.01)
            channel_params['p'] = p
            num_kraus = 4 ** n_qubits
            st.info(f"💡 Depolarizing: E(ρ) = (1-p)ρ + p·I/{2**n_qubits}")
            st.caption(f"Генерируется {num_kraus} операторов Крауса (4^{n_qubits})")

        elif channel_type == 'Amplitude Damping':
            gamma = st.slider("Параметр γ (затухание амплитуды)", 0.0, 1.0, 0.3, 0.01)
            channel_params['gamma'] = gamma
            st.info("💡 Моделирует релаксацию энергии (потерю фотона)")

        elif channel_type == 'Phase Damping':
            lambda_ = st.slider("Параметр λ (затухание фазы)", 0.0, 0.5, 0.2, 0.01)
            channel_params['lambda'] = lambda_
            st.info("💡 Моделирует декогеренцию без потери энергии")

        elif channel_type == 'Two-Qubit Depolarizing':
            p = st.slider("Параметр p (2-кубитная деполяризация)", 0.0, 0.5, 0.1, 0.01)
            channel_params['p'] = p
            st.info("💡 Two-qubit depolarizing channel")

            # Выбор модели корреляций
            correlation_model = st.selectbox("Модель корреляций", ['Symmetric', 'Asymmetric', 'General'])
            if correlation_model == 'Asymmetric':
                p1 = st.slider("Параметр p1 (кубит 1)", 0.0, 0.5, 0.05, 0.01)
                p2 = st.slider("Параметр p2 (кубит 2)", 0.0, 0.5, 0.05, 0.01)
                p_corr = st.slider("Корреляция p_corr", 0.0, 0.2, 0.01, 0.01)
                channel_params['correlation_model'] = 'asymmetric'
                channel_params['p1'] = p1
                channel_params['p2'] = p2
                channel_params['p_corr'] = p_corr
            elif correlation_model == 'General':
                st.info("💡 Полная параметризация 16 паули-ошибок")
                # Используем случайные параметры для General модели
                channel_params['correlation_model'] = 'general'
            else:
                channel_params['correlation_model'] = 'symmetric'

        elif channel_type == 'Random CPTP':
            use_seed = st.checkbox("Использовать seed", value=False)
            if use_seed:
                seed = st.number_input("Seed", value=42, step=1)
                channel_params['seed'] = seed
            st.info("💡 Случайный CPTP канал (Choi matrix)")

        # 3. Параметры томографии
        st.subheader("3️⃣ Томография")
        # Для 3 кубитов нужно больше shots для точности
        default_shots = 1000 if n_qubits <= 2 else 3000
        max_shots = 100000 if n_qubits <= 2 else 50000  # Ограничение для 3 кубитов
        shots = st.number_input("Число измерений (shots)", min_value=100, max_value=max_shots, value=default_shots, step=100)

        if n_qubits == 3:
            st.info(f"💡 Для 3 кубитов рекомендуется минимум 3000 shots для приемлемой точности")

        st.subheader("4️⃣ Шум измерений")
        add_noise = st.checkbox("Добавить ошибки считывания", value=False)
        readout_error = 0.0
        if add_noise:
            readout_error = st.slider("Ошибка считывания", 0.0, 0.1, 0.01, 0.001)

        st.subheader("5️⃣ Алгоритм")
        method = st.selectbox("Метод реконструкции", ['LSQ', 'MLE', 'Tikhonov', 'MaxEntropy', 'L1'])

        # Параметры регуляризации
        regularization_lambda = None
        if method in ['Tikhonov', 'L1']:
            use_auto_lambda = st.checkbox("Автоматический выбор λ", value=True)
            if not use_auto_lambda:
                regularization_lambda = st.slider("Параметр регуляризации λ", 0.0001, 1.0, 0.01, 0.0001, format="%.4f")
            else:
                st.info("💡 λ будет выбран через кросс-валидацию")

        # Выбор режима измерений
        st.subheader("6️⃣ Измерения")
        measurement_selection = st.selectbox(
            "Режим измерений",
            ['full', 'minimal', 'random', 'optimized'],
            help="full: все базисы, minimal: минимальный набор, random: случайный выбор, optimized: оптимизация по condition number"
        )

        subset_size = None
        if measurement_selection in ['random', 'optimized']:
            # Вычисляем общее число базисов
            n_bases = 4 ** n_qubits
            subset_size = st.slider(
                "Число базисов",
                min_value=3 ** n_qubits + 1,  # Минимум
                max_value=n_bases,
                value=n_bases // 2
            )

        # Исключение базисов
        exclude_bases = st.multiselect(
            "Исключить базисы (опционально)",
            [],  # Будет заполнено динамически
            help="Исключить определённые базисы из измерений"
        )

        st.markdown("---")

        # Кнопка запуска
        run_button = st.button("🚀 Запустить томографию", type="primary", use_container_width=True)

        st.markdown("---")

        # Дополнительные опции
        st.subheader("📊 Дополнительно")
        show_protocol = st.checkbox("Показать протокол", value=False)

        # Множественные прогоны
        st.subheader("🔄 Статистический анализ")
        run_multiple = st.checkbox("Множественные прогоны", value=False)
        if run_multiple:
            n_runs = st.slider("Число прогонов", 5, 50, 10)
        else:
            n_runs = 1

    # Основная область
    if run_button:
        with st.spinner('🔬 Выполняется квантовая томография процесса...'):
            try:
                # Создаём канал
                true_channel = create_channel(channel_type, channel_params, n_qubits)

                if true_channel is None:
                    st.error("❌ Не удалось создать канал")
                    return

                # Инициализация QPT
                qpt = QuantumProcessTomography(n_qubits=n_qubits, shots=shots)

                if not run_multiple:
                    # Единичный прогон
                    result = qpt.run_tomography(
                        true_channel,
                        reconstruction_method=method,
                        add_measurement_noise=add_noise,
                        readout_error=readout_error,
                        measurement_selection=measurement_selection,
                        excluded_bases=exclude_bases if exclude_bases else None,
                        subset_size=subset_size,
                        regularization_lambda=regularization_lambda
                    )

                    # Анализ качества
                    quality = analyze_tomography_quality(result)

                    # Сохраняем в session state
                    st.session_state['result'] = result
                    st.session_state['quality'] = quality
                    st.session_state['true_channel'] = true_channel
                    st.session_state['channel_params'] = channel_params
                    st.session_state['channel_type'] = channel_type

                else:
                    # Множественные прогоны
                    # Примечание: run_multiple_tomographies не поддерживает все новые параметры
                    # Используем базовые параметры
                    results = qpt.run_multiple_tomographies(
                        true_channel,
                        n_runs=n_runs,
                        reconstruction_method=method
                    )
                    stats = statistical_analysis_multiple_runs(results)

                    st.session_state['results_multiple'] = results
                    st.session_state['stats'] = stats
                    st.session_state['true_channel'] = true_channel
                    st.session_state['channel_type'] = channel_type

                st.success('✅ Томография успешно завершена!')

            except Exception as e:
                st.error(f"❌ Ошибка при выполнении томографии: {e}")
                import traceback
                st.code(traceback.format_exc())

    # Отображение результатов
    if 'result' in st.session_state and not run_multiple:
        result = st.session_state['result']
        quality = st.session_state['quality']
        true_channel = st.session_state['true_channel']
        channel_params = st.session_state['channel_params']
        channel_type_stored = st.session_state['channel_type']

        # Метрики в карточках
        st.markdown("### 📈 Ключевые метрики")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                label="Process Fidelity",
                value=f"{result.process_fidelity:.4f}",
                delta=f"{(result.process_fidelity - 0.95):.4f}" if result.process_fidelity > 0.95 else None
            )

        with col2:
            cptp_status = "✅ Да" if quality['is_cptp'] else "❌ Нет"
            st.metric(label="CPTP", value=cptp_status)

        with col3:
            st.metric(label="TP ошибка", value=f"{quality['tp_error']:.2e}")

        with col4:
            st.metric(label="Ранг Крауса", value=f"{quality['kraus_rank']}")

        # Табы для визуализации
        tabs = st.tabs(["📊 Визуализация каналов", "📉 Графики и аналитика", "📝 Протокол"])

        with tabs[0]:
            st.markdown("### Визуализация квантового канала")

            # Матрица Чой
            st.subheader("Матрица Чой")
            choi = result.reconstructed_channel.get_choi_matrix()
            fig_choi = plot_choi_matrix(choi)
            st.plotly_chart(fig_choi, width='stretch')

            # Операторы Крауса
            st.subheader("Операторы Крауса")
            kraus_ops = result.reconstructed_channel.get_kraus_operators()
            fig_kraus = plot_kraus_operators(kraus_ops)
            st.plotly_chart(fig_kraus, width='stretch')

            # Детали операторов Крауса
            with st.expander("🔍 Детали операторов Крауса"):
                for i, K in enumerate(kraus_ops[:5]):  # Показываем первые 5
                    st.markdown(f"**Оператор K_{i}**")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.text("Действительная часть:")
                        st.dataframe(pd.DataFrame(np.real(K)), width='stretch')
                    with col_b:
                        st.text("Мнимая часть:")
                        st.dataframe(pd.DataFrame(np.imag(K)), width='stretch')

            # PTM
            st.subheader("Pauli Transfer Matrix")
            fig_ptm = plot_ptm_matrix(result.reconstructed_channel)
            if fig_ptm:
                st.plotly_chart(fig_ptm, width='stretch')

            # Блох-сфера (только для 1 кубита)
            if n_qubits == 1:
                st.subheader("3D представление на сфере Блоха")
                fig_bloch = plot_bloch_sphere_trajectory(result.reconstructed_channel)
                if fig_bloch:
                    st.plotly_chart(fig_bloch, width='stretch')
            elif n_qubits == 3:
                st.info("ℹ️ Визуализация сферы Блоха доступна только для 1-кубитных систем")

        with tabs[1]:
            st.markdown("### Аналитика и сравнение")

            # Сравнение параметров
            st.subheader("Сравнение истинных и восстановленных параметров")

            if channel_type_stored in ['Depolarizing', 'Amplitude Damping', 'Phase Damping']:
                try:
                    identity = KrausChannel.from_unitary(np.eye(2**n_qubits, dtype=complex), name="Identity")
                    estimated = estimate_error_rates(
                        result.reconstructed_channel,
                        identity,
                        error_model=channel_type_stored.lower().replace(' ', '_')
                    )

                    param_name = 'p' if 'Depolarizing' in channel_type_stored else ('gamma' if 'Amplitude' in channel_type_stored else 'lambda')
                    true_val = channel_params.get(param_name, 0)
                    est_val = estimated.get('parameter', 0)

                    comp_df = pd.DataFrame({
                        'Параметр': [param_name],
                        'Истинное значение': [true_val],
                        'Восстановленное значение': [est_val],
                        'Относительная ошибка (%)': [abs(true_val - est_val) / true_val * 100 if true_val > 0 else 0]
                    })

                    st.dataframe(comp_df, width='stretch')

                    # График сравнения
                    fig_comp = go.Figure(data=[
                        go.Bar(name='Истинное', x=[param_name], y=[true_val], marker_color='lightblue'),
                        go.Bar(name='Восстановленное', x=[param_name], y=[est_val], marker_color='salmon')
                    ])
                    fig_comp.update_layout(
                        title="Сравнение параметров",
                        yaxis_title="Значение",
                        height=400
                    )
                    st.plotly_chart(fig_comp, width='stretch')

                except Exception as e:
                    st.warning(f"Не удалось оценить параметры: {e}")

            # Дополнительная информация
            st.subheader("Дополнительная информация")
            info_df = pd.DataFrame({
                'Метрика': [
                    'Process Fidelity',
                    'Число операторов Крауса',
                    'Ранг Крауса',
                    'TP ошибка',
                    'CPTP валидность'
                ],
                'Значение': [
                    f"{result.process_fidelity:.6f}",
                    str(quality['n_kraus_operators']),
                    str(quality['kraus_rank']),
                    f"{quality['tp_error']:.2e}",
                    "Да" if quality['is_cptp'] else "Нет"
                ]
            })
            st.dataframe(info_df, width='stretch')

        with tabs[2]:
            if show_protocol:
                st.markdown("### 📝 Протокол томографии")

                # Подготовленные состояния
                st.subheader("Подготовленные состояния")
                basis_size = 2 ** n_qubits
                n_states = basis_size ** 2
                st.write(f"Всего состояний: {n_states} (полный базис)")

                # Информация о состояниях
                st.info(f"Использован полный базис из {n_states} состояний в пространстве размерности {basis_size}")
                with st.expander("Подробнее о базисе"):
                    st.markdown("""
                    Для томографии используются состояния в вычислительном базисе:
                    - Для 1 кубита: |0⟩, |1⟩, |+⟩, |-⟩
                    - Для 2 кубитов: |00⟩, |01⟩, |10⟩, |11⟩ и их суперпозиции
                    """)

                # Параметры измерений
                st.subheader("Параметры измерений")
                meas_info = pd.DataFrame({
                    'Параметр': ['Shots', 'Базис измерений', 'Ошибка считывания', 'Метод реконструкции'],
                    'Значение': [str(shots), 'Pauli basis', f"{readout_error:.3f}" if add_noise else "0.000", method]
                })
                st.dataframe(meas_info, width='stretch')

                # Этапы реконструкции
                st.subheader("Этапы реконструкции")
                stages = [
                    "1️⃣ Подготовка входных состояний",
                    "2️⃣ Применение канала",
                    "3️⃣ Измерение в базисе Паули",
                    "4️⃣ Построение матрицы Чой из измерений",
                    f"5️⃣ {method} оптимизация для CPTP",
                    "6️⃣ Декомпозиция Крауса",
                    "7️⃣ Вычисление метрик качества"
                ]
                for stage in stages:
                    st.markdown(stage)
            else:
                st.info("Включите опцию 'Показать протокол' в боковой панели")

    # Множественные прогоны
    if 'results_multiple' in st.session_state and run_multiple:
        results_multiple = st.session_state['results_multiple']
        stats = st.session_state['stats']

        st.markdown("### 🔄 Статистический анализ (множественные прогоны)")

        # Метрики
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Средняя fidelity", f"{stats['fidelity']['mean']:.4f}")

        with col2:
            st.metric("Std отклонение", f"{stats['fidelity']['std']:.4f}")

        with col3:
            st.metric("Min fidelity", f"{stats['fidelity']['min']:.4f}")

        with col4:
            st.metric("Max fidelity", f"{stats['fidelity']['max']:.4f}")

        # Распределение fidelity
        st.subheader("Распределение Process Fidelity")

        fidelities = [r.process_fidelity for r in results_multiple]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=fidelities,
            nbinsx=20,
            marker_color='lightskyblue',
            opacity=0.75
        ))
        fig_hist.update_layout(
            title=f"Гистограмма Process Fidelity ({n_runs} прогонов)",
            xaxis_title="Process Fidelity",
            yaxis_title="Частота",
            height=400
        )
        st.plotly_chart(fig_hist, width='stretch')

        # Временной ряд
        st.subheader("Эволюция Fidelity по прогонам")

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(
            x=list(range(1, n_runs + 1)),
            y=fidelities,
            mode='lines+markers',
            marker=dict(size=8, color='royalblue'),
            line=dict(color='royalblue', width=2)
        ))
        fig_ts.add_hline(
            y=stats['fidelity']['mean'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Среднее: {stats['fidelity']['mean']:.4f}"
        )
        fig_ts.update_layout(
            title="Process Fidelity vs. Номер прогона",
            xaxis_title="Номер прогона",
            yaxis_title="Process Fidelity",
            height=400
        )
        st.plotly_chart(fig_ts, width='stretch')

        # Статистика
        st.subheader("Детальная статистика")

        stats_df = pd.DataFrame({
            'Метрика': ['Среднее', 'Стд. отклонение', 'Минимум', 'Максимум', 'Медиана'],
            'Process Fidelity': [
                f"{stats['fidelity']['mean']:.6f}",
                f"{stats['fidelity']['std']:.6f}",
                f"{stats['fidelity']['min']:.6f}",
                f"{stats['fidelity']['max']:.6f}",
                f"{stats['fidelity']['median']:.6f}"
            ],
            'Kraus Rank': [
                f"{stats['kraus_rank']['mean']:.2f}",
                f"{stats['kraus_rank']['std']:.2f}",
                '-',
                '-',
                '-'
            ]
        })
        st.dataframe(stats_df, width='stretch')

        # Дополнительная информация о Kraus rank
        st.info(f"Наиболее частый Kraus Rank (мода): {stats['kraus_rank']['mode']}")

    # Информация о проекте
    st.markdown("---")
    with st.expander("ℹ️ О проекте NoiseLab++"):
        st.markdown("""
        **NoiseLab++** - это комплексная платформа для исследования квантовых каналов и томографии процессов.

        **Возможности:**
        - Симуляция различных моделей квантового шума (1-3 кубита)
        - Квантовая томография процессов (QPT) с методами LSQ и MLE
        - Полная визуализация: матрицы Чой, операторы Крауса, PTM, сфера Блоха
        - Статистический анализ с множественными прогонами
        - Метрики качества: process fidelity, CPTP validation, error estimation

        **Технологии:**
        - Python 3.11+
        - NumPy, SciPy для численных вычислений
        - Qiskit для квантовых операций
        - Plotly для интерактивных графиков
        - Streamlit для веб-интерфейса

        ---

        Разработано как часть исследовательского проекта по квантовым вычислениям и томографии.
        """)


if __name__ == '__main__':
    main()
