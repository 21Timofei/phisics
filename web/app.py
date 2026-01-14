"""
Flask API сервер для NoiseLab++ веб-интерфейса
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from noiselab.channels.noise_models import (
    DepolarizingChannel,
    AmplitudeDampingChannel,
    PhaseDampingChannel
)
from noiselab.channels.random import random_cptp_channel
from noiselab.tomography.qpt import QuantumProcessTomography
from noiselab.metrics.validation import analyze_tomography_quality, estimate_error_rates
from noiselab.channels.kraus import KrausChannel


app = Flask(__name__)
CORS(app)  # Разрешаем кросс-доменные запросы


@app.route('/api/health', methods=['GET'])
def health_check():
    """Проверка работоспособности сервера"""
    return jsonify({"status": "ok", "message": "NoiseLab++ API is running"})


@app.route('/api/channels/list', methods=['GET'])
def list_channels():
    """Получить список доступных типов каналов"""
    channels = {
        "1_qubit": [
            {"id": "depolarizing", "name": "Depolarizing Channel",
             "parameters": [{"name": "p", "min": 0, "max": 0.75, "default": 0.1}]},
            {"id": "amplitude_damping", "name": "Amplitude Damping",
             "parameters": [{"name": "gamma", "min": 0, "max": 1, "default": 0.3}]},
            {"id": "phase_damping", "name": "Phase Damping",
             "parameters": [{"name": "lambda", "min": 0, "max": 0.5, "default": 0.2}]},
            {"id": "random", "name": "Random CPTP Channel", "parameters": []}
        ],
        "2_qubit": [
            {"id": "two_qubit_depolarizing", "name": "Two-Qubit Depolarizing",
             "parameters": [{"name": "p", "min": 0, "max": 0.5, "default": 0.1}]},
            {"id": "random", "name": "Random CPTP Channel", "parameters": []}
        ]
    }
    return jsonify(channels)


@app.route('/api/tomography/run', methods=['POST'])
def run_tomography():
    """
    Запустить квантовую томографию

    POST данные:
    {
        "n_qubits": 1 или 2,
        "channel_type": "depolarizing" и т.д.,
        "channel_params": {"p": 0.1, ...},
        "shots": 1000,
        "method": "LSQ" или "MLE",
        "readout_error": 0.0
    }
    """
    try:
        data = request.json

        # Параметры
        n_qubits = data.get('n_qubits', 1)
        channel_type = data.get('channel_type', 'depolarizing')
        channel_params = data.get('channel_params', {})
        shots = data.get('shots', 1000)
        method = data.get('method', 'LSQ')
        readout_error = data.get('readout_error', 0.0)

        # Создаём канал
        channel = create_channel(channel_type, channel_params, n_qubits)

        if channel is None:
            return jsonify({"error": "Invalid channel type"}), 400

        # Запускаем томографию
        qpt = QuantumProcessTomography(n_qubits=n_qubits, shots=shots)
        result = qpt.run_tomography(
            channel,
            reconstruction_method=method,
            add_measurement_noise=(readout_error > 0),
            readout_error=readout_error
        )

        # Анализ качества
        quality = analyze_tomography_quality(result)

        # Choi matrix (для визуализации)
        choi = result.reconstructed_channel.get_choi_matrix()
        choi_real = choi.real.tolist()
        choi_imag = choi.imag.tolist()

        # Операторы Крауса
        kraus_ops = result.reconstructed_channel.get_kraus_operators()
        kraus_info = []
        for i, K in enumerate(kraus_ops):
            weight = np.trace(K.conj().T @ K).real
            kraus_info.append({
                "index": i,
                "weight": float(weight),
                "matrix_real": K.real.tolist(),
                "matrix_imag": K.imag.tolist()
            })

        # Оценка параметров
        estimated_params = {}
        if channel_type in ['depolarizing', 'amplitude_damping', 'phase_damping']:
            identity = KrausChannel.from_unitary(np.eye(2**n_qubits), name="Identity")
            error_model = channel_type.replace('_', ' ')
            try:
                estimated = estimate_error_rates(
                    result.reconstructed_channel,
                    identity,
                    error_model=channel_type
                )
                estimated_params = {
                    "parameter": float(estimated.get('parameter', 0)),
                    "fit_fidelity": float(estimated.get('fit_fidelity', 0))
                }
            except:
                pass

        # Формируем ответ
        response = {
            "success": True,
            "result": {
                "process_fidelity": float(result.process_fidelity),
                "is_cptp": quality['is_cptp'],
                "tp_error": float(quality['tp_error']),
                "kraus_rank": quality['kraus_rank'],
                "n_kraus_operators": quality['n_kraus_operators'],
                "choi_matrix": {
                    "real": choi_real,
                    "imag": choi_imag
                },
                "kraus_operators": kraus_info,
                "estimated_parameters": estimated_params,
                "true_parameters": channel_params
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "success": False}), 500


@app.route('/api/tomography/multiple', methods=['POST'])
def run_multiple_tomographies():
    """
    Запустить N прогонов томографии для статистического анализа
    """
    try:
        data = request.json

        n_qubits = data.get('n_qubits', 1)
        channel_type = data.get('channel_type', 'depolarizing')
        channel_params = data.get('channel_params', {})
        shots = data.get('shots', 1000)
        n_runs = data.get('n_runs', 10)

        # Создаём канал
        channel = create_channel(channel_type, channel_params, n_qubits)

        # Множественные прогоны
        qpt = QuantumProcessTomography(n_qubits=n_qubits, shots=shots)
        results = qpt.run_multiple_tomographies(channel, n_runs=n_runs)

        # Статистика
        from noiselab.metrics.validation import statistical_analysis_multiple_runs
        stats = statistical_analysis_multiple_runs(results)

        response = {
            "success": True,
            "statistics": {
                "n_runs": stats['n_runs'],
                "fidelity_mean": float(stats['fidelity']['mean']),
                "fidelity_std": float(stats['fidelity']['std']),
                "fidelity_min": float(stats['fidelity']['min']),
                "fidelity_max": float(stats['fidelity']['max']),
                "kraus_rank_mean": float(stats['kraus_rank']['mean'])
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


def create_channel(channel_type, params, n_qubits):
    """Создать канал по типу и параметрам"""
    if channel_type == 'depolarizing':
        p = params.get('p', 0.1)
        return DepolarizingChannel(p, n_qubits=n_qubits)

    elif channel_type == 'amplitude_damping' and n_qubits == 1:
        gamma = params.get('gamma', 0.3)
        return AmplitudeDampingChannel(gamma)

    elif channel_type == 'phase_damping' and n_qubits == 1:
        lambda_ = params.get('lambda', 0.2)
        return PhaseDampingChannel(lambda_)

    elif channel_type == 'random':
        return random_cptp_channel(n_qubits, seed=params.get('seed'))

    elif channel_type == 'two_qubit_depolarizing' and n_qubits == 2:
        from noiselab.channels.two_qubit_noise import TwoQubitDepolarizing
        p = params.get('p', 0.1)
        return TwoQubitDepolarizing(p)

    return None


if __name__ == '__main__':
    print("=" * 70)
    print("NoiseLab++ API Server")
    print("=" * 70)
    print("Сервер запущен на http://localhost:5000")
    print("\nДоступные endpoints:")
    print("  GET  /api/health              - Проверка работоспособности")
    print("  GET  /api/channels/list       - Список доступных каналов")
    print("  POST /api/tomography/run      - Запуск томографии")
    print("  POST /api/tomography/multiple - Множественные прогоны")
    print("=" * 70)

    app.run(debug=True, host='0.0.0.0', port=5000)
