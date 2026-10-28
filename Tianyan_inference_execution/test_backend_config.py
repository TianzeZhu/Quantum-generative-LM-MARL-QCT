from types import SimpleNamespace

import pytest

from backend_config import install_sparse_readout_fallback


class FakeBackend:
    def __init__(self, results, calibration_error=None):
        self.results = results
        self.calibration_error = calibration_error
        self.calibration_requests = []

    def query_experiment(self, query_id, readout_calibration=False):
        self.calibration_requests.append(readout_calibration)
        if readout_calibration and self.calibration_error is not None:
            raise self.calibration_error
        return self.results


def _device_for(backend):
    return SimpleNamespace(
        circuit_executor=SimpleNamespace(cqlib_backend=backend),
    )


def test_wide_hardware_readout_uses_sparse_shot_probabilities():
    backend = FakeBackend(
        [
            {
                "resultStatus": [
                    [0, 1],
                    [0, 0],
                    [1, 0],
                    [1, 0],
                ]
            }
        ]
    )
    install_sparse_readout_fallback(_device_for(backend), wire_count=90)

    with pytest.warns(RuntimeWarning, match="sparse uncalibrated"):
        results = backend.query_experiment("job-id", readout_calibration=True)

    assert backend.calibration_requests == [False]
    assert results[0]["probability"] == {
        "00": pytest.approx(1 / 3),
        "01": pytest.approx(2 / 3),
    }


def test_small_hardware_readout_keeps_cqlib_calibration():
    expected = [{"probability": {"0": 1.0}}]
    backend = FakeBackend(expected)
    install_sparse_readout_fallback(_device_for(backend), wire_count=5)

    assert backend.query_experiment("job-id", readout_calibration=True) is expected
    assert backend.calibration_requests == [True]


def test_dense_calibration_overflow_falls_back_without_resubmitting():
    backend = FakeBackend(
        [{"resultStatus": [[0], [0], [1]]}],
        calibration_error=OverflowError("dense allocation overflow"),
    )
    install_sparse_readout_fallback(_device_for(backend), wire_count=5)

    with pytest.warns(RuntimeWarning, match="sparse uncalibrated"):
        results = backend.query_experiment("job-id", readout_calibration=True)

    assert backend.calibration_requests == [True, False]
    assert results[0]["probability"] == {
        "0": pytest.approx(0.5),
        "1": pytest.approx(0.5),
    }
