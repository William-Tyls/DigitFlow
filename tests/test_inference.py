import numpy as np
import pytest

from digit_recognition.inference import ModelLoadError, Predictor


def test_missing_and_corrupt_models_report_recovery(tmp_path):
    with pytest.raises(ModelLoadError, match="Run"):
        Predictor(tmp_path / "missing.pt")
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"this is not a checkpoint")
    with pytest.raises(ModelLoadError, match="Retrain"):
        Predictor(corrupt)


def test_blank_prediction_is_absent(predictor):
    assert predictor.predict(np.zeros((280, 280), dtype=np.uint8)) is None


def test_probabilities_are_normalized_and_features_are_input_dependent(predictor):
    first = np.zeros((280, 280), dtype=np.uint8)
    first[40:240, 130:150] = 255
    second = first.copy()
    second[40:60, 70:200] = 255
    a, b = predictor.predict(first), predictor.predict(second)
    assert a.probabilities.shape == (10,)
    assert np.isfinite(a.probabilities).all()
    assert (a.probabilities >= 0).all()
    assert float(a.probabilities.sum()) == pytest.approx(1, abs=1e-6)
    assert a.digit == int(np.argmax(a.probabilities))
    assert not np.array_equal(a.feature_maps[0], b.feature_maps[0])
    np.testing.assert_allclose(a.contributions,
                               predictor.weights * a.hidden_activations[None, :])
    # Evaluation is deterministic: animation never feeds back into the CNN.
    np.testing.assert_array_equal(a.probabilities, predictor.predict(first).probabilities)
