import numpy as np
import pytest

from digit_recognition.preprocessing import prepare_digit


def test_blank_and_tiny_noise_are_not_digits():
    image = np.zeros((100, 100), dtype=np.uint8)
    assert prepare_digit(image) is None
    image[5, 5] = 255
    assert prepare_digit(image) is None


def test_translation_and_canvas_size_do_not_change_normalized_ink():
    ink = np.zeros((40, 30), dtype=np.uint8)
    ink[3:35, 5:10] = 255
    ink[30:35, 5:26] = 255
    first = np.zeros((100, 100), dtype=np.uint8)
    second = np.zeros((300, 300), dtype=np.uint8)
    first[0:40, 0:30] = ink
    second[213:253, 230:260] = ink
    np.testing.assert_array_equal(prepare_digit(first), prepare_digit(second))


@pytest.mark.parametrize("shape", [(1, 100), (100, 1), (20, 30)])
def test_edge_touching_ink_is_finite_and_centered(shape):
    image = np.full(shape, 255, dtype=np.uint8)
    result = prepare_digit(image)
    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    yy, xx = np.indices(result.shape)
    assert abs((result * xx).sum() / result.sum() - 13.5) <= 0.51
    assert abs((result * yy).sum() / result.sum() - 13.5) <= 0.51


def test_invalid_input_reports_clear_error():
    with pytest.raises(ValueError, match="uint8"):
        prepare_digit(np.zeros((28, 28), dtype=np.float32))
