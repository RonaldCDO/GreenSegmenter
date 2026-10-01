import numpy as np
import pytest
from PIL import Image

from greensegmenter.domain import PreprocessConfig
from greensegmenter.processing import preprocess_image, to_rgb_array


def test_grayscale_is_converted_to_rgb() -> None:
    image = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    result = to_rgb_array(image)
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[..., 0], image)


def test_large_image_is_resized_without_changing_aspect_ratio() -> None:
    image = Image.new("RGB", (200, 100), "green")
    result = preprocess_image(image, PreprocessConfig(max_dimension=128))
    assert result.original.shape == (100, 200, 3)
    assert result.processed.shape == (64, 128, 3)


def test_small_image_is_not_upscaled() -> None:
    image = Image.new("RGB", (64, 32), "green")
    result = preprocess_image(image, PreprocessConfig(max_dimension=128))
    assert result.processed.shape == (32, 64, 3)


def test_invalid_max_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="entre 128 e 4096"):
        preprocess_image(Image.new("RGB", (10, 10)), PreprocessConfig(64))

