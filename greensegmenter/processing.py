from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from greensegmenter.domain import PreprocessConfig


@dataclass(frozen=True)
class PreprocessedImage:
    original: np.ndarray
    processed: np.ndarray
    original_size: tuple[int, int]


def to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Normaliza uma entrada PIL/NumPy para RGB uint8 sem alterar o original."""
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = Image.fromarray(image).convert("RGB")
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        else:
            raise ValueError("A imagem deve ser grayscale, RGB ou RGBA.")
    if not isinstance(image, Image.Image):
        raise ValueError("Formato de imagem não suportado.")
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    return np.asarray(normalized, dtype=np.uint8).copy()


def preprocess_image(
    image: Image.Image | np.ndarray, config: PreprocessConfig
) -> PreprocessedImage:
    if not 128 <= config.max_dimension <= 4096:
        raise ValueError("A dimensão máxima deve estar entre 128 e 4096 pixels.")

    original = to_rgb_array(image)
    height, width = original.shape[:2]
    scale = min(1.0, config.max_dimension / max(height, width))
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    processed = np.asarray(
        Image.fromarray(original).resize(target, Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    return PreprocessedImage(original, processed, (width, height))


def restore_float_map(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    restored = Image.fromarray(values.astype(np.float32), mode="F").resize(
        size, Image.Resampling.BILINEAR
    )
    return np.clip(np.asarray(restored, dtype=np.float32), 0.0, 1.0)
