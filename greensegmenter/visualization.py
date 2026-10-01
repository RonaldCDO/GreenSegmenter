import numpy as np


VEGETATION_COLOR = np.array([38, 208, 124], dtype=np.float32)
MASK_OVERLAY_COLOR = np.array([255, 0, 255], dtype=np.float32)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask.astype(bool)] = VEGETATION_COLOR.astype(np.uint8)
    return result


def overlay_mask(image: np.ndarray, mask: np.ndarray, opacity: float) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError("A máscara não está alinhada com a imagem.")
    alpha = float(np.clip(opacity, 0.0, 1.0))
    result = image.astype(np.float32).copy()
    selected = mask.astype(bool)
    result[selected] = (
        (1.0 - alpha) * result[selected] + alpha * MASK_OVERLAY_COLOR
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def colorize_confidence(probability: np.ndarray) -> np.ndarray:
    """Mapa azul-amarelo simples; intensidade representa probabilidade estimada."""
    p = np.clip(probability, 0.0, 1.0)
    red = (255 * p).astype(np.uint8)
    green = (210 * p).astype(np.uint8)
    blue = (255 * (1.0 - p)).astype(np.uint8)
    return np.stack([red, green, blue], axis=-1)
