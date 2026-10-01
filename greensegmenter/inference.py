import numpy as np

from greensegmenter.domain import Prediction


class ExGVegetationSegmenter:
    """Segmentador de vegetação baseado no índice de excesso de verde (ExG)."""

    def predict(self, image: np.ndarray, device: str = "CPU") -> Prediction:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("A inferência requer uma imagem RGB.")

        if device == "GPU (CuPy)":
            try:
                import cupy as cp
                if cp.cuda.runtime.getDeviceCount() < 1:
                    raise RuntimeError
                rgb_gpu = cp.asarray(image, dtype=cp.float32) / 255.0
                red, green, blue = rgb_gpu[..., 0], rgb_gpu[..., 1], rgb_gpu[..., 2]
                excess_green = 2.0 * green - red - blue
                probability = cp.asnumpy(1.0 / (1.0 + cp.exp(-8.0 * (excess_green - 0.08))))
            except (ImportError, RuntimeError):
                raise ValueError("GPU CuPy selecionada, mas CUDA/CuPy não está disponível.")
        else:
            rgb = image.astype(np.float32) / 255.0
            red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            excess_green = 2.0 * green - red - blue
            probability = 1.0 / (1.0 + np.exp(-8.0 * (excess_green - 0.08)))
        return Prediction(
            probability=probability.astype(np.float32),
            model_name="Segmentador ExG",
            model_version="exg-1",
            is_demo=False,
        )


def available_devices() -> list[str]:
    devices = ["CPU"]
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            devices.append("GPU (CuPy)")
    except (ImportError, RuntimeError):
        pass
    return devices
