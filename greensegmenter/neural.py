"""Inferência binária opcional com modelos TorchScript fornecidos pelo usuário."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NeuralPrediction:
    probability: np.ndarray
    mask: np.ndarray


@lru_cache(maxsize=1)
def _load_model(path: str, modified_ns: int):
    import torch

    return torch.jit.load(path, map_location="cpu").eval()


def predict_neural(
    image: np.ndarray,
    weights_path: str | None,
    device: str,
    threshold: float,
) -> NeuralPrediction:
    """Aceita RGB [0,1] NCHW e logits binários N×1×H×W ou N×2×H×W.

    O modelo deve incluir internamente qualquer normalização/redimensionamento
    exigido pelo seu treinamento. Dois canais significam fundo e vegetação.
    """
    if not weights_path or not Path(weights_path).is_file():
        raise ValueError("Carregue um modelo TorchScript binário antes de segmentar.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("A inferência neural requer uma imagem RGB.")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ValueError("Instale o suporte neural com: uv sync --extra neural") from exc
    try:
        path = Path(weights_path).resolve()
        model = _load_model(str(path), path.stat().st_mtime_ns)
        # CPU é o padrão portátil; GPU ExG é uma configuração independente.
        tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        tensor = tensor.to(dtype=torch.float32).unsqueeze(0) / 255.0
        with torch.inference_mode():
            logits = model(tensor)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
                raise ValueError("O modelo deve retornar logits N×C×H×W.")
            if logits.shape[0] != 1 or logits.shape[1] not in (1, 2):
                raise ValueError("O modelo deve retornar um ou dois canais binários.")
            logits = functional.interpolate(
                logits, image.shape[:2], mode="bilinear", align_corners=False
            )
            probability = (
                logits.sigmoid()[0, 0]
                if logits.shape[1] == 1 else logits.softmax(dim=1)[0, 1]
            ).cpu().numpy().astype(np.float32)
        if not np.isfinite(probability).all():
            raise ValueError("O modelo retornou probabilidades inválidas.")
        return NeuralPrediction(probability, probability >= float(threshold))
    except (RuntimeError, OSError) as exc:
        raise ValueError(f"Não foi possível executar o modelo TorchScript: {exc}") from exc
