from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    max_dimension: int = 1024


@dataclass(frozen=True)
class Prediction:
    probability: np.ndarray
    model_name: str
    model_version: str
    is_demo: bool


class Segmenter(Protocol):
    """Contrato mínimo para qualquer modelo de segmentação binária."""

    def predict(self, image: np.ndarray, device: str = "CPU") -> Prediction:
        """Retorna probabilidade por pixel no intervalo [0, 1]."""
        ...
