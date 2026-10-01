from dataclasses import dataclass, field

import cv2
import numpy as np

from greensegmenter.domain import Segmenter
from greensegmenter.tiling import Tile


@dataclass
class BatchSession:
    selected_indices: list[int]
    position: int = 0
    saved_masks: dict[int, np.ndarray] = field(default_factory=dict)
    session_id: str | None = None
    review_later_indices: set[int] = field(default_factory=set)
    model_generated_masks: dict[int, np.ndarray] = field(default_factory=dict)
    model_generated_indices: set[int] = field(default_factory=set)

    @property
    def current_index(self) -> int:
        return self.selected_indices[self.position]


def move_session_position(session: BatchSession, offset: int) -> int:
    """Move a sessão na fila selecionada, sem ultrapassar suas extremidades."""
    if not session.selected_indices:
        raise ValueError("A sessão não possui tiles selecionados.")
    session.position = min(
        max(session.position + int(offset), 0), len(session.selected_indices) - 1
    )
    return session.current_index


def score_tiles(
    tiles: list[Tile],
    segmenter: Segmenter,
    exg_threshold: float,
    minimum_presence: float,
    device: str = "CPU",
) -> tuple[list[float], list[int]]:
    """Retorna presença ExG percentual e índices aprovados pelo filtro."""
    scores, selected = [], []
    for index, tile in enumerate(tiles):
        probability = segmenter.predict(tile.image, device=device).probability
        valid = probability[: tile.valid_height, : tile.valid_width]
        presence = float((valid >= exg_threshold).mean() * 100)
        scores.append(presence)
        if presence >= minimum_presence:
            selected.append(index)
    return scores, selected


def structural_similarity(first: Tile, second: Tile) -> float:
    """Calcula similaridade estrutural global (SSIM) em percentual.

    As áreas válidas são convertidas para cinza e reduzidas para tornar a
    comparação independente do tamanho e menos sensível a ruído de pixel.
    """

    def prepare(tile: Tile) -> np.ndarray:
        valid = tile.image[: tile.valid_height, : tile.valid_width]
        gray = cv2.cvtColor(valid, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float64)

    left, right = prepare(first), prepare(second)
    mean_left, mean_right = left.mean(), right.mean()
    variance_left, variance_right = left.var(), right.var()
    covariance = ((left - mean_left) * (right - mean_right)).mean()
    dynamic_range = 255.0
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    score = (
        (2 * mean_left * mean_right + c1) * (2 * covariance + c2)
        / ((mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2))
    )
    return float(np.clip(score, 0, 1) * 100)


def filter_similar_tiles(
    tiles: list[Tile], candidate_indices: list[int], threshold: float
) -> tuple[list[int], dict[int, tuple[int, float]]]:
    """Manté o primeiro tile e marca candidatos semelhantes aos já mantidos."""
    if not 0 <= threshold <= 100:
        raise ValueError("O limiar de similaridade deve estar entre 0% e 100%.")
    selected: list[int] = []
    similar: dict[int, tuple[int, float]] = {}
    for index in candidate_indices:
        best_index, best_score = -1, -1.0
        for retained_index in selected:
            score = structural_similarity(tiles[retained_index], tiles[index])
            if score > best_score:
                best_index, best_score = retained_index, score
        if best_score >= threshold:
            similar[index] = (best_index, best_score)
        else:
            selected.append(index)
    return selected, similar


def filter_new_similar_tiles(
    new_tiles: list[Tile],
    candidate_indices: list[int],
    reference_tiles: list[Tile],
    threshold: float,
) -> tuple[list[int], dict[int, tuple[str, int, float]]]:
    """Filtra novos candidatos contra a sessão e contra novos tiles mantidos."""
    if not 0 <= threshold <= 100:
        raise ValueError("O limiar de similaridade deve estar entre 0% e 100%.")
    selected: list[int] = []
    similar: dict[int, tuple[str, int, float]] = {}
    for index in candidate_indices:
        best_kind, best_index, best_score = "", -1, -1.0
        for reference_index, reference in enumerate(reference_tiles):
            score = structural_similarity(reference, new_tiles[index])
            if score > best_score:
                best_kind, best_index, best_score = "session", reference_index, score
        for retained_index in selected:
            score = structural_similarity(new_tiles[retained_index], new_tiles[index])
            if score > best_score:
                best_kind, best_index, best_score = "new", retained_index, score
        if best_score >= threshold:
            similar[index] = (best_kind, best_index, best_score)
        else:
            selected.append(index)
    return selected, similar


def tile_thumbnail(tile: Tile, status: str) -> np.ndarray:
    """Marca aprovados, semelhantes e rejeitados sem ocultá-los."""
    thumbnail = tile.image.copy()
    color = {
        "selected": (34, 197, 94),
        "similar": (245, 158, 11),
        "rejected": (120, 120, 120),
    }[status]
    cv2.rectangle(thumbnail, (1, 1), (thumbnail.shape[1] - 2, thumbnail.shape[0] - 2), color, 8)
    if status != "selected":
        overlay = np.full_like(thumbnail, (80, 80, 80))
        weight = 0.25 if status == "similar" else 0.45
        thumbnail = cv2.addWeighted(thumbnail, 1 - weight, overlay, weight, 0)
        cv2.rectangle(thumbnail, (1, 1), (thumbnail.shape[1] - 2, thumbnail.shape[0] - 2), color, 8)
    return thumbnail
