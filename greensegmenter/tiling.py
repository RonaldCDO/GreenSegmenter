from dataclasses import dataclass

import numpy as np
from PIL import Image

from greensegmenter.processing import to_rgb_array


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    x: int
    y: int
    valid_width: int
    valid_height: int
    source_index: int = 0
    source_name: str = "imagem"

    @property
    def caption(self) -> str:
        return (
            f"x={self.x}, y={self.y} · "
            f"área válida {self.valid_width}×{self.valid_height} · {self.source_name}"
        )


def sliding_window_tiles(
    image: Image.Image | np.ndarray,
    tile_size: int = 512,
    overlap_percent: int = 20,
    source_index: int = 0,
    source_name: str = "imagem",
) -> list[Tile]:
    """Divide uma imagem em tiles quadrados, cobrindo bordas com padding preto."""
    if not 128 <= tile_size <= 2048:
        raise ValueError("O tile deve ter entre 128 e 2048 pixels.")
    if not 0 <= overlap_percent <= 75:
        raise ValueError("A sobreposição deve estar entre 0% e 75%.")

    rgb = to_rgb_array(image)
    height, width = rgb.shape[:2]
    stride = max(1, round(tile_size * (1 - overlap_percent / 100)))

    def positions(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        result = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if result[-1] != last:
            result.append(last)
        return result

    tiles: list[Tile] = []
    for y in positions(height):
        for x in positions(width):
            crop = rgb[y : y + tile_size, x : x + tile_size]
            valid_height, valid_width = crop.shape[:2]
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:valid_height, :valid_width] = crop
            tiles.append(
                Tile(padded, x, y, valid_width, valid_height, source_index, source_name)
            )
    return tiles
