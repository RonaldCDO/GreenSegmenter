import numpy as np
import pytest

from greensegmenter.tiling import sliding_window_tiles


def test_sliding_window_covers_image_and_anchors_last_tile_to_border() -> None:
    image = np.zeros((900, 1100, 3), dtype=np.uint8)
    tiles = sliding_window_tiles(image, tile_size=512, overlap_percent=0)
    coordinates = {(tile.x, tile.y) for tile in tiles}
    assert coordinates == {
        (0, 0), (512, 0), (588, 0),
        (0, 388), (512, 388), (588, 388),
    }
    assert all(tile.image.shape == (512, 512, 3) for tile in tiles)


def test_small_image_is_padded_to_square_tile() -> None:
    image = np.full((100, 200, 3), 77, dtype=np.uint8)
    tile = sliding_window_tiles(image, tile_size=256)[0]
    assert tile.valid_width == 200
    assert tile.valid_height == 100
    assert np.all(tile.image[:100, :200] == 77)
    assert np.all(tile.image[100:] == 0)


@pytest.mark.parametrize("overlap", [-1, 76])
def test_invalid_overlap_is_rejected(overlap: int) -> None:
    with pytest.raises(ValueError, match="sobreposição"):
        sliding_window_tiles(np.zeros((10, 10, 3)), 512, overlap)
