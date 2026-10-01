import json
import zipfile
from pathlib import Path

import numpy as np

from greensegmenter.annotation import export_annotation_batch
from greensegmenter.batch import (
    BatchSession,
    filter_new_similar_tiles,
    filter_similar_tiles,
    move_session_position,
    score_tiles,
    structural_similarity,
)
from greensegmenter.domain import Prediction
from greensegmenter.tiling import Tile


class GreenChannelSegmenter:
    def predict(self, image: np.ndarray, device: str = "CPU") -> Prediction:
        return Prediction(image[..., 1].astype(np.float32) / 255, "Teste", "1", False)


def make_tile(green_fraction: float, x: int) -> Tile:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    green_pixels = round(image.shape[0] * green_fraction)
    image[:green_pixels, :, 1] = 255
    return Tile(image, x, 0, 20, 20)


def test_filter_selects_all_tiles_above_minimum_presence() -> None:
    tiles = [make_tile(0.1, 0), make_tile(0.5, 20), make_tile(0.9, 40)]
    scores, selected = score_tiles(tiles, GreenChannelSegmenter(), 0.5, 40, "CPU")
    assert scores == [10.0, 50.0, 90.0]
    assert selected == [1, 2]


def test_similarity_filter_keeps_first_identical_tile() -> None:
    tiles = [make_tile(0.5, 0), make_tile(0.5, 20), make_tile(0.9, 40)]
    selected, similar = filter_similar_tiles(tiles, [0, 1, 2], 90)
    assert selected == [0, 2]
    assert similar[1][0] == 0
    assert similar[1][1] == 100.0


def test_structural_similarity_distinguishes_different_tiles() -> None:
    assert structural_similarity(make_tile(0.1, 0), make_tile(0.9, 20)) < 90


def test_new_similarity_filter_compares_with_existing_session() -> None:
    reference = [make_tile(0.5, 0)]
    new_tiles = [make_tile(0.5, 20), make_tile(0.9, 40)]
    selected, similar = filter_new_similar_tiles(new_tiles, [0, 1], reference, 90)
    assert selected == [1]
    assert similar[0] == ("session", 0, 100.0)


def test_batch_session_tracks_current_tile_and_saved_masks() -> None:
    session = BatchSession([2, 5])
    assert session.current_index == 2
    session.saved_masks[2] = np.ones((2, 2), dtype=bool)
    session.position = 1
    assert session.current_index == 5


def test_session_navigation_respects_selected_order_and_boundaries() -> None:
    session = BatchSession([2, 5, 9])

    assert move_session_position(session, -1) == 2
    assert move_session_position(session, 1) == 5
    assert move_session_position(session, 1) == 9
    assert move_session_position(session, 1) == 9
    assert move_session_position(session, -1) == 5


def test_batch_export_contains_every_saved_mask_and_global_coco() -> None:
    tiles = [make_tile(1, 0), make_tile(1, 20)]
    masks = {
        0: np.pad(np.ones((10, 10), dtype=bool), ((0, 10), (0, 10))),
        1: np.pad(np.ones((10, 10), dtype=bool), ((0, 10), (0, 10))),
    }
    zip_path, coco_path, summary = export_annotation_batch(
        masks, tiles, (20, 40, 3), "vegetacao", 0
    )
    with zipfile.ZipFile(zip_path) as archive:
        assert len(archive.namelist()) == 2
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    assert len(coco["annotations"]) == 2
    assert max(coco["annotations"][1]["segmentation"][0]) >= 20
    assert summary["tiles_revisados"] == 2


def test_batch_export_preserves_multiple_source_images() -> None:
    tiles = [make_tile(1, 0), make_tile(1, 0)]
    tiles[0] = Tile(tiles[0].image, 0, 0, 20, 20, 0, "a.png")
    tiles[1] = Tile(tiles[1].image, 0, 0, 20, 20, 1, "b.png")
    sources = [
        {"name": "a.png", "image": np.zeros((20, 20, 3), dtype=np.uint8)},
        {"name": "b.png", "image": np.zeros((20, 20, 3), dtype=np.uint8)},
    ]
    masks = {0: np.ones((20, 20), dtype=bool), 1: np.ones((20, 20), dtype=bool)}
    _, coco_path, _ = export_annotation_batch(masks, tiles, sources, "vegetacao", 0)
    coco = json.loads(Path(coco_path).read_text(encoding="utf-8"))
    assert [image["file_name"] for image in coco["images"]] == ["a.png", "b.png"]
    assert {item["image_id"] for item in coco["annotations"]} == {1, 2}
