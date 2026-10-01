import json
from pathlib import Path

import numpy as np

from greensegmenter.annotation import (
    MASK_BRUSH_COLOR,
    clean_mask,
    editor_value,
    export_annotation,
    mask_from_editor,
    merge_mask_region,
    polygons_from_mask,
)
from greensegmenter.tiling import Tile


def make_tile(size: int = 64, x: int = 100, y: int = 200) -> Tile:
    return Tile(np.zeros((size, size, 3), dtype=np.uint8), x, y, size, size)


def test_editor_round_trip_preserves_binary_mask() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[5:20, 7:22] = True
    value = editor_value(np.zeros((32, 32, 3), dtype=np.uint8), mask)
    np.testing.assert_array_equal(mask_from_editor(value), mask)
    np.testing.assert_array_equal(
        np.asarray(value["composite"])[..., :3], np.asarray(value["background"])
    )
    assert tuple(np.asarray(value["layers"][0])[5, 7]) == (255, 0, 255, 180)
    assert MASK_BRUSH_COLOR == ("#ff00ff", 180 / 255)


def test_small_components_are_removed() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:4, 2:4] = True
    mask[10:30, 10:30] = True
    cleaned = clean_mask(mask, min_component_area=10)
    assert not cleaned[2:4, 2:4].any()
    assert cleaned[10:30, 10:30].all()


def test_only_holes_below_limit_are_filled() -> None:
    mask = np.ones((50, 50), dtype=bool)
    mask[10:13, 10:13] = False
    mask[25:35, 25:35] = False
    cleaned = clean_mask(mask, fill_holes=True, max_hole_area=20)
    assert cleaned[10:13, 10:13].all()
    assert not cleaned[25:35, 25:35].any()


def test_region_segmentation_adds_pixels_without_changing_outside() -> None:
    base = np.zeros((8, 9), dtype=bool)
    base[1, 1] = True
    regional = np.ones((3, 4), dtype=bool)

    merged = merge_mask_region(base, regional, (3, 2, 7, 5))

    assert merged[1, 1]
    assert merged[2:5, 3:7].all()
    assert merged.sum() == 13


def test_polygon_coordinates_are_translated_to_original_image() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:31, 20:41] = True
    polygons = polygons_from_mask(mask, make_tile(), simplification=0)
    points = np.asarray(polygons[0]).reshape(-1, 2)
    assert points[:, 0].min() == 120
    assert points[:, 1].min() == 210


def test_export_creates_png_and_coco_json() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:31, 20:41] = True
    png, coco, summary = export_annotation(
        mask, make_tile(), (500, 600, 3), "planta", 0
    )
    assert Path(png).is_file()
    payload = json.loads(Path(coco).read_text(encoding="utf-8"))
    assert payload["images"][0]["width"] == 600
    assert payload["categories"][0]["name"] == "planta"
    assert payload["annotations"][0]["segmentation"][0][0] >= 100
    assert summary["polígonos_exportados"] == 1
