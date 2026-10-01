import numpy as np
from gradio.components.image_editor import EditorData

import app
from greensegmenter.annotation import mask_from_editor
from greensegmenter.batch import BatchSession
from greensegmenter.tiling import Tile


def test_exg_segmentation_can_be_opened_and_exported_for_annotation() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 16:, 1] = 255
    tile = Tile(image, 0, 0, 32, 32)
    preview, _, prediction = app.run_tile_analysis(
        tile, image, 0.5, 0.45, "CPU", "ExG — excesso de verde", None
    )
    review = app.start_review(
        tile, 0.5, "CPU", "ExG — excesso de verde", None, True,
        new_prediction=prediction,
    )
    editor_value = review[0].preprocess(EditorData(**review[0].value))
    mask = mask_from_editor(editor_value)
    assert np.array_equal(mask, preview > 0)
    assert mask[:, 16:].all()
    assert not mask[:, :16].any()
    exported = app.create_exports(editor_value, tile, image, "vegetacao", 0)
    assert exported[0].endswith(".png")
    assert exported[1].endswith(".json")


def test_saved_masks_are_loaded_without_neural_inference() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    tile = Tile(image, 0, 0, 8, 8)
    saved = np.eye(8, dtype=bool)
    session = BatchSession([0], saved_masks={0: saved})
    review = app.start_review(
        tile, 0.5, "CPU", "Rede neural — TorchScript", None, True, session
    )
    editor_value = review[0].preprocess(EditorData(**review[0].value))
    assert np.array_equal(mask_from_editor(editor_value), saved)
