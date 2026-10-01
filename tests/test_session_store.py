import json

import numpy as np
from PIL import Image

from greensegmenter import session_store
from greensegmenter.tiling import Tile


def sample_data():
    source = np.full((6, 8, 3), 90, dtype=np.uint8)
    tile = Tile(source[:4, :4].copy(), 0, 0, 4, 4, 0, "campo.png")
    return [tile], [{"name": "campo.png", "image": source, "shape": source.shape}]


def test_create_persist_and_resume_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()

    session = session_store.create_session(
        tiles, collection, [0], {"limiar": 0.42}
    )
    session.saved_masks[0] = np.eye(4, dtype=bool)
    session_store.persist_progress(session, tiles, background_index=0)

    resumed, resumed_tiles, resumed_collection, manifest = session_store.load_session(
        session.session_id
    )
    assert resumed.session_id == session.session_id
    assert np.array_equal(resumed.saved_masks[0], np.eye(4, dtype=bool))
    assert resumed_tiles[0].image.size == 0
    assert resumed_collection[0]["shape"] == (6, 8, 3)
    assert manifest["status"] == "complete"
    assert manifest["background_indices"] == [0]
    assert manifest["settings"] == {"limiar": 0.42}


def test_manifest_is_valid_and_session_is_listed(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])

    manifest_path = tmp_path / "sessions" / session.session_id / "session.json"
    assert json.loads(manifest_path.read_text())["session_id"] == session.session_id
    assert session_store.list_sessions() == [session.session_id]
    assert not manifest_path.with_suffix(".json.tmp").exists()


def test_loading_legacy_session_adds_source_pixel_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])
    manifest_path = tmp_path / "sessions" / session.session_id / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sources"][0].pop("pixel_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _, _, _, migrated = session_store.load_session(session.session_id)

    expected = session_store.image_pixel_hash(collection[0]["image"])
    assert migrated["sources"][0]["pixel_sha256"] == expected
    assert session_store.session_source_hashes(session.session_id) == {expected}


def test_append_samples_preserves_existing_indices_and_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])
    session.saved_masks[0] = np.ones((4, 4), dtype=bool)
    session_store.persist_progress(session, tiles)
    new_source = np.full((6, 8, 3), 150, dtype=np.uint8)
    new_collection = [{"name": "campo_2.png", "image": new_source, "shape": new_source.shape}]
    new_tiles = [Tile(new_source[:4, :4], 0, 0, 4, 4, 0, "campo_2.png")]

    merged_tiles, merged_collection, added = session_store.append_samples(
        session, tiles, collection, new_tiles, new_collection, [0], {"similaridade": 90}
    )
    resumed, _, _, manifest = session_store.load_session(session.session_id)

    assert added == [1]
    assert len(merged_tiles) == len(merged_collection) == 2
    assert resumed.selected_indices == [0, 1]
    assert 0 in resumed.saved_masks
    assert manifest["tiles"][1]["source_index"] == 1
    assert manifest["sources"][1]["pixel_sha256"] == session_store.image_pixel_hash(new_source)
    assert manifest["status"] == "in_progress"
    assert manifest["additions"][0]["selected_count"] == 1


def test_excluding_source_removes_its_tiles_without_deleting_files(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])

    source_count, tile_count = session_store.exclude_session_sources(
        session.session_id, [0]
    )
    resumed, resumed_tiles, _, manifest = session_store.load_session(session.session_id)

    assert (source_count, tile_count) == (1, 1)
    assert resumed.selected_indices == []
    assert resumed_tiles[0].image.size == 0
    assert manifest["excluded_source_indices"] == [0]
    assert manifest["excluded_indices"] == [0]
    assert (tmp_path / "sessions" / session.session_id / "tiles/tile_000000.png").is_file()


def test_import_model_predictions_preserves_reviewed_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    second = Tile(tiles[0].image.copy(), 4, 0, 4, 4, 0, "campo.png")
    session = session_store.create_session([tiles[0], second], collection, [0, 1])
    reviewed = np.eye(4, dtype=bool)
    session.saved_masks[0] = reviewed
    session_store.persist_progress(session, [tiles[0], second])
    predictions = tmp_path / "predictions"
    probabilities = tmp_path / "probabilities"
    predictions.mkdir()
    probabilities.mkdir()
    Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255).save(
        predictions / "tile_000001.png"
    )
    Image.fromarray(np.ones((4, 4), dtype=np.uint16) * 32768).save(
        probabilities / "tile_000001.png"
    )

    generated = session_store.import_model_predictions(
        session.session_id, predictions, probabilities, 2, 1,
        {"model": "binary-test-model", "weights": "ema"},
    )
    resumed, _, _, manifest = session_store.load_session(session.session_id)

    assert generated == [1]
    assert np.array_equal(resumed.saved_masks[0], reviewed)
    assert np.all(resumed.model_generated_masks[1])
    assert manifest["saved_indices"] == [0]
    assert manifest["model_generated_indices"] == [1]
    assert manifest["inference"]["weights"] == "ema"
    loaded_probability = session_store.probability_file(session.session_id, 1)
    assert loaded_probability is not None
    assert np.allclose(loaded_probability, 32768 / 65535)


def test_review_later_flag_and_summary_are_restored(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])
    session.saved_masks[0] = np.ones((4, 4), dtype=bool)

    session_store.persist_progress(
        session, tiles, review_index=0, needs_review=True
    )
    resumed, _, _, _ = session_store.load_session(session.session_id)

    assert resumed.review_later_indices == {0}
    assert session_store.session_summary(resumed) == {
        "total": 1,
        "novos": 0,
        "salvos": 1,
        "gerados_por_rede": 0,
        "revisão_posterior": 1,
    }

    _, resumed_tiles, _, _ = session_store.load_session(session.session_id)
    session_store.persist_progress(
        resumed, resumed_tiles, review_index=0, needs_review=False
    )
    reopened, _, _, _ = session_store.load_session(session.session_id)
    assert reopened.review_later_indices == set()


def test_loads_model_generated_masks_without_marking_them_reviewed(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "SESSIONS_ROOT", tmp_path / "sessions")
    tiles, collection = sample_data()
    session = session_store.create_session(tiles, collection, [0])
    root = tmp_path / "sessions" / session.session_id
    Image.fromarray(np.eye(4, dtype=np.uint8) * 255, mode="L").save(
        root / "masks" / "tile_000000.png"
    )
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_generated_indices"] = [0]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    resumed, resumed_tiles, resumed_collection, _ = session_store.load_session(
        session.session_id
    )

    assert resumed.saved_masks == {}
    assert np.array_equal(resumed.model_generated_masks[0], np.eye(4, dtype=bool))
    assert resumed_tiles[0].image.size > 0
    assert resumed_collection[0]["image"].size > 0
    assert session_store.session_summary(resumed)["gerados_por_rede"] == 1

    resumed.saved_masks[0] = resumed.model_generated_masks.pop(0)
    session_store.persist_progress(resumed, resumed_tiles)
    reviewed, _, _, reviewed_manifest = session_store.load_session(session.session_id)
    assert 0 in reviewed.saved_masks
    assert reviewed.model_generated_masks == {}
    assert reviewed_manifest["model_generated_indices"] == []
