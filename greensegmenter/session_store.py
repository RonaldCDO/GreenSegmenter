import json
import hashlib
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from greensegmenter.batch import BatchSession
from greensegmenter.tiling import Tile


SESSIONS_ROOT = Path(__file__).resolve().parents[1] / "workspace" / "sessions"


def image_pixel_hash(image: np.ndarray) -> str:
    """Identifica de forma estável dimensões e pixels de uma imagem RGB."""
    rgb = np.ascontiguousarray(image.astype(np.uint8))
    digest = hashlib.sha256()
    digest.update(f"{rgb.shape[0]}x{rgb.shape[1]}x{rgb.shape[2]}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _ensure_source_hashes(root: Path, manifest: dict[str, Any]) -> bool:
    migrated = False
    for record in manifest["sources"]:
        if not record.get("pixel_sha256"):
            image = np.asarray(
                Image.open(root / record["file"]).convert("RGB"), dtype=np.uint8
            )
            record["pixel_sha256"] = image_pixel_hash(image)
            migrated = True
    return migrated


def list_sessions() -> list[str]:
    if not SESSIONS_ROOT.exists():
        return []
    return sorted(
        [path.name for path in SESSIONS_ROOT.iterdir() if (path / "session.json").is_file()],
        reverse=True,
    )


def create_session(
    tiles: list[Tile],
    collection: list[dict[str, Any]],
    selected_indices: list[int],
    settings: dict[str, Any] | None = None,
) -> BatchSession:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
    root = SESSIONS_ROOT / session_id
    (root / "sources").mkdir(parents=True, exist_ok=False)
    (root / "tiles").mkdir()
    (root / "masks").mkdir()

    source_records = []
    for index, item in enumerate(collection):
        name = f"source_{index:04d}.png"
        Image.fromarray(item["image"].astype(np.uint8), mode="RGB").save(root / "sources" / name)
        height, width = item["image"].shape[:2]
        source_records.append({
            "name": item["name"], "file": f"sources/{name}",
            "width": int(width), "height": int(height),
            "pixel_sha256": image_pixel_hash(item["image"]),
        })

    tile_records = []
    for index, tile in enumerate(tiles):
        name = f"tile_{index:06d}.png"
        Image.fromarray(tile.image.astype(np.uint8), mode="RGB").save(root / "tiles" / name)
        tile_records.append({
            "file": f"tiles/{name}", "x": tile.x, "y": tile.y,
            "valid_width": tile.valid_width, "valid_height": tile.valid_height,
            "source_index": tile.source_index, "source_name": tile.source_name,
        })
    session = BatchSession(list(selected_indices), session_id=session_id)
    manifest = {
        "version": 1,
        "session_id": session_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "position": 0,
        "selected_indices": list(selected_indices),
        "saved_indices": [],
        "model_generated_indices": [],
        "background_indices": [],
        "review_later_indices": [],
        "sources": source_records,
        "tiles": tile_records,
        "settings": settings or {},
        "status": "in_progress",
    }
    _atomic_json(root / "session.json", manifest)
    return session


def persist_progress(
    session: BatchSession,
    tiles: list[Tile],
    settings: dict[str, Any] | None = None,
    background_index: int | None = None,
    review_index: int | None = None,
    needs_review: bool | None = None,
) -> None:
    if not session.session_id:
        return
    root = SESSIONS_ROOT / session.session_id
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    for index, mask in session.saved_masks.items():
        target = root / "masks" / f"tile_{index:06d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(target)
    backgrounds = set(manifest.get("background_indices", []))
    if background_index is not None:
        backgrounds.add(background_index)
    reviews = set(session.review_later_indices)
    if review_index is not None and needs_review is not None:
        if needs_review:
            reviews.add(review_index)
            session.review_later_indices.add(review_index)
        else:
            reviews.discard(review_index)
            session.review_later_indices.discard(review_index)
    manifest.update({
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "position": session.position,
        "saved_indices": sorted(session.saved_masks),
        "model_generated_indices": sorted(
            session.model_generated_indices - set(session.saved_masks)
        ),
        "background_indices": sorted(backgrounds),
        "review_later_indices": sorted(reviews),
        "settings": {**manifest.get("settings", {}), **(settings or {})},
        "status": (
            "complete"
            if len(session.saved_masks) == len(session.selected_indices)
            else "in_progress"
        ),
    })
    _atomic_json(root / "session.json", manifest)


def append_samples(
    session: BatchSession,
    tiles: list[Tile],
    collection: list[dict[str, Any]],
    new_tiles: list[Tile],
    new_collection: list[dict[str, Any]],
    selected_indices: list[int],
    settings: dict[str, Any] | None = None,
) -> tuple[list[Tile], list[dict[str, Any]], list[int]]:
    """Anexa novas fontes e tiles sem alterar os índices existentes da sessão."""
    if not session.session_id:
        raise ValueError("A sessão precisa estar salva antes de receber amostras.")
    if not selected_indices:
        raise ValueError("Selecione ao menos um novo tile para adicionar à sessão.")
    root = SESSIONS_ROOT / session.session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_offset = len(manifest["sources"])
    tile_offset = len(manifest["tiles"])

    source_records = list(manifest["sources"])
    for local_index, item in enumerate(new_collection):
        source_index = source_offset + local_index
        name = f"source_{source_index:04d}.png"
        image = item["image"].astype(np.uint8)
        Image.fromarray(image, mode="RGB").save(root / "sources" / name)
        height, width = image.shape[:2]
        source_records.append({
            "name": item["name"], "file": f"sources/{name}",
            "width": int(width), "height": int(height),
            "pixel_sha256": image_pixel_hash(image),
        })

    adjusted_tiles: list[Tile] = []
    tile_records = list(manifest["tiles"])
    for local_index, tile in enumerate(new_tiles):
        tile_index = tile_offset + local_index
        name = f"tile_{tile_index:06d}.png"
        adjusted = Tile(
            tile.image, tile.x, tile.y, tile.valid_width, tile.valid_height,
            source_offset + tile.source_index, tile.source_name,
        )
        Image.fromarray(adjusted.image.astype(np.uint8), mode="RGB").save(
            root / "tiles" / name
        )
        adjusted_tiles.append(adjusted)
        tile_records.append({
            "file": f"tiles/{name}", "x": adjusted.x, "y": adjusted.y,
            "valid_width": adjusted.valid_width,
            "valid_height": adjusted.valid_height,
            "source_index": adjusted.source_index,
            "source_name": adjusted.source_name,
        })

    was_complete = set(manifest["selected_indices"]) <= set(session.saved_masks)
    appended_indices = [tile_offset + index for index in selected_indices]
    session.selected_indices.extend(appended_indices)
    if was_complete:
        session.position = session.selected_indices.index(appended_indices[0])
    manifest.update({
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "position": session.position,
        "selected_indices": session.selected_indices,
        "sources": source_records,
        "tiles": tile_records,
        "settings": {**manifest.get("settings", {}), **(settings or {})},
        "status": "in_progress",
    })
    additions = list(manifest.get("additions", []))
    additions.append({
        "added_at": datetime.now().isoformat(timespec="seconds"),
        "source_count": len(new_collection),
        "tile_count": len(new_tiles),
        "selected_count": len(appended_indices),
    })
    manifest["additions"] = additions
    _atomic_json(manifest_path, manifest)

    merged_collection = [*collection, *new_collection]
    return [*tiles, *adjusted_tiles], merged_collection, appended_indices


def load_session(session_id: str) -> tuple[BatchSession, list[Tile], list[dict[str, Any]], dict[str, Any]]:
    root = SESSIONS_ROOT / session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _ensure_source_hashes(root, manifest):
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(manifest_path, manifest)
    saved = set(manifest.get("saved_indices", []))
    generated = set(manifest.get("model_generated_indices", [])) - saved
    selected = list(manifest["selected_indices"])
    pending_positions = [pos for pos, index in enumerate(selected) if index not in saved]
    position = pending_positions[0] if pending_positions else max(0, len(selected) - 1)
    current_index = selected[position] if pending_positions else None
    current_source = (
        manifest["tiles"][current_index]["source_index"]
        if current_index is not None else None
    )
    collection = []
    for source_index, record in enumerate(manifest["sources"]):
        if source_index == current_source:
            image = np.asarray(
                Image.open(root / record["file"]).convert("RGB"), dtype=np.uint8
            ).copy()
        else:
            image = np.empty((0, 0, 3), dtype=np.uint8)
        collection.append({
            "name": record["name"], "image": image,
            "shape": (record["height"], record["width"], 3),
        })
    tiles = []
    for index, record in enumerate(manifest["tiles"]):
        if index != current_index:
            image = np.empty((0, 0, 3), dtype=np.uint8)
        else:
            image = np.asarray(Image.open(root / record["file"]).convert("RGB"), dtype=np.uint8).copy()
        tiles.append(Tile(
            image, record["x"], record["y"], record["valid_width"],
            record["valid_height"], record["source_index"], record["source_name"],
        ))
    masks = {
        index: np.asarray(Image.open(root / "masks" / f"tile_{index:06d}.png"), dtype=np.uint8) > 0
        for index in saved
    }
    generated_masks = {}
    if current_index in generated:
        generated_masks[current_index] = np.asarray(
            Image.open(root / "masks" / f"tile_{current_index:06d}.png"),
            dtype=np.uint8,
        ) > 0
    session = BatchSession(
        selected, position, masks, session_id,
        set(manifest.get("review_later_indices", [])),
        generated_masks, generated,
    )
    return session, tiles, collection, manifest


def session_source_hashes(session_id: str) -> set[str]:
    """Obtém hashes das fontes e migra manifestos antigos quando necessário."""
    root = SESSIONS_ROOT / session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _ensure_source_hashes(root, manifest):
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _atomic_json(manifest_path, manifest)
    return {record["pixel_sha256"] for record in manifest["sources"]}


def session_source_summary(session_id: str) -> list[dict[str, Any]]:
    """Lista fontes ativas e excluídas, distinguindo originais de adições."""
    root = SESSIONS_ROOT / session_id
    manifest = json.loads((root / "session.json").read_text(encoding="utf-8"))
    added_count = sum(item.get("source_count", 0) for item in manifest.get("additions", []))
    original_count = manifest.get(
        "original_source_count", max(0, len(manifest["sources"]) - added_count)
    )
    excluded = set(manifest.get("excluded_source_indices", []))
    tile_counts: dict[int, int] = {}
    for record in manifest["tiles"]:
        source_index = int(record["source_index"])
        tile_counts[source_index] = tile_counts.get(source_index, 0) + 1
    return [
        {
            "index": index,
            "name": record["name"],
            "origin": "original" if index < original_count else "adicionada",
            "excluded": index in excluded,
            "tile_count": tile_counts.get(index, 0),
        }
        for index, record in enumerate(manifest["sources"])
    ]


def exclude_session_tiles(session_id: str, indices: list[int]) -> int:
    """Remove tiles da sessão ativa sem apagar seus artefatos do disco."""
    root = SESSIONS_ROOT / session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = {index for index in indices if 0 <= index < len(manifest["tiles"])}
    previously_excluded = set(manifest.get("excluded_indices", []))
    newly_excluded = valid - previously_excluded
    excluded = previously_excluded | valid
    selected = [index for index in manifest["selected_indices"] if index not in excluded]
    active_sources = {manifest["tiles"][index]["source_index"] for index in selected}
    excluded_sources = set(manifest.get("excluded_source_indices", []))
    excluded_sources.update(
        index for index in range(len(manifest["sources"])) if index not in active_sources
    )
    saved = set(manifest.get("saved_indices", []))
    pending_positions = [position for position, index in enumerate(selected) if index not in saved]
    manifest.update({
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "selected_indices": selected,
        "excluded_indices": sorted(excluded),
        "excluded_source_indices": sorted(excluded_sources),
        "position": pending_positions[0] if pending_positions else max(0, len(selected) - 1),
        "status": "complete" if set(selected) <= saved else "in_progress",
    })
    _atomic_json(manifest_path, manifest)
    return len(newly_excluded)


def exclude_session_sources(session_id: str, source_indices: list[int]) -> tuple[int, int]:
    """Exclui fontes e todos os seus tiles da sessão ativa."""
    root = SESSIONS_ROOT / session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_sources = {
        index for index in source_indices if 0 <= index < len(manifest["sources"])
    }
    previous = set(manifest.get("excluded_source_indices", []))
    manifest["excluded_source_indices"] = sorted(previous | valid_sources)
    _atomic_json(manifest_path, manifest)
    tile_indices = [
        index for index, record in enumerate(manifest["tiles"])
        if record["source_index"] in valid_sources
    ]
    excluded_tiles = exclude_session_tiles(session_id, tile_indices)
    return len(valid_sources - previous), excluded_tiles


def import_model_predictions(
    session_id: str,
    prediction_dir: Path,
    probability_dir: Path | None,
    total_active: int,
    reviewed_count: int,
    inference: dict[str, Any],
) -> list[int]:
    """Importa predições em lote mantendo revisões humanas como fonte de verdade."""
    root = SESSIONS_ROOT / session_id
    manifest_path = root / "session.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = list(manifest["selected_indices"])
    if not 0 <= reviewed_count <= total_active <= len(selected):
        raise ValueError("A faixa revisada/gerada não cabe na seleção da sessão.")
    reviewed = selected[:reviewed_count]
    if not set(reviewed) <= set(manifest.get("saved_indices", [])):
        raise ValueError("Todos os tiles da faixa revisada precisam ter máscaras salvas.")
    active = selected[:total_active]
    generated = active[reviewed_count:]
    for index in generated:
        filename = f"tile_{index:06d}.png"
        source = prediction_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"Predição ausente: {source}")
        with Image.open(source) as mask, Image.open(
            root / manifest["tiles"][index]["file"]
        ) as tile_image:
            expected = tile_image.size
            if mask.size != expected:
                raise ValueError(
                    f"Predição {filename} possui tamanho {mask.size}; esperado {expected}."
                )
        shutil.copy2(source, root / "masks" / filename)
        if probability_dir is not None:
            probability = probability_dir / filename
            if not probability.is_file():
                raise FileNotFoundError(f"Probabilidade ausente: {probability}")
            (root / "probabilities").mkdir(exist_ok=True)
            shutil.copy2(probability, root / "probabilities" / filename)

    active_set = set(active)
    excluded = set(range(len(manifest["tiles"]))) - active_set
    active_sources = {manifest["tiles"][index]["source_index"] for index in active}
    manifest.update({
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "position": reviewed_count if generated else max(0, len(active) - 1),
        "selected_indices": active,
        "model_generated_indices": generated,
        "excluded_indices": sorted(excluded),
        "excluded_source_indices": sorted(
            index for index in range(len(manifest["sources"]))
            if index not in active_sources
        ),
        "inference": inference,
        "status": "in_progress" if generated else "complete",
    })
    _atomic_json(manifest_path, manifest)
    return generated


def session_summary(session: BatchSession) -> dict[str, int]:
    saved = set(session.saved_masks)
    review = set(session.review_later_indices)
    generated = set(session.model_generated_indices) - saved
    selected = set(session.selected_indices)
    return {
        "total": len(selected),
        "novos": len(selected - saved - generated),
        "salvos": len(saved),
        "gerados_por_rede": len(generated),
        "revisão_posterior": len(review),
    }


def tile_file(session_id: str, tile_index: int) -> np.ndarray:
    path = SESSIONS_ROOT / session_id / "tiles" / f"tile_{tile_index:06d}.png"
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def mask_file(session_id: str, tile_index: int) -> np.ndarray:
    path = SESSIONS_ROOT / session_id / "masks" / f"tile_{tile_index:06d}.png"
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def probability_file(session_id: str, tile_index: int) -> np.ndarray | None:
    path = SESSIONS_ROOT / session_id / "probabilities" / f"tile_{tile_index:06d}.png"
    if not path.is_file():
        return None
    return np.asarray(Image.open(path), dtype=np.float32) / 65535.0


def source_file(session_id: str, source_index: int) -> np.ndarray:
    manifest = json.loads(
        (SESSIONS_ROOT / session_id / "session.json").read_text(encoding="utf-8")
    )
    path = SESSIONS_ROOT / session_id / manifest["sources"][source_index]["file"]
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
