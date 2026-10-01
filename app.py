from typing import Any
from pathlib import Path
import io
import json
import tempfile
import zipfile

import gradio as gr
import cv2
import numpy as np
from PIL import Image

from greensegmenter.annotation import (
    MASK_BRUSH_COLOR,
    clean_mask,
    editor_value,
    export_annotation,
    export_annotation_batch,
    mask_from_editor,
    mask_statistics,
    merge_mask_region,
)
from greensegmenter.batch import (
    BatchSession,
    filter_new_similar_tiles,
    filter_similar_tiles,
    move_session_position,
    score_tiles,
    tile_thumbnail,
)
from greensegmenter.inference import ExGVegetationSegmenter, available_devices
from greensegmenter.visualization import overlay_mask
from greensegmenter.neural import predict_neural

from greensegmenter.processing import to_rgb_array
from greensegmenter.session_store import (
    append_samples,
    create_session,
    image_pixel_hash,
    list_sessions,
    load_session,
    mask_file,
    persist_progress,
    exclude_session_sources,
    exclude_session_tiles,
    source_file,
    session_summary,
    session_source_summary,
    session_source_hashes,
    tile_file,
)
from greensegmenter.tiling import Tile, sliding_window_tiles


SEGMENTER = ExGVegetationSegmenter()
DEVICES = available_devices()
DEFAULT_DEVICE = "CPU"


def accept_upload(image: Image.Image | None):
    if image is None:
        raise gr.Error("Envie uma imagem agrícola para continuar.")
    rgb = to_rgb_array(image)
    height, width = rgb.shape[:2]
    metadata = {
        "largura": width,
        "altura": height,
        "canais": 3,
        "pixels": f"{width * height:,}".replace(",", "."),
    }
    return rgb, metadata, gr.Tabs(selected="preprocess")


def load_image_collection(
    files: list[str] | None, directory_files: list[str] | None
):
    paths = list(dict.fromkeys([*(files or []), *(directory_files or [])]))
    if not paths:
        raise gr.Error("Selecione uma ou mais imagens ou uma pasta.")
    collection = []
    for path in paths:
        try:
            with Image.open(path) as image:
                array = to_rgb_array(image)
        except Exception as exc:
            raise gr.Error(f"Não foi possível abrir {Path(path).name}: {exc}") from exc
        collection.append({"name": Path(path).name, "image": array, "shape": array.shape})
    names = [item["name"] for item in collection]
    gallery = [(item["image"], item["name"]) for item in collection]
    summary = {"imagens": len(collection), "nomes": names}
    return (
        collection,
        collection[0]["image"],
        gallery,
        gr.Dropdown(choices=names, value=names, multiselect=True),
        summary,
        gr.Tabs(selected="preprocess"),
        gr.Accordion(open=True),
    )


def inspect_collection_image(
    collection: list[dict[str, Any]] | None, evt: gr.SelectData
):
    if not collection:
        raise gr.Error("Carregue uma coleção de imagens primeiro.")
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    item = collection[int(index)]
    height, width = item["image"].shape[:2]
    return (
        int(index),
        item["image"],
        f"**{item['name']} selecionada** · {width}×{height}px",
    )


def use_collection_image_directly(
    collection: list[dict[str, Any]] | None, selected_index: int | None
):
    if not collection:
        raise gr.Error("Carregue uma ou mais imagens primeiro.")
    index = 0 if selected_index is None else int(selected_index)
    item = collection[index]
    image = item["image"]
    height, width = image.shape[:2]
    size = max(height, width)
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:height, :width] = image
    tile = Tile(padded, 0, 0, width, height, index, item["name"])
    tiles = [tile]
    session = create_session(tiles, collection, [0], {"entrada": "imagem direta"})
    return (
        tiles,
        [0],
        session,
        tile,
        tile.image,
        image,
        f"Imagem {item['name']} carregada diretamente, sem pré-processamento.",
        gr.Tabs(selected="segment"),
        gr.Accordion(open=True),
        False,
    )


def generate_tiles(
    collection: list[dict[str, Any]] | None,
    selected_names: list[str] | None,
    tile_size: int,
    overlap: int,
    exg_threshold: float,
    minimum_presence: float,
    filter_similar: bool,
    similarity_threshold: float,
    device: str,
):
    if not collection:
        raise gr.Error("Carregue imagens antes de gerar os tiles.")
    requested = set(selected_names or [])
    chosen = [item for item in collection if item["name"] in requested]
    if not chosen:
        raise gr.Error("Selecione ao menos uma imagem para o pré-processamento.")
    tiles: list[Tile] = []
    for source_index, item in enumerate(collection):
        if item["name"] not in requested:
            continue
        tiles.extend(
            sliding_window_tiles(
                item["image"], int(tile_size), int(overlap), source_index, item["name"]
            )
        )
    try:
        scores, selected = score_tiles(
            tiles, SEGMENTER, float(exg_threshold), float(minimum_presence), device
        )
        similar = {}
        if filter_similar:
            selected, similar = filter_similar_tiles(
                tiles, selected, float(similarity_threshold)
            )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    gallery = build_tile_gallery(tiles, scores, selected, similar)
    stride = round(int(tile_size) * (1 - int(overlap) / 100))
    status = (
        f"**{len(selected)} de {len(tiles)} tiles aprovados em {len(chosen)} imagem(ns)** · "
        f"{tile_size}×{tile_size}px · "
        f"stride {stride}px · sobreposição {overlap}%  \n"
        f"Filtro: presença ExG ≥ {minimum_presence}% com limiar {exg_threshold}. "
        + (
            f"{len(similar)} semelhante(s) removido(s) com limiar de "
            f"{float(similarity_threshold):g}%. "
            if filter_similar
            else "Filtro de semelhantes desativado. "
        )
        + "Bordas: verde = aprovado, amarela = semelhante e cinza = rejeitado por ExG."
    )
    return (
        gallery, tiles, scores, selected, similar, None, None, status,
        gr.Accordion(open=True),
    )


def generate_additional_samples(
    files: list[str] | None,
    directory_files: list[str] | None,
    session: BatchSession | None,
    existing_tiles: list[Tile] | None,
    tile_size: int,
    overlap: int,
    exg_threshold: float,
    minimum_presence: float,
    ignore_processed_images: bool,
    filter_similar: bool,
    similarity_threshold: float,
    device: str,
):
    if session is None or not session.session_id or existing_tiles is None:
        raise gr.Error("Retome uma sessão salva antes de adicionar amostras.")
    paths = list(dict.fromkeys([*(files or []), *(directory_files or [])]))
    if not paths:
        raise gr.Error("Selecione ao menos uma nova imagem.")
    new_collection: list[dict[str, Any]] = []
    ignored_names: list[str] = []
    known_hashes = (
        session_source_hashes(session.session_id) if ignore_processed_images else set()
    )
    batch_hashes: set[str] = set()
    for path in paths:
        try:
            with Image.open(path) as image:
                array = to_rgb_array(image)
        except Exception as exc:
            raise gr.Error(f"Não foi possível abrir {Path(path).name}: {exc}") from exc
        pixel_hash = image_pixel_hash(array)
        if ignore_processed_images and (
            pixel_hash in known_hashes or pixel_hash in batch_hashes
        ):
            ignored_names.append(Path(path).name)
            continue
        batch_hashes.add(pixel_hash)
        new_collection.append({
            "name": Path(path).name, "image": array, "shape": array.shape,
        })
    if not new_collection:
        raise gr.Error(
            "Todas as imagens selecionadas já foram processadas nesta sessão."
            if ignored_names else "Nenhuma imagem válida foi selecionada."
        )
    new_tiles: list[Tile] = []
    for source_index, item in enumerate(new_collection):
        new_tiles.extend(sliding_window_tiles(
            item["image"], int(tile_size), int(overlap), source_index, item["name"]
        ))
    scores, selected = score_tiles(
        new_tiles, SEGMENTER, float(exg_threshold), float(minimum_presence), device
    )
    similar: dict[int, tuple[str, int, float]] = {}
    if filter_similar:
        reference_tiles = []
        reference_indices = []
        for index in session.selected_indices:
            tile = existing_tiles[index]
            image = tile.image if tile.image.size else tile_file(session.session_id, index)
            valid = image[: tile.valid_height, : tile.valid_width]
            compact = cv2.resize(valid, (64, 64), interpolation=cv2.INTER_AREA)
            reference_tiles.append(Tile(
                compact, tile.x, tile.y, 64, 64,
                tile.source_index, tile.source_name,
            ))
            reference_indices.append(index)
        selected, similar = filter_new_similar_tiles(
            new_tiles, selected, reference_tiles, float(similarity_threshold)
        )
    selected_set = set(selected)
    gallery = []
    for index, tile in enumerate(new_tiles):
        if index in selected_set:
            tile_status, label = "selected", "✓ APROVADO"
        elif index in similar:
            kind, reference, similarity = similar[index]
            reference_label = (
                f"tile {reference_indices[reference] + 1} da sessão"
                if kind == "session" else f"novo tile {reference + 1}"
            )
            tile_status = "similar"
            label = f"≈ SEMELHANTE ao {reference_label} ({similarity:.2f}%)"
        else:
            tile_status, label = "rejected", "– rejeitado por ExG"
        gallery.append((
            tile_thumbnail(tile, tile_status),
            f"{label} · Novo tile {index + 1} · ExG {scores[index]:.2f}% · {tile.caption}",
        ))
    similar_message = (
        "Nenhum tile semelhante foi retirado da seleção automática."
        if not similar
        else f"{len(similar)} tile(s) semelhante(s) retirado(s) da seleção automática."
    )
    status = (
        f"**{len(selected)} de {len(new_tiles)} novos tiles prontos para adicionar.** "
        f"{similar_message} "
        f"{len(ignored_names)} imagem(ns) já processada(s) ignorada(s)."
    )
    return new_collection, new_tiles, selected, gallery, status


def confirm_additional_samples(
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
    new_tiles: list[Tile] | None,
    new_collection: list[dict[str, Any]] | None,
    selected: list[int] | None,
    similarity_threshold: float,
):
    if session is None or tiles is None or collection is None:
        raise gr.Error("Retome uma sessão salva antes de confirmar as amostras.")
    if not new_tiles or not new_collection or not selected:
        raise gr.Error("Gere e aprove ao menos um novo tile antes de confirmar.")
    try:
        _, _, appended = append_samples(
            session, tiles, collection, new_tiles, new_collection, list(selected),
            {"limiar_similaridade_adicao": float(similarity_threshold)},
        )
        resumed = resume_saved_session(session.session_id)
    except (OSError, ValueError, KeyError) as exc:
        raise gr.Error(f"Não foi possível adicionar as amostras: {exc}") from exc
    return (*resumed, [], f"**{len(appended)} novo(s) tile(s) adicionado(s) à sessão.**")


def build_tile_gallery(
    tiles: list[Tile],
    scores: list[float],
    selected: list[int],
    similar: dict[int, tuple[int, float]] | None = None,
) -> list[tuple[np.ndarray, str]]:
    selected_set = set(selected)
    similar = similar or {}
    def status(index: int) -> str:
        if index in selected_set:
            return "selected"
        if index in similar:
            return "similar"
        return "rejected"

    def caption(index: int) -> str:
        if index in selected_set:
            return "✓ APROVADO"
        if index in similar:
            reference, similarity = similar[index]
            return f"≈ SEMELHANTE ao tile {reference + 1} ({similarity:.2f}%)"
        return "– rejeitado por ExG"

    return [
        (
            tile_thumbnail(tile, status(index)),
            f"{caption(index)} · "
            f"Tile {index + 1} · ExG {scores[index]:.2f}% · {tile.caption}",
        )
        for index, tile in enumerate(tiles)
    ]


def inspect_tile(
    tiles: list[Tile] | None,
    scores: list[float] | None,
    selected: list[int] | None,
    similar: dict[int, tuple[int, float]] | None,
    evt: gr.SelectData,
):
    if not tiles or scores is None:
        raise gr.Error("Gere os tiles antes de selecionar.")
    index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    tile = tiles[int(index)]
    is_selected = int(index) in set(selected or [])
    duplicate = (similar or {}).get(int(index))
    selection_status = "selecionado para o lote" if is_selected else "fora do lote"
    if duplicate and not is_selected:
        selection_status = (
            f"semelhante ao tile {duplicate[0] + 1} ({duplicate[1]:.2f}%)"
        )
    label = (
        f"**Tile {int(index) + 1} · "
        f"{selection_status}.**  \n"
        f"ExG {scores[int(index)]:.2f}% · {tile.caption} · "
        "Use os botões + ou − para alterar a seleção."
    )
    return tile.image, tile, int(index), label


def change_tile_selection(
    tiles: list[Tile] | None,
    scores: list[float] | None,
    selected: list[int] | None,
    similar: dict[int, tuple[int, float]] | None,
    inspected_index: int | None,
    add: bool,
):
    if not tiles or scores is None or inspected_index is None:
        raise gr.Error("Clique em um tile para selecioná-lo na prévia.")
    approved = list(selected or [])
    if add and inspected_index not in approved:
        approved.append(inspected_index)
        approved.sort()
    if not add and inspected_index in approved:
        approved.remove(inspected_index)
    action = "adicionado ao lote" if add else "removido do lote"
    label = (
        f"**Tile {inspected_index + 1} {action}.**  \n"
        f"ExG {scores[inspected_index]:.2f}% · total selecionado: {len(approved)}"
    )
    status = (
        f"**{len(approved)} de {len(tiles)} tiles selecionados.** "
        "Clique para inspecionar; use + para adicionar ou − para remover."
    )
    return build_tile_gallery(tiles, scores, approved, similar), approved, label, status


def add_inspected_tile(tiles, scores, selected, similar, inspected_index):
    return change_tile_selection(tiles, scores, selected, similar, inspected_index, True)


def remove_inspected_tile(tiles, scores, selected, similar, inspected_index):
    return change_tile_selection(tiles, scores, selected, similar, inspected_index, False)


def open_segmentation(tile: Tile | None):
    if tile is None:
        raise gr.Error("Selecione um tile na galeria para continuar.")
    return gr.Tabs(selected="segment")


def start_batch(
    tiles: list[Tile] | None,
    selected: list[int] | None,
    collection: list[dict[str, Any]] | None,
):
    if not tiles or not selected:
        raise gr.Error("O filtro não aprovou nenhum tile. Ajuste os critérios.")
    session = create_session(
        tiles, collection or [], list(selected), {"entrada": "sliding window"}
    )
    tile = tiles[session.current_index]
    source = collection[tile.source_index]["image"] if collection else tile.image
    return (
        session,
        tile,
        tile.image,
        source,
        f"Tile 1 de {len(selected)} · índice original {session.current_index + 1}",
        gr.Tabs(selected="segment"),
        gr.Accordion(open=True),
    )


def run_tile_analysis(
    tile: Tile | None,
    source_image: np.ndarray | None,
    threshold: float,
    opacity: float,
    device: str,
    segmentation_method: str,
    model_weights: str | None,
):
    if tile is None:
        raise gr.Error("Selecione um tile antes de executar a segmentação.")
    try:
        prediction = (
            predict_neural(tile.image, model_weights, device, threshold)
            if segmentation_method == "Rede neural — TorchScript"
            else SEGMENTER.predict(tile.image, device=device)
        )
    except (ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    mask_array = prediction.probability >= float(threshold)
    return (
        mask_array.astype(np.uint8) * 255,
        f"Máscara gerada com {segmentation_method}. Abra a revisão para editar.",
        {
            "tile_key": tile_identity(tile),
            "mask": mask_array,
            "probability": prediction.probability,
            "method": segmentation_method,
        },
    )


def tile_identity(tile: Tile) -> tuple[object, ...]:
    return (
        tile.source_index, tile.source_name, tile.x, tile.y,
        tile.valid_width, tile.valid_height,
    )


def render_current_mask_overlay(
    tile: Tile | None,
    session: BatchSession | None,
    latest_prediction: dict[str, Any] | None,
    opacity: float,
) -> np.ndarray | None:
    """Sobrepõe a melhor máscara disponível ao tile atualmente carregado."""
    if tile is None or not tile.image.size:
        return None
    mask_array: np.ndarray | None = None
    if (
        latest_prediction is not None
        and tuple(latest_prediction.get("tile_key", ())) == tile_identity(tile)
    ):
        mask_array = np.asarray(latest_prediction.get("mask"), dtype=bool)
    if mask_array is None and session is not None:
        index = session.current_index
        mask_array = session.saved_masks.get(index)
        if mask_array is None:
            mask_array = session.model_generated_masks.get(index)
        if (
            mask_array is None
            and session.session_id
            and index in session.model_generated_indices
        ):
            mask_array = mask_file(session.session_id, index)
            session.model_generated_masks[index] = mask_array
    if mask_array is None:
        return tile.image
    return overlay_mask(tile.image, np.asarray(mask_array, dtype=bool), float(opacity))


def start_review(
    tile: Tile | None,
    threshold: float,
    device: str,
    segmentation_method: str,
    model_weights: str | None,
    show_original: bool,
    session: BatchSession | None = None,
    new_prediction: dict[str, Any] | None = None,
):
    if tile is None:
        raise gr.Error("Selecione um tile antes de abrir o editor.")
    recent = (
        new_prediction
        if new_prediction is not None
        and tuple(new_prediction.get("tile_key", ())) == tile_identity(tile)
        else None
    )
    existing = (
        session.saved_masks.get(session.current_index)
        if session is not None and session.current_index in session.saved_masks
        else None
    )
    generated = (
        session.model_generated_masks.get(session.current_index)
        if session is not None
        else None
    )
    if (
        generated is None
        and session is not None
        and session.session_id
        and session.current_index in session.model_generated_indices
    ):
        generated = mask_file(session.session_id, session.current_index)
        session.model_generated_masks[session.current_index] = generated
    if recent is not None:
        mask = np.asarray(recent["mask"], dtype=bool)
        probability = np.asarray(recent["probability"], dtype=np.float32)
        loaded_status = (
            "Nova segmentação carregada no editor no lugar da máscara persistida. "
            "A alteração será gravada somente ao salvar a revisão."
        )
    elif existing is not None:
        mask = existing.astype(bool)
        probability = mask.astype(np.float32)
        loaded_status = (
            "Anotação persistida carregada sem executar uma nova segmentação."
        )
    elif generated is not None:
        mask = generated.astype(bool)
        probability = mask.astype(np.float32)
        loaded_status = (
            "Máscara gerada por rede neural carregada para revisão humana."
        )
    elif session is not None and session.current_index in session.review_later_indices:
        mask = np.zeros(tile.image.shape[:2], dtype=bool)
        probability = mask.astype(np.float32)
        loaded_status = (
            "Tile marcado para revisão sem máscara persistida. O editor foi aberto "
            "vazio e nenhuma segmentação foi executada."
        )
    elif segmentation_method == "Rede neural — TorchScript":
        try:
            prediction = predict_neural(
                tile.image, model_weights, device, threshold
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        mask = prediction.mask
        probability = prediction.probability
        loaded_status = "Máscara neural binária carregada para edição."
    else:
        prediction = SEGMENTER.predict(tile.image, device=device)
        probability = prediction.probability
        mask = probability >= float(threshold)
        loaded_status = "Máscara ExG carregada para edição."
    return (
        configured_editor(tile, mask, show_original),
        segmentation_preview,
        probability,
        [],
        mask_statistics(mask),
        loaded_status + " Use o pincel magenta para adicionar e a borracha para remover.",
        gr.Accordion(open=True),
        mask.copy(),
    )


def start_explicit_new_review_segmentation(
    tile: Tile | None,
    threshold: float,
    device: str,
    segmentation_method: str,
    model_weights: str | None,
    show_original: bool,
):
    """Executa nova inferência somente após o comando explícito da pessoa usuária."""
    return start_review(
        tile, threshold, device, segmentation_method, model_weights,
        show_original, session=None,
    )


def configure_segmentation_method(method: str):
    neural = method == "Rede neural — TorchScript"
    return (
        gr.File(visible=neural, interactive=neural),
        "Carregue um modelo binário TorchScript (.pt/.pth). Consulte o formato no README."
        if neural else "ExG é uma heurística e não utiliza pesos.",
    )


def configured_editor(tile: Tile, mask: np.ndarray, show_original: bool = True):
    """Reconstrói o editor com canvas idêntico às dimensões do tile."""
    height, width = tile.image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(
            mask.astype(np.uint8), (width, height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    background = tile.image if show_original else np.zeros_like(tile.image)
    return gr.ImageEditor(
        value=editor_value(background, mask),
        type="numpy",
        image_mode="RGBA",
        width=width,
        sources=(),
        transforms=(),
        brush=gr.Brush(
            default_size=20,
            colors=[MASK_BRUSH_COLOR],
            default_color=MASK_BRUSH_COLOR,
            color_mode="fixed",
        ),
        eraser=gr.Eraser(default_size=30),
        format="png",
        layers=gr.LayerOptions(
            allow_additional_layers=False,
            layers=["Máscara"],
        ),
        canvas_size=(width, height),
        fixed_canvas=True,
    )


def open_review_step(tile: Tile | None):
    """Monta a aba e o editor antes de enviar background e layer."""
    if tile is None:
        raise gr.Error("Selecione um tile antes de abrir o editor.")
    return gr.Tabs(selected="review"), gr.Accordion(open=True)


def apply_mask_treatment(
    editor: dict[str, Any] | None,
    tile: Tile | None,
    history: list[np.ndarray] | None,
    min_area: int,
    fill_holes: bool,
    max_hole_area: int,
    morphology: str,
    kernel_size: int,
    show_original: bool,
):
    if tile is None:
        raise gr.Error("Nenhum tile está selecionado.")
    try:
        current = mask_from_editor(editor)
        treated = clean_mask(
            current,
            int(min_area),
            bool(fill_holes),
            int(max_hole_area),
            morphology,
            int(kernel_size),
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    new_history = [*(history or []), current.copy()]
    status = (
        f"Tratamento aplicado. Histórico: {len(new_history)} estado(s). "
        "O PNG editado continua sendo a fonte de verdade."
    )
    return (
        configured_editor(tile, treated, show_original),
        new_history,
        mask_statistics(treated),
        status,
        treated.copy(),
    )


def regenerate_from_threshold(
    editor: dict[str, Any] | None,
    tile: Tile | None,
    probability: np.ndarray | None,
    history: list[np.ndarray] | None,
    threshold: float,
    show_original: bool,
):
    if tile is None or probability is None:
        raise gr.Error("Abra o editor a partir de uma segmentação antes de recalcular.")
    current = mask_from_editor(editor)
    regenerated = probability >= float(threshold)
    new_history = [*(history or []), current.copy()]
    return (
        configured_editor(tile, regenerated, show_original),
        new_history,
        mask_statistics(regenerated),
        f"Máscara ExG recalculada com limiar {float(threshold):.2f}.",
        regenerated.copy(),
    )


def restore_initial_mask(
    editor: dict[str, Any] | None,
    tile: Tile | None,
    initial_mask: np.ndarray | None,
    history: list[np.ndarray] | None,
    show_original: bool,
):
    if tile is None or initial_mask is None:
        raise gr.Error("A máscara ExG original não está disponível.")
    current = mask_from_editor(editor)
    new_history = [*(history or []), current.copy()]
    return (
        configured_editor(tile, initial_mask, show_original),
        new_history,
        mask_statistics(initial_mask),
        "Máscara ExG inicial restaurada.",
        initial_mask.copy(),
    )


def undo_last_treatment(
    tile: Tile | None,
    history: list[np.ndarray] | None,
    show_original: bool,
):
    if tile is None or not history:
        raise gr.Error("Não há tratamento anterior para desfazer.")
    previous = history[-1]
    remaining = history[:-1]
    return (
        configured_editor(tile, previous, show_original),
        remaining,
        mask_statistics(previous),
        f"Último tratamento desfeito. Restam {len(remaining)} estado(s) no histórico.",
        previous.copy(),
    )


def toggle_editor_background(
    show_original: bool,
    editor: dict[str, Any] | None,
    tile: Tile | None,
    fallback_mask: np.ndarray | None,
):
    """Troca apenas o fundo do editor, preservando exatamente a layer atual."""
    if tile is None:
        raise gr.Error("Nenhum tile está selecionado para edição.")
    try:
        mask = mask_from_editor(editor)
    except ValueError as exc:
        if fallback_mask is None:
            raise gr.Error(str(exc)) from exc
        mask = fallback_mask.astype(bool)
    return configured_editor(tile, mask, show_original)


def prepare_roi_selection(tile: Tile | None):
    if tile is None:
        raise gr.Error("Nenhum tile está selecionado para revisão.")
    return tile.image, [], "Clique no primeiro canto e depois no canto oposto da área."


def select_roi_corner(
    tile: Tile | None,
    points: list[list[int]] | None,
    evt: gr.SelectData,
):
    if tile is None:
        raise gr.Error("Nenhum tile está selecionado para revisão.")
    raw = evt.index
    if not isinstance(raw, (tuple, list)) or len(raw) < 2:
        raise gr.Error("Não foi possível identificar o ponto selecionado.")
    point = [int(raw[0]), int(raw[1])]
    selected = list(points or [])
    selected = [point] if len(selected) >= 2 else [*selected, point]
    preview = tile.image.copy()
    for x, y in selected:
        cv2.circle(preview, (x, y), 5, (255, 170, 0), -1)
    if len(selected) == 2:
        cv2.rectangle(preview, tuple(selected[0]), tuple(selected[1]), (255, 170, 0), 3)
        x1, x2 = sorted((selected[0][0], selected[1][0]))
        y1, y2 = sorted((selected[0][1], selected[1][1]))
        status = f"Área delimitada: x={x1}:{x2}, y={y1}:{y2} ({x2-x1}×{y2-y1}px)."
    else:
        status = "Primeiro canto definido. Clique no canto oposto."
    return preview, selected, status


def apply_segmentation_in_roi(
    editor: dict[str, Any] | None,
    tile: Tile | None,
    points: list[list[int]] | None,
    history: list[np.ndarray] | None,
    threshold: float,
    device: str,
    segmentation_method: str,
    model_weights: str | None,
    show_original: bool,
    working_mask: np.ndarray | None,
):
    if tile is None or not points or len(points) != 2:
        raise gr.Error("Delimite a área com dois cliques antes de segmentar.")
    x1, x2 = sorted((int(points[0][0]), int(points[1][0])))
    y1, y2 = sorted((int(points[0][1]), int(points[1][1])))
    x2 = min(x2 + 1, tile.image.shape[1])
    y2 = min(y2 + 1, tile.image.shape[0])
    if x2 <= x1 or y2 <= y1:
        raise gr.Error("A área selecionada precisa possuir largura e altura.")
    region = tile.image[y1:y2, x1:x2]
    try:
        try:
            editor_mask = mask_from_editor(editor)
        except ValueError:
            if working_mask is None:
                raise
            editor_mask = None
        if working_mask is not None:
            current = working_mask.astype(bool).copy()
            if editor_mask is not None:
                if editor_mask.shape != current.shape:
                    editor_mask = cv2.resize(
                        editor_mask.astype(np.uint8),
                        (current.shape[1], current.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                current |= editor_mask.astype(bool)
        elif editor_mask is not None:
            current = editor_mask.astype(bool)
        else:
            raise ValueError("Não existe uma máscara atual para complementar.")
        if segmentation_method == "Rede neural — TorchScript":
            regional_mask = predict_neural(
                region, model_weights, device, threshold
            ).mask
        else:
            regional_mask = SEGMENTER.predict(region, device=device).probability >= float(threshold)
        combined = merge_mask_region(current, regional_mask, (x1, y1, x2, y2))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    updated_history = [*(history or []), current.copy()]
    return (
        configured_editor(tile, combined, show_original),
        updated_history,
        mask_statistics(combined),
        (
            f"Segmentação aplicada somente à área {x2-x1}×{y2-y1}px "
            + (
                f"com limiar {float(threshold):.2f} "
                if segmentation_method != "Rede neural — TorchScript"
                else "com a classificação neural "
            )
            + "e adicionada à máscara atual sem remover pixels já anotados."
        ),
        combined.copy(),
        [],
        tile.image,
        "Área aplicada. Delimite uma nova região para repetir o processamento.",
    )


def create_exports(
    editor: dict[str, Any] | None,
    tile: Tile | None,
    source_image: np.ndarray | None,
    category_name: str,
    simplification: float,
):
    if tile is None or source_image is None:
        raise gr.Error("Não há uma anotação pronta para exportar.")
    try:
        mask = mask_from_editor(editor)
        return export_annotation(
            mask, tile, source_image.shape, category_name.strip(), float(simplification)
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def save_review_and_advance(
    editor: dict[str, Any] | None,
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
    needs_later_review: bool,
    working_mask: np.ndarray | None,
):
    if session is None or not tiles:
        raise gr.Error("A fila de revisão não foi iniciada.")
    try:
        mask = mask_from_editor(editor)
    except ValueError as exc:
        if working_mask is None:
            raise gr.Error(str(exc)) from exc
        mask = working_mask.astype(bool)
    session.saved_masks[session.current_index] = mask.copy()
    session.model_generated_masks.pop(session.current_index, None)
    session.model_generated_indices.discard(session.current_index)
    if needs_later_review:
        session.review_later_indices.add(session.current_index)
    return advance_batch_session(
        session, tiles, collection,
        needs_later_review=True if needs_later_review else None,
    )


def clear_later_review_flag(
    session: BatchSession | None,
    tiles: list[Tile] | None,
):
    """Remove a marca somente por solicitação explícita e persiste imediatamente."""
    if session is None or not tiles:
        raise gr.Error("Nenhuma sessão de revisão foi iniciada.")
    index = session.current_index
    if index not in session.review_later_indices:
        return session, False, "Este tile não está marcado para revisão posterior.", session_summary(session)
    session.review_later_indices.discard(index)
    persist_progress(
        session, tiles, review_index=index, needs_review=False
    )
    return (
        session,
        False,
        f"Marca de revisão posterior removida explicitamente do tile {index + 1}.",
        session_summary(session),
    )


def skip_review_as_background(
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
):
    """Registra revisão explícita como background e avança para o próximo tile."""
    if session is None or not tiles:
        raise gr.Error("A fila de revisão não foi iniciada.")
    tile = tiles[session.current_index]
    session.saved_masks[session.current_index] = np.zeros(
        tile.image.shape[:2], dtype=bool
    )
    return advance_batch_session(
        session, tiles, collection,
        marked_background=True, needs_later_review=None,
    )


def advance_batch_session(
    session: BatchSession,
    tiles: list[Tile],
    collection: list[dict[str, Any]] | None,
    marked_background: bool = False,
    needs_later_review: bool | None = None,
):
    completed_index = session.current_index
    saved = len(session.saved_masks)
    total = len(session.selected_indices)
    action = "marcado como background" if marked_background else "revisão salva"
    pending = [
        index for index in session.selected_indices if index not in session.saved_masks
    ]
    if not pending:
        persist_progress(
            session, tiles,
            background_index=completed_index if marked_background else None,
            review_index=completed_index,
            needs_review=needs_later_review,
        )
        _release_completed_tile(tiles, completed_index)
        _release_completed_source(session, tiles, collection, completed_index)
        return (
            session,
            tiles[session.current_index],
            None,
            None,
            f"{action.capitalize()}. Revisão concluída: {saved}/{total} tiles salvos.",
            gr.Tabs(selected="export"),
            False,
            None,
        )
    next_index = pending[0]
    session.position = session.selected_indices.index(next_index)
    persist_progress(
        session, tiles,
        background_index=completed_index if marked_background else None,
        review_index=completed_index,
        needs_review=needs_later_review,
    )
    _release_completed_tile(tiles, completed_index)
    _release_completed_source(session, tiles, collection, completed_index)
    tile = tiles[session.current_index]
    if not tile.image.size and session.session_id:
        image = tile_file(session.session_id, session.current_index)
        tile = Tile(
            image, tile.x, tile.y, tile.valid_width, tile.valid_height,
            tile.source_index, tile.source_name,
        )
        tiles[session.current_index] = tile
    source = _source_image(session, collection, tile)
    return (
        session,
        tile,
        tile.image,
        source,
        f"{action.capitalize()}. Tile {session.position + 1} de {total} · {saved} salvo(s)",
        gr.Tabs(selected="segment"),
        False,
        None,
    )


def _release_completed_tile(tiles: list[Tile], index: int) -> None:
    """Mantém metadados do tile e libera seus pixels após persistência em disco."""
    tile = tiles[index]
    tiles[index] = Tile(
        np.empty((0, 0, 3), dtype=np.uint8), tile.x, tile.y,
        tile.valid_width, tile.valid_height, tile.source_index, tile.source_name,
    )


def _release_completed_source(
    session: BatchSession,
    tiles: list[Tile],
    collection: list[dict[str, Any]] | None,
    completed_index: int,
) -> None:
    if not collection:
        return
    source_index = tiles[completed_index].source_index
    pending_same_source = any(
        index not in session.saved_masks and tiles[index].source_index == source_index
        for index in session.selected_indices
    )
    if not pending_same_source:
        item = collection[source_index]
        item.setdefault("shape", item["image"].shape)
        item["image"] = np.empty((0, 0, 3), dtype=np.uint8)


def _source_image(
    session: BatchSession,
    collection: list[dict[str, Any]] | None,
    tile: Tile,
) -> np.ndarray:
    if collection and collection[tile.source_index]["image"].size:
        return collection[tile.source_index]["image"]
    if session.session_id:
        return source_file(session.session_id, tile.source_index)
    return tile.image


def resume_saved_session(session_id: str | None):
    if not session_id:
        raise gr.Error("Selecione uma sessão salva.")
    try:
        session, tiles, collection, manifest = load_session(session_id)
    except (OSError, ValueError, KeyError) as exc:
        raise gr.Error(f"Não foi possível retomar a sessão: {exc}") from exc
    complete = len(session.saved_masks) == len(session.selected_indices)
    if complete:
        return (
            collection, tiles, session.selected_indices, session, None, None, None,
            f"Sessão {session_id} concluída e pronta para exportação.",
            {"configurações": manifest.get("settings", {}), "tiles": session_summary(session)},
            gr.Tabs(selected="export"),
            gr.Accordion(open=False),
            False,
        )
    index = session.current_index
    tile = tiles[index]
    if not tile.image.size:
        image = tile_file(session_id, index)
        tile = Tile(
            image, tile.x, tile.y, tile.valid_width, tile.valid_height,
            tile.source_index, tile.source_name,
        )
        tiles[index] = tile
    source = _source_image(session, collection, tile)
    return (
        collection, tiles, session.selected_indices, session, tile, tile.image, source,
        f"Sessão retomada · tile {session.position + 1} de {len(session.selected_indices)} · "
        f"{len(session.saved_masks)} já salvo(s).",
        {"configurações": manifest.get("settings", {}), "tiles": session_summary(session)},
        gr.Tabs(selected="segment"),
        gr.Accordion(open=True),
        session.current_index in session.review_later_indices,
    )


def refresh_saved_sessions():
    sessions = list_sessions()
    return gr.Dropdown(choices=sessions, value=sessions[0] if sessions else None)


def session_samples_view(session: BatchSession | None):
    if session is None or not session.session_id:
        raise gr.Error("Retome uma sessão salva para gerenciar suas amostras.")
    records = session_source_summary(session.session_id)
    gallery = []
    choices = []
    for record in records:
        image = source_file(session.session_id, record["index"])
        height, width = image.shape[:2]
        scale = min(1.0, 320 / max(height, width))
        thumbnail = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        state = "EXCLUÍDA" if record["excluded"] else "ATIVA"
        origin = "ORIGINAL" if record["origin"] == "original" else "ADICIONADA"
        label = (
            f"{record['index'] + 1} · {origin} · {state} · "
            f"{record['tile_count']} tiles · {record['name']}"
        )
        gallery.append((thumbnail, label))
        if not record["excluded"]:
            choices.append(f"{record['index'] + 1} · {record['name']}")
    summary = {
        "amostras_totais": len(records),
        "amostras_ativas": sum(not record["excluded"] for record in records),
        "amostras_excluídas": sum(record["excluded"] for record in records),
        "originais": sum(record["origin"] == "original" for record in records),
        "adicionadas": sum(record["origin"] == "adicionada" for record in records),
    }
    return (
        gallery,
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
        summary,
    )


def exclude_source_from_session(
    session: BatchSession | None, selection: str | None, confirmed: bool
):
    if session is None or not session.session_id or not selection:
        raise gr.Error("Selecione uma amostra ativa para excluir.")
    if not confirmed:
        raise gr.Error("Confirme a exclusão antes de continuar.")
    source_index = int(selection.split("·", 1)[0].strip()) - 1
    source_count, tile_count = exclude_session_sources(session.session_id, [source_index])
    resumed = resume_saved_session(session.session_id)
    view = session_samples_view(resumed[3])
    return (
        *resumed, *view,
        f"{source_count} amostra(s) e {tile_count} tile(s) excluído(s) da sessão ativa.",
        False,
    )


def exclude_tiles_from_session(
    session: BatchSession | None, first_tile_number: int, confirmed: bool
):
    if session is None or not session.session_id:
        raise gr.Error("Retome uma sessão salva antes de excluir tiles.")
    if not confirmed:
        raise gr.Error("Confirme a exclusão antes de continuar.")
    _, tiles, _, _ = load_session(session.session_id)
    first_index = int(first_tile_number) - 1
    if first_index < 0 or first_index >= len(tiles):
        raise gr.Error(f"Informe um número entre 1 e {len(tiles)}.")
    count = exclude_session_tiles(session.session_id, list(range(first_index, len(tiles))))
    resumed = resume_saved_session(session.session_id)
    view = session_samples_view(resumed[3])
    return (
        *resumed, *view,
        f"{count} tile(s) excluído(s) a partir do tile {int(first_tile_number)}.",
        False,
    )


def save_preprocessed_directory(
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
    selected: list[int] | None,
):
    if not tiles or not collection:
        raise gr.Error("Gere os tiles antes de salvar o diretório.")
    session = create_session(
        tiles, collection, list(selected or range(len(tiles))),
        {"entrada": "diretório de tiles pré-processados"},
    )
    path = Path(__file__).resolve().parent / "workspace" / "sessions" / session.session_id
    return str(path), f"Diretório salvo: `{path}`"


def export_all_preprocessed_tiles(
    tiles: list[Tile] | None,
    selected: list[int] | None,
    session: BatchSession | None,
):
    """Exporta todos os tiles gerados, independentemente do filtro ou revisão."""
    if not tiles:
        raise gr.Error("Gere ou carregue os tiles antes de preparar o download.")
    selected_set = set(selected or [])
    export_dir = Path(tempfile.mkdtemp(prefix="greensegmenter_preprocessed_"))
    archive_path = export_dir / "todos_os_tiles_pre_processados.zip"
    records = []
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, tile in enumerate(tiles):
            image = tile.image
            if not image.size:
                if session is None or not session.session_id:
                    raise gr.Error(f"Os pixels do tile {index + 1} não estão disponíveis.")
                image = tile_file(session.session_id, index)
            safe_source = Path(tile.source_name).stem.replace(" ", "_")
            filename = f"{safe_source}_tile_{index + 1:06d}_x{tile.x}_y{tile.y}.png"
            buffer = io.BytesIO()
            Image.fromarray(image.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
            archive.writestr(f"tiles/{filename}", buffer.getvalue())
            records.append({
                "index": index,
                "file": f"tiles/{filename}",
                "source": tile.source_name,
                "x": tile.x,
                "y": tile.y,
                "valid_width": tile.valid_width,
                "valid_height": tile.valid_height,
                "selected": index in selected_set,
            })
        archive.writestr(
            "tiles_manifest.json",
            json.dumps({"total": len(tiles), "tiles": records}, ensure_ascii=False, indent=2),
        )
    return str(archive_path), {
        "tiles_exportados": len(tiles),
        "selecionados": len(selected_set),
        "não_selecionados": len(tiles) - len(selected_set),
    }


def load_local_tile_directory(path: str | None):
    if not path or not path.strip():
        raise gr.Error("Informe o diretório local criado pela aplicação.")
    candidate = Path(path.strip()).expanduser()
    session_id = candidate.name if candidate.name else path.strip()
    return resume_saved_session(session_id)


def session_tile_options(session: BatchSession | None, state_filter: str):
    if session is None:
        raise gr.Error("Nenhuma sessão de tiles foi carregada.")
    saved = set(session.saved_masks)
    review = set(session.review_later_indices)
    generated = set(session.model_generated_indices) - saved
    indices = []
    for index in session.selected_indices:
        status = "revisão posterior" if index in review else (
            "revisado por humano" if index in saved else (
                "gerado por rede neural" if index in generated else "novo"
            )
        )
        if state_filter == "Novos" and index in saved | generated:
            continue
        if state_filter == "Salvos" and index not in saved:
            continue
        if state_filter == "Revisão posterior" and index not in review:
            continue
        if state_filter == "Gerados por rede" and index not in generated:
            continue
        indices.append(f"{index + 1} · {status}")
    return session_summary(session), gr.Dropdown(choices=indices, value=indices[0] if indices else None)


def select_session_tile(
    selection: str | None,
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
):
    if not selection or session is None or not tiles:
        raise gr.Error("Selecione um tile da sessão.")
    index = int(selection.split("·", 1)[0].strip()) - 1
    session.position = session.selected_indices.index(index)
    tile = tiles[index]
    if not tile.image.size:
        if not session.session_id:
            raise gr.Error("Os pixels desse tile não estão disponíveis.")
        image = tile_file(session.session_id, index)
        tile = Tile(
            image, tile.x, tile.y, tile.valid_width, tile.valid_height,
            tile.source_index, tile.source_name,
        )
        tiles[index] = tile
    source = _source_image(session, collection, tile)
    status = "marcado para revisão" if index in session.review_later_indices else (
        "revisado por humano" if index in session.saved_masks else (
            "gerado por rede neural"
            if index in session.model_generated_indices else "novo"
        )
    )
    # Máscaras neurais são mantidas sob demanda para a navegação não acumular RAM.
    session.model_generated_masks = {
        cached_index: cached_mask
        for cached_index, cached_mask in session.model_generated_masks.items()
        if cached_index == index
    }
    saved_mask = session.saved_masks.get(index)
    if saved_mask is None:
        saved_mask = session.model_generated_masks.get(index)
    if (
        saved_mask is None
        and session.session_id
        and index in session.model_generated_indices
    ):
        saved_mask = mask_file(session.session_id, index)
        session.model_generated_masks[index] = saved_mask
    mask_preview = (
        saved_mask.astype(np.uint8) * 255 if saved_mask is not None else None
    )
    return (
        session, tile, tile.image, source,
        f"Tile {session.position + 1} de {len(session.selected_indices)} · "
        f"índice original {index + 1} · {status}.",
        gr.Tabs(selected="segment"), gr.Accordion(open=True),
        index in session.review_later_indices,
        mask_preview,
    )


def navigate_session_tile(
    offset: int,
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
):
    """Navega pela seleção e carrega pixels e máscara persistida do novo tile."""
    if session is None or not tiles:
        raise gr.Error("Retome uma sessão antes de navegar pelos tiles.")
    try:
        index = move_session_position(session, offset)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return select_session_tile(str(index + 1), session, tiles, collection)


def previous_session_tile(session, tiles, collection):
    return navigate_session_tile(-1, session, tiles, collection)


def next_session_tile(session, tiles, collection):
    return navigate_session_tile(1, session, tiles, collection)


def current_session_mask_preview(
    session: BatchSession | None,
) -> np.ndarray | None:
    """Carrega a máscara atual ao retomar uma sessão, sem nova inferência."""
    if session is None or not session.selected_indices:
        return None
    index = session.current_index
    saved_mask = session.saved_masks.get(index)
    if saved_mask is None:
        saved_mask = session.model_generated_masks.get(index)
    if (
        saved_mask is None
        and session.session_id
        and index in session.model_generated_indices
    ):
        saved_mask = mask_file(session.session_id, index)
        session.model_generated_masks[index] = saved_mask
    preview = saved_mask.astype(np.uint8) * 255 if saved_mask is not None else None
    return preview


def load_preprocessed_tiles(files: list[str] | None):
    if not files:
        raise gr.Error("Selecione tiles pré-processados para continuar.")
    collection, tiles = [], []
    for index, path in enumerate(files):
        with Image.open(path) as image:
            array = to_rgb_array(image)
        name = Path(path).name
        collection.append({"name": name, "image": array, "shape": array.shape})
        height, width = array.shape[:2]
        size = max(height, width)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[:height, :width] = array
        tiles.append(Tile(padded, 0, 0, width, height, index, name))
    selected = list(range(len(tiles)))
    session = create_session(
        tiles, collection, selected, {"entrada": "tiles pré-processados"}
    )
    first = tiles[0]
    return (
        collection,
        collection[0]["image"],
        tiles,
        selected,
        session,
        first,
        first.image,
        f"{len(tiles)} tile(s) pré-processado(s) carregado(s).",
        gr.Tabs(selected="segment"),
        gr.Accordion(open=True),
    )


def create_batch_exports(
    session: BatchSession | None,
    tiles: list[Tile] | None,
    collection: list[dict[str, Any]] | None,
    category_name: str,
    simplification: float,
):
    if session is None or not tiles or not collection:
        raise gr.Error("Não existe uma sessão de revisão para exportar.")
    if not session.saved_masks:
        raise gr.Error("Salve ao menos um tile antes de exportar o lote parcial.")
    try:
        masks_zip, coco_json, summary = export_annotation_batch(
            session.saved_masks, tiles, collection,
            category_name.strip(), float(simplification),
        )
        saved = set(session.saved_masks)
        pending = set(session.selected_indices) - saved
        tiles_zip = export_saved_tiles(session, tiles)
        return masks_zip, coco_json, tiles_zip, {
            **summary,
            "tipo_exportação": "completa" if not pending else "parcial",
            "tiles_totais": len(session.selected_indices),
            "tiles_pendentes": len(pending),
            "revisão_posterior": len(session.review_later_indices),
        }
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc


def export_saved_tiles(session: BatchSession, tiles: list[Tile]) -> str:
    """Cria um ZIP com todos os tiles, separados pela existência de máscara."""
    export_dir = Path(tempfile.mkdtemp(prefix="greensegmenter_tiles_"))
    archive_path = export_dir / "tiles_rgb.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, tile in enumerate(tiles):
            image = tile.image
            if not image.size:
                if not session.session_id:
                    raise ValueError(f"Os pixels do tile {index + 1} não estão disponíveis.")
                image = tile_file(session.session_id, index)
            safe_source = Path(tile.source_name).stem.replace(" ", "_")
            filename = (
                f"{safe_source}_tile_{index + 1:04d}_x{tile.x}_y{tile.y}.png"
            )
            folder = "com_mascara" if index in session.saved_masks else "sem_mascara"
            png_path = export_dir / filename
            Image.fromarray(image.astype(np.uint8), mode="RGB").save(png_path)
            archive.write(png_path, arcname=f"{folder}/{filename}")
    return str(archive_path)


CSS = """
.hero {max-width: 900px; margin: 0 auto .5rem; text-align: center;}
.upload-start {max-width: 820px; margin: 1rem auto;}
#editor-workspace.expanded-mask-editor {
  position: fixed; inset: 10px; z-index: 10000; padding: 12px;
  overflow: auto; background: var(--background-fill-primary);
  border: 1px solid var(--border-color-primary); border-radius: 10px;
}
#editor-workspace.expanded-mask-editor #mask-editor {
  max-width: none !important; margin: 0 auto;
  transform-origin: top center;
}
"""


with gr.Blocks(title="GreenSegmenter | Anotação de vegetação") as demo:
    collection_state = gr.State(value=None)
    source_state = gr.State(value=None)
    tiles_state = gr.State(value=None)
    tile_scores_state = gr.State(value=None)
    filtered_indices_state = gr.State(value=None)
    similar_tiles_state = gr.State(value=None)
    additional_collection_state = gr.State(value=None)
    additional_tiles_state = gr.State(value=None)
    additional_selected_state = gr.State(value=None)
    batch_session_state = gr.State(value=None)
    selected_tile_state = gr.State(value=None)
    inspected_tile_index_state = gr.State(value=None)
    selected_collection_index_state = gr.State(value=None)
    original_mask_state = gr.State(value=None)
    probability_state = gr.State(value=None)
    treatment_history_state = gr.State(value=None)
    latest_prediction_state = gr.State(value=None)
    roi_points_state = gr.State(value=None)
    working_mask_state = gr.State(value=None)

    gr.Markdown(
        "# GreenSegmenter\n### Anotação de vegetação com ExG e redes neurais",
        elem_classes="hero",
    )
    with gr.Tabs(selected="home") as workflow:
        with gr.Tab("⌂ Início", id="home"):
            gr.Markdown(
                "## Workspace de imagens agrícolas\nEscolha por onde começar. "
                "As ferramentas podem ser usadas independentemente conforme os dados disponíveis."
            )
            with gr.Row():
                go_images = gr.Button("📁 Carregar imagens", variant="primary")
                go_preprocess = gr.Button("▦ Pré-processamento")
                go_segment = gr.Button("◉ Segmentar")
                go_review = gr.Button("✏️ Revisar máscaras")
                go_export = gr.Button("📦 Exportar")
            with gr.Accordion("↻ Retomar uma sessão salva", open=True):
                gr.Markdown(
                    "O progresso é salvo localmente a cada tile concluído. "
                    "Tiles já revisados são retirados da memória e permanecem disponíveis em disco."
                )
                with gr.Row():
                    saved_session = gr.Dropdown(
                        choices=list_sessions(),
                        value=list_sessions()[0] if list_sessions() else None,
                        label="Sessão",
                    )
                    refresh_sessions = gr.Button("↻ Atualizar lista")
                    resume_session_button = gr.Button(
                        "Retomar sessão", variant="primary"
                    )
                resumed_settings = gr.JSON(label="Configurações persistidas")

        with gr.Tab("📁 Imagens", id="images"):
            with gr.Column(elem_classes="upload-start"):
                gr.Markdown(
                    "## Coleção de imagens\nEnvie arquivos individuais, vários arquivos ou uma pasta inteira."
                )
                with gr.Row():
                    image_files = gr.File(
                        label="Uma ou mais imagens",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    image_directory = gr.File(
                        label="Diretório de imagens",
                        file_count="directory",
                        file_types=["image"],
                        type="filepath",
                    )
                load_collection = gr.Button("Carregar coleção", variant="primary")
                collection_summary = gr.JSON(label="Resumo da coleção")
                with gr.Accordion(
                    "▸ Visualizar imagens carregadas", open=False
                ) as collection_accordion:
                    collection_gallery = gr.Gallery(
                        label="Imagens carregadas", columns=4, rows=2, height=420
                    )
                    selected_collection_label = gr.Markdown(
                        "Clique em uma imagem para selecioná-la."
                    )
                    use_image_directly = gr.Button(
                        "◉ Usar imagem selecionada diretamente na segmentação",
                        variant="primary",
                    )
                gr.Markdown(
                    "### Atalho para dados já pré-processados\nCarregue tiles diretamente "
                    "para ignorar o sliding window e seguir às comparações."
                )
                preprocessed_files = gr.File(
                    label="Tiles já pré-processados",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                load_preprocessed = gr.Button(
                    "Abrir tiles diretamente na segmentação", variant="secondary"
                )
                gr.Markdown("### Retomar um diretório criado pela aplicação")
                local_tile_directory = gr.Textbox(
                    label="Diretório local de tiles",
                    placeholder="workspace/sessions/<id-da-sessao>",
                )
                load_local_directory = gr.Button("Abrir diretório de tiles")

        with gr.Tab("▦ Pré-processamento", id="preprocess"):
            gr.Markdown(
                "## Sliding window\nConfigure o tamanho e a sobreposição. Tiles de borda "
                "recebem padding para manter o formato quadrado."
            )
            selected_sources = gr.Dropdown(
                choices=[], value=[], multiselect=True,
                label="Imagens que serão pré-processadas",
            )
            with gr.Row():
                tile_size = gr.Dropdown(
                    choices=[256, 384, 512, 768, 1024],
                    value=512,
                    label="Tamanho do tile (px)",
                )
                overlap = gr.Slider(
                    0, 75, value=20, step=5, label="Sobreposição (%)"
                )
                compute_device = gr.Dropdown(
                    choices=DEVICES,
                    value=DEFAULT_DEVICE,
                    label="Dispositivo",
                    info=(
                        "GPU aparece somente quando CuPy e CUDA estão disponíveis."
                    ),
                )
            with gr.Row():
                filter_exg_threshold = gr.Slider(
                    0, 1, value=0.5, step=0.01,
                    label="Limiar ExG para considerar um pixel verde",
                )
                minimum_exg_presence = gr.Slider(
                    0, 100, value=5, step=0.5,
                    label="Presença mínima de ExG no tile (%)",
                )
            with gr.Row():
                filter_similar = gr.Checkbox(
                    value=True,
                    label="Filtrar semelhantes",
                    info="Mantém o primeiro tile de cada grupo visualmente semelhante.",
                )
                similarity_threshold = gr.Slider(
                    0, 100, value=90, step=1,
                    label="Limiar de similaridade (%)",
                    info="Quanto maior, mais parecidos os tiles precisam ser.",
                )
                create_tiles = gr.Button("▦ Gerar e filtrar tiles", variant="primary")
            tiling_status = gr.Markdown("Configure o método e gere os tiles.")
            with gr.Accordion(
                "▸ Galeria de tiles e seleção do lote", open=False
            ) as tile_gallery_accordion:
                tile_gallery = gr.Gallery(
                    label="Tiles — clique para selecionar",
                    columns=4,
                    rows=2,
                    height=480,
                    object_fit="contain",
                    allow_preview=False,
                    format="png",
                )
                with gr.Row():
                    selected_tile_preview = gr.Image(
                        label="Tile selecionado", interactive=False, height=320
                    )
                    with gr.Column():
                        selected_tile_label = gr.Markdown(
                            "Clique em qualquer tile para inspecioná-lo."
                        )
                        with gr.Row():
                            add_tile = gr.Button("＋ Adicionar ao lote", variant="primary")
                            remove_tile = gr.Button("− Remover do lote", variant="secondary")
                        continue_tile = gr.Button(
                            "Confirmar aprovados e iniciar lote →", variant="primary"
                        )
                with gr.Row():
                    save_tile_directory = gr.Button("💾 Salvar diretório de tiles")
                    saved_tile_directory = gr.Textbox(
                        label="Diretório salvo", interactive=False
                    )
                saved_tile_directory_status = gr.Markdown()
                with gr.Row():
                    prepare_all_tiles_download = gr.Button(
                        "Preparar download de todos os tiles"
                    )
                    download_all_tiles = gr.DownloadButton(
                        "Baixar todos os tiles pré-processados (ZIP)"
                    )
                all_tiles_export_summary = gr.JSON(label="Resumo do download de tiles")

            with gr.Accordion("＋ Adicionar amostras à sessão", open=False):
                gr.Markdown(
                    "Retome uma sessão salva e selecione novas imagens. Os novos tiles "
                    "serão comparados também com os tiles que já pertencem à sessão."
                )
                with gr.Row():
                    additional_files = gr.File(
                        label="Novas imagens",
                        file_count="multiple",
                        file_types=["image"],
                        type="filepath",
                    )
                    additional_directory = gr.File(
                        label="Diretório com novas imagens",
                        file_count="directory",
                        file_types=["image"],
                        type="filepath",
                    )
                ignore_processed_images = gr.Checkbox(
                    value=True,
                    label="Ignorar imagens já processadas",
                    info=(
                        "Compara os pixels das novas imagens com as fontes que já "
                        "pertencem à sessão antes de gerar os tiles."
                    ),
                )
                generate_additional = gr.Button(
                    "Gerar novos tiles", variant="secondary"
                )
                additional_status = gr.Markdown(
                    "Retome uma sessão e escolha as novas imagens."
                )
                additional_gallery = gr.Gallery(
                    label="Novos tiles — verde: aprovado; amarelo: semelhante; cinza: ExG",
                    columns=4, rows=2, height=420, object_fit="contain",
                    allow_preview=True, format="png",
                )
                confirm_additional = gr.Button(
                    "Confirmar e adicionar à sessão", variant="primary"
                )

            with gr.Accordion("⚙ Gerenciar amostras e tiles da sessão", open=False):
                gr.Markdown(
                    "Visualize as imagens originais e adicionadas. A exclusão retira "
                    "amostras e tiles da fila e das exportações, mas preserva os "
                    "arquivos no disco para recuperação."
                )
                refresh_session_samples = gr.Button("Visualizar amostras da sessão")
                session_samples_summary = gr.JSON(label="Resumo das amostras")
                session_samples_gallery = gr.Gallery(
                    label="Amostras utilizadas",
                    columns=3, rows=2, height=420, object_fit="contain",
                    allow_preview=True, format="png",
                )
                with gr.Row():
                    session_source_to_exclude = gr.Dropdown(
                        choices=[], label="Amostra ativa para excluir"
                    )
                    exclude_session_source_button = gr.Button(
                        "Excluir amostra e seus tiles", variant="stop"
                    )
                with gr.Row():
                    first_tile_to_exclude = gr.Number(
                        value=1, precision=0, minimum=1,
                        label="Excluir a partir do tile número",
                    )
                    exclude_session_tiles_button = gr.Button(
                        "Excluir este tile e todos os seguintes", variant="stop"
                    )
                confirm_session_exclusion = gr.Checkbox(
                    value=False,
                    label="Confirmo que desejo excluir os itens da sessão ativa",
                )
                session_management_status = gr.Markdown()

        with gr.Tab("◉ Segmentação", id="segment"):
            gr.Markdown(
                "## Segmentação\nA inferência é executada somente no tile "
                "selecionado."
            )
            with gr.Accordion("Tiles da sessão", open=False):
                with gr.Row():
                    tile_state_filter = gr.Radio(
                        ["Todos", "Novos", "Salvos", "Gerados por rede", "Revisão posterior"],
                        value="Todos",
                        label="Mostrar",
                    )
                    refresh_tile_states = gr.Button("Atualizar estados")
                tile_session_summary = gr.JSON(label="Resumo dos tiles")
                with gr.Row():
                    session_tile_selector = gr.Dropdown(label="Tile")
                    load_session_tile = gr.Button("Carregar tile selecionado")
            with gr.Accordion("Método e pesos", open=True):
                segmentation_method = gr.Dropdown(
                    choices=["ExG — excesso de verde", "Rede neural — TorchScript"],
                    value="ExG — excesso de verde",
                    label="Método de segmentação",
                    info="ExG é processamento de imagem e não utiliza arquivo de pesos.",
                )
                model_weights = gr.File(
                    label="Modelo TorchScript binário (.pt/.pth)",
                    interactive=True,
                    visible=False,
                    file_count="single",
                    file_types=[".pt", ".pth"],
                    value=None,
                    type="filepath",
                )
                model_configuration_status = gr.Markdown(
                    "ExG é uma heurística e não utiliza pesos."
                )
            with gr.Accordion(
                "▸ Tile atual e resultado", open=False
            ) as current_tile_accordion:
                batch_progress = gr.Markdown("Fila de tiles ainda não iniciada.")
                with gr.Row():
                    previous_tile = gr.Button(
                        "← Tile anterior", variant="secondary"
                    )
                    next_tile = gr.Button(
                        "Próximo tile →", variant="secondary"
                    )
                with gr.Row():
                    tile_for_analysis = gr.Image(
                        label="Tile selecionado", interactive=False, height=360
                    )
                    segmentation_preview = gr.Image(
                        label="Segmentação resultante",
                        interactive=False,
                        height=360,
                    )
                    current_mask_overlay = gr.Image(
                        label="Overlay da máscara",
                        interactive=False,
                        height=360,
                    )
            with gr.Row():
                threshold = gr.Slider(0, 1, value=0.5, step=0.01, label="Limiar")
                opacity = gr.Slider(0, 1, value=0.45, step=0.05, label="Opacidade")
                analyze = gr.Button("◉ Executar segmentação", variant="primary")
            analysis_status = gr.Markdown("Execute a análise do tile selecionado.")
            gr.Markdown("### Próxima ação")
            with gr.Row():
                open_review = gr.Button(
                    "✏️ Revisar máscara e exportar anotações", variant="primary"
                )
                segment_skip_background = gr.Button(
                    "→ Salvar como background e avançar", variant="secondary"
                )

        with gr.Tab("✏️ Revisão", id="review") as review_tab:
            gr.Markdown(
                "## Edição da anotação\nPinte regiões ausentes em magenta ou use a borracha "
                "para apagar predições incorretas. Não recorte nem redimensione o canvas, "
                "pois ele permanece alinhado às coordenadas do tile."
            )
            with gr.Row():
                reload_editor = gr.Button(
                    "◉ Substituir o editor por uma nova segmentação",
                    variant="secondary",
                )
                gr.Markdown(
                    "Esse botão executa novamente o método selecionado. A máscara "
                    "persistida só será sobrescrita se você salvar a revisão depois."
                )
                skip_as_background = gr.Button(
                    "→ Marcar como background e avançar",
                    variant="primary",
                )
            show_original_in_editor = gr.Checkbox(
                value=True,
                label="Mostrar imagem original no fundo",
                info="Desmarque para visualizar somente a layer de segmentação sobre fundo preto.",
            )
            with gr.Accordion(
                "▸ Abrir editor de máscara", open=False
            ) as editor_accordion:
                with gr.Row():
                    with gr.Column(elem_id="editor-workspace"):
                        expand_editor = gr.Button(
                            "⛶ Expandir/recolher editor", variant="secondary"
                        )
                        annotation_editor = gr.ImageEditor(
                            label="Máscara editável — magenta=classe anotada",
                            type="numpy",
                            image_mode="RGBA",
                            width=512,
                            elem_id="mask-editor",
                            sources=(),
                            transforms=(),
                            brush=gr.Brush(
                                default_size=20,
                                colors=[MASK_BRUSH_COLOR],
                                default_color=MASK_BRUSH_COLOR,
                                color_mode="fixed",
                            ),
                            eraser=gr.Eraser(default_size=30),
                            format="png",
                            layers=gr.LayerOptions(
                                allow_additional_layers=False,
                                layers=["Máscara"],
                            ),
                            canvas_size=(512, 512),
                            fixed_canvas=True,
                        )
                    with gr.Column():
                        editor_status = gr.Markdown("Abra esta etapa a partir da segmentação.")
                        annotation_stats = gr.JSON(label="Estatísticas da máscara")
                        with gr.Accordion("Limiar ExG", open=False):
                            editor_threshold = gr.Slider(
                                0, 1, value=0.5, step=0.01,
                                label="Limiar customizável",
                            )
                            regenerate = gr.Button("Recalcular máscara pelo limiar")
                        with gr.Accordion("Limpeza e buracos", open=False):
                            min_component_area = gr.Number(
                                value=25, precision=0, minimum=0,
                                label="Remover componentes menores que (pixels)",
                            )
                            fill_holes_control = gr.Checkbox(
                                value=True, label="Preencher buracos internos"
                            )
                            max_hole_area = gr.Number(
                                value=500, precision=0, minimum=0,
                                label="Área máxima do buraco (0 = todos)",
                            )
                            morphology = gr.Radio(
                                ["Nenhuma", "Abertura", "Fechamento"],
                                value="Nenhuma",
                                label="Operação morfológica",
                            )
                            kernel_size = gr.Radio(
                                [3, 5, 7, 9], value=3, label="Kernel morfológico"
                            )
                            apply_treatment = gr.Button(
                                "Aplicar tratamento", variant="primary"
                            )
                        with gr.Accordion("Segmentar uma área delimitada", open=False):
                            gr.Markdown(
                                "Clique em dois cantos opostos. A predição será aplicada "
                                "somente dentro do retângulo e adicionada à máscara atual."
                            )
                            roi_selector = gr.Image(
                                label="Seleção da área",
                                interactive=False,
                                height=320,
                            )
                            roi_status = gr.Markdown(
                                "Prepare a seleção para delimitar uma área."
                            )
                            roi_segmentation_method = gr.Dropdown(
                                choices=[
                                    "ExG — excesso de verde",
                                    "Rede neural — TorchScript",
                                ],
                                value="ExG — excesso de verde",
                                label="Método aplicado somente na área",
                            )
                            roi_threshold = gr.Slider(
                                0,
                                1,
                                value=0.5,
                                step=0.01,
                                label="Limiar da segmentação na área",
                                info=(
                                    "Valores menores são mais permissivos; valores "
                                    "maiores exigem maior probabilidade ExG ou rede neural."
                                ),
                            )
                            with gr.Row():
                                prepare_roi = gr.Button("Delimitar nova área")
                                apply_roi_segmentation = gr.Button(
                                    "Aplicar segmentação na área", variant="primary"
                                )
                        with gr.Row():
                            undo_treatment = gr.Button("↶ Desfazer tratamento")
                            restore_mask = gr.Button("Restaurar ExG inicial")
            gr.Markdown(
                "### Exportação\nA máscara PNG preserva todos os pixels. O COCO JSON "
                "contém contornos externos em coordenadas da imagem original."
            )
            with gr.Row():
                category_name = gr.Textbox(value="vegetacao", label="Nome da classe")
                polygon_simplification = gr.Slider(
                    0, 10, value=1.5, step=0.25,
                    label="Simplificação dos polígonos (px)",
                )
                export_button = gr.Button("Preparar arquivos", variant="primary")
            export_summary = gr.JSON(label="Resumo da exportação")
            with gr.Row():
                download_mask = gr.DownloadButton("Baixar máscara PNG")
                download_coco = gr.DownloadButton("Baixar COCO JSON")
            needs_later_review = gr.Checkbox(
                value=False,
                label="Manter/marcar este tile para revisão posterior",
            )
            clear_later_review = gr.Button(
                "Remover explicitamente a marca de revisão posterior",
                variant="secondary",
            )
            save_and_next = gr.Button(
                "✓ Salvar revisão e avançar para o próximo tile", variant="primary"
            )

        with gr.Tab("📦 Exportação", id="export"):
            gr.Markdown(
                "## Exportação consolidada ou parcial\nExporte os tiles já salvos a "
                "qualquer momento. Tiles novos ainda pendentes não serão incluídos. "
                "O ZIP contém as máscaras PNG e o COCO usa coordenadas globais."
            )
            final_category_name = gr.Textbox(value="vegetacao", label="Nome da classe")
            final_simplification = gr.Slider(
                0, 10, value=1.5, step=0.25, label="Simplificação dos polígonos (px)"
            )
            final_export_button = gr.Button(
                "📦 Exportar tiles salvos", variant="primary"
            )
            final_export_summary = gr.JSON(label="Resumo do lote")
            with gr.Row():
                download_masks_zip = gr.DownloadButton("Baixar todas as máscaras (ZIP)")
                download_batch_coco = gr.DownloadButton("Baixar COCO consolidado")
                download_tiles_zip = gr.DownloadButton(
                    "Baixar todos os tiles RGB (ZIP)"
                )

    for button, destination in (
        (go_images, "images"),
        (go_preprocess, "preprocess"),
        (go_segment, "segment"),
        (go_review, "review"),
        (go_export, "export"),
    ):
        button.click(lambda tab=destination: gr.Tabs(selected=tab), outputs=workflow)
    refresh_sessions.click(
        refresh_saved_sessions,
        outputs=saved_session,
        queue=False,
    )
    resume_session_event = resume_session_button.click(
        resume_saved_session,
        inputs=saved_session,
        outputs=[
            collection_state,
            tiles_state,
            filtered_indices_state,
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            resumed_settings,
            workflow,
            current_tile_accordion,
            needs_later_review,
        ],
    )
    resume_session_event.then(
        current_session_mask_preview,
        inputs=batch_session_state,
        outputs=segmentation_preview,
        queue=False,
    )
    resume_session_event.then(
        render_current_mask_overlay,
        inputs=[
            selected_tile_state,
            batch_session_state,
            latest_prediction_state,
            opacity,
        ],
        outputs=current_mask_overlay,
        queue=False,
    )
    load_collection.click(
        load_image_collection,
        inputs=[image_files, image_directory],
        outputs=[
            collection_state,
            source_state,
            collection_gallery,
            selected_sources,
            collection_summary,
            workflow,
            collection_accordion,
        ],
    )
    collection_gallery.select(
        inspect_collection_image,
        inputs=collection_state,
        outputs=[
            selected_collection_index_state,
            source_state,
            selected_collection_label,
        ],
        queue=False,
    )
    use_image_directly.click(
        use_collection_image_directly,
        inputs=[collection_state, selected_collection_index_state],
        outputs=[
            tiles_state,
            filtered_indices_state,
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            current_tile_accordion,
            needs_later_review,
        ],
    )
    create_tiles.click(
        generate_tiles,
        inputs=[
            collection_state,
            selected_sources,
            tile_size,
            overlap,
            filter_exg_threshold,
            minimum_exg_presence,
            filter_similar,
            similarity_threshold,
            compute_device,
        ],
        outputs=[
            tile_gallery,
            tiles_state,
            tile_scores_state,
            filtered_indices_state,
            similar_tiles_state,
            selected_tile_preview,
            selected_tile_state,
            tiling_status,
            tile_gallery_accordion,
        ],
    )
    tile_gallery.select(
        inspect_tile,
        inputs=[
            tiles_state,
            tile_scores_state,
            filtered_indices_state,
            similar_tiles_state,
        ],
        outputs=[
            selected_tile_preview,
            selected_tile_state,
            inspected_tile_index_state,
            selected_tile_label,
        ],
    )
    add_tile.click(
        add_inspected_tile,
        inputs=[
            tiles_state,
            tile_scores_state,
            filtered_indices_state,
            similar_tiles_state,
            inspected_tile_index_state,
        ],
        outputs=[tile_gallery, filtered_indices_state, selected_tile_label, tiling_status],
        queue=False,
    )
    remove_tile.click(
        remove_inspected_tile,
        inputs=[
            tiles_state,
            tile_scores_state,
            filtered_indices_state,
            similar_tiles_state,
            inspected_tile_index_state,
        ],
        outputs=[tile_gallery, filtered_indices_state, selected_tile_label, tiling_status],
        queue=False,
    )
    generate_additional.click(
        generate_additional_samples,
        inputs=[
            additional_files,
            additional_directory,
            batch_session_state,
            tiles_state,
            tile_size,
            overlap,
            filter_exg_threshold,
            minimum_exg_presence,
            ignore_processed_images,
            filter_similar,
            similarity_threshold,
            compute_device,
        ],
        outputs=[
            additional_collection_state,
            additional_tiles_state,
            additional_selected_state,
            additional_gallery,
            additional_status,
        ],
    )
    confirm_additional.click(
        confirm_additional_samples,
        inputs=[
            batch_session_state,
            tiles_state,
            collection_state,
            additional_tiles_state,
            additional_collection_state,
            additional_selected_state,
            similarity_threshold,
        ],
        outputs=[
            collection_state,
            tiles_state,
            filtered_indices_state,
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            resumed_settings,
            workflow,
            current_tile_accordion,
            needs_later_review,
            additional_gallery,
            additional_status,
        ],
    )
    refresh_session_samples.click(
        session_samples_view,
        inputs=batch_session_state,
        outputs=[
            session_samples_gallery,
            session_source_to_exclude,
            session_samples_summary,
        ],
    )
    management_outputs = [
        collection_state,
        tiles_state,
        filtered_indices_state,
        batch_session_state,
        selected_tile_state,
        tile_for_analysis,
        source_state,
        batch_progress,
        resumed_settings,
        workflow,
        current_tile_accordion,
        needs_later_review,
        session_samples_gallery,
        session_source_to_exclude,
        session_samples_summary,
        session_management_status,
        confirm_session_exclusion,
    ]
    exclude_session_source_button.click(
        exclude_source_from_session,
        inputs=[
            batch_session_state,
            session_source_to_exclude,
            confirm_session_exclusion,
        ],
        outputs=management_outputs,
    )
    exclude_session_tiles_button.click(
        exclude_tiles_from_session,
        inputs=[
            batch_session_state,
            first_tile_to_exclude,
            confirm_session_exclusion,
        ],
        outputs=management_outputs,
    )
    continue_tile.click(
        start_batch,
        inputs=[tiles_state, filtered_indices_state, collection_state],
        outputs=[
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            current_tile_accordion,
        ],
    )
    load_preprocessed.click(
        load_preprocessed_tiles,
        inputs=preprocessed_files,
        outputs=[
            collection_state,
            source_state,
            tiles_state,
            filtered_indices_state,
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            batch_progress,
            workflow,
            current_tile_accordion,
        ],
    )
    load_local_directory.click(
        load_local_tile_directory,
        inputs=local_tile_directory,
        outputs=[
            collection_state,
            tiles_state,
            filtered_indices_state,
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            resumed_settings,
            workflow,
            current_tile_accordion,
            needs_later_review,
        ],
    )
    save_tile_directory.click(
        save_preprocessed_directory,
        inputs=[tiles_state, collection_state, filtered_indices_state],
        outputs=[saved_tile_directory, saved_tile_directory_status],
    )
    prepare_all_tiles_download.click(
        export_all_preprocessed_tiles,
        inputs=[tiles_state, filtered_indices_state, batch_session_state],
        outputs=[download_all_tiles, all_tiles_export_summary],
    )
    refresh_tile_states.click(
        session_tile_options,
        inputs=[batch_session_state, tile_state_filter],
        outputs=[tile_session_summary, session_tile_selector],
        queue=False,
    )
    tile_state_filter.change(
        session_tile_options,
        inputs=[batch_session_state, tile_state_filter],
        outputs=[tile_session_summary, session_tile_selector],
        queue=False,
    )
    load_session_tile_event = load_session_tile.click(
        select_session_tile,
        inputs=[
            session_tile_selector,
            batch_session_state,
            tiles_state,
            collection_state,
        ],
        outputs=[
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            current_tile_accordion,
            needs_later_review,
            segmentation_preview,
        ],
    )
    load_session_tile_event.then(
        render_current_mask_overlay,
        inputs=[
            selected_tile_state,
            batch_session_state,
            latest_prediction_state,
            opacity,
        ],
        outputs=current_mask_overlay,
        queue=False,
    )
    session_navigation_outputs = [
        batch_session_state,
        selected_tile_state,
        tile_for_analysis,
        source_state,
        batch_progress,
        workflow,
        current_tile_accordion,
        needs_later_review,
        segmentation_preview,
    ]
    previous_tile_event = previous_tile.click(
        previous_session_tile,
        inputs=[batch_session_state, tiles_state, collection_state],
        outputs=session_navigation_outputs,
    )
    next_tile_event = next_tile.click(
        next_session_tile,
        inputs=[batch_session_state, tiles_state, collection_state],
        outputs=session_navigation_outputs,
    )
    for navigation_event in (previous_tile_event, next_tile_event):
        navigation_event.then(
            render_current_mask_overlay,
            inputs=[
                selected_tile_state,
                batch_session_state,
                latest_prediction_state,
                opacity,
            ],
            outputs=current_mask_overlay,
            queue=False,
        )
    analysis_event = analyze.click(
        run_tile_analysis,
        inputs=[
            selected_tile_state,
            source_state,
            threshold,
            opacity,
            compute_device,
            segmentation_method,
            model_weights,
        ],
        outputs=[segmentation_preview, analysis_status, latest_prediction_state],
        show_progress="full",
    )
    analysis_event.then(
        render_current_mask_overlay,
        inputs=[
            selected_tile_state,
            batch_session_state,
            latest_prediction_state,
            opacity,
        ],
        outputs=current_mask_overlay,
        queue=False,
    )
    tile_for_analysis.change(
        render_current_mask_overlay,
        inputs=[
            selected_tile_state,
            batch_session_state,
            latest_prediction_state,
            opacity,
        ],
        outputs=current_mask_overlay,
        queue=False,
    )
    opacity.change(
        render_current_mask_overlay,
        inputs=[
            selected_tile_state,
            batch_session_state,
            latest_prediction_state,
            opacity,
        ],
        outputs=current_mask_overlay,
        queue=False,
    )
    review_tab.select(
        lambda tile: (
            gr.Accordion(open=True)
            if tile is not None
            else gr.Accordion(open=False)
        ),
        inputs=selected_tile_state,
        outputs=editor_accordion,
        queue=False,
    ).then(
        start_review,
        inputs=[
            selected_tile_state,
            threshold,
            compute_device,
            segmentation_method,
            model_weights,
            show_original_in_editor,
            batch_session_state,
            latest_prediction_state,
        ],
        outputs=[
            annotation_editor,
            original_mask_state,
            probability_state,
            treatment_history_state,
            annotation_stats,
            editor_status,
            editor_accordion,
            working_mask_state,
        ],
    ).then(lambda value: value, threshold, editor_threshold).then(
        prepare_roi_selection,
        inputs=selected_tile_state,
        outputs=[roi_selector, roi_points_state, roi_status],
        queue=False,
    )
    for review_button in (open_review,):
        review_button.click(
            open_review_step,
            inputs=selected_tile_state,
            outputs=[workflow, editor_accordion],
        ).then(
            start_review,
            inputs=[
                selected_tile_state,
                threshold,
                compute_device,
                segmentation_method,
                model_weights,
                show_original_in_editor,
                batch_session_state,
                latest_prediction_state,
            ],
            outputs=[
                annotation_editor,
                original_mask_state,
                probability_state,
                treatment_history_state,
                annotation_stats,
                editor_status,
                editor_accordion,
                working_mask_state,
            ],
        ).then(lambda value: value, threshold, editor_threshold).then(
            prepare_roi_selection,
            inputs=selected_tile_state,
            outputs=[roi_selector, roi_points_state, roi_status],
            queue=False,
        )
    reload_editor.click(
        lambda: gr.Accordion(open=True),
        outputs=editor_accordion,
        queue=False,
    ).then(
        start_explicit_new_review_segmentation,
        inputs=[
            selected_tile_state,
            editor_threshold,
            compute_device,
            segmentation_method,
            model_weights,
            show_original_in_editor,
        ],
        outputs=[
            annotation_editor,
            original_mask_state,
            probability_state,
            treatment_history_state,
            annotation_stats,
            editor_status,
            editor_accordion,
            working_mask_state,
        ],
    ).then(
        prepare_roi_selection,
        inputs=selected_tile_state,
        outputs=[roi_selector, roi_points_state, roi_status],
        queue=False,
    )
    prepare_roi.click(
        prepare_roi_selection,
        inputs=selected_tile_state,
        outputs=[roi_selector, roi_points_state, roi_status],
        queue=False,
    )
    roi_selector.select(
        select_roi_corner,
        inputs=[selected_tile_state, roi_points_state],
        outputs=[roi_selector, roi_points_state, roi_status],
        queue=False,
    )
    apply_roi_segmentation.click(
        apply_segmentation_in_roi,
        inputs=[
            annotation_editor,
            selected_tile_state,
            roi_points_state,
            treatment_history_state,
            roi_threshold,
            compute_device,
            roi_segmentation_method,
            model_weights,
            show_original_in_editor,
            working_mask_state,
        ],
        outputs=[
            annotation_editor,
            treatment_history_state,
            annotation_stats,
            editor_status,
            working_mask_state,
            roi_points_state,
            roi_selector,
            roi_status,
        ],
        show_progress="full",
    )
    segmentation_method.change(
        configure_segmentation_method,
        inputs=segmentation_method,
        outputs=[model_weights, model_configuration_status],
        queue=False,
    )
    apply_treatment.click(
        apply_mask_treatment,
        inputs=[
            annotation_editor,
            selected_tile_state,
            treatment_history_state,
            min_component_area,
            fill_holes_control,
            max_hole_area,
            morphology,
            kernel_size,
            show_original_in_editor,
        ],
        outputs=[annotation_editor, treatment_history_state, annotation_stats, editor_status, working_mask_state],
    )
    regenerate.click(
        regenerate_from_threshold,
        inputs=[
            annotation_editor,
            selected_tile_state,
            probability_state,
            treatment_history_state,
            editor_threshold,
            show_original_in_editor,
        ],
        outputs=[annotation_editor, treatment_history_state, annotation_stats, editor_status, working_mask_state],
    )
    restore_mask.click(
        restore_initial_mask,
        inputs=[
            annotation_editor,
            selected_tile_state,
            original_mask_state,
            treatment_history_state,
            show_original_in_editor,
        ],
        outputs=[annotation_editor, treatment_history_state, annotation_stats, editor_status, working_mask_state],
    )
    undo_treatment.click(
        undo_last_treatment,
        inputs=[
            selected_tile_state,
            treatment_history_state,
            show_original_in_editor,
        ],
        outputs=[annotation_editor, treatment_history_state, annotation_stats, editor_status, working_mask_state],
    )
    show_original_in_editor.change(
        toggle_editor_background,
        inputs=[
            show_original_in_editor,
            annotation_editor,
            selected_tile_state,
            working_mask_state,
        ],
        outputs=annotation_editor,
        queue=False,
    )
    expand_editor.click(
        fn=None,
        js="""() => {
            const workspace = document.getElementById('editor-workspace');
            if (!workspace) return;
            const expanded = workspace.classList.toggle('expanded-mask-editor');
            document.body.style.overflow = expanded ? 'hidden' : '';
            const editor = document.getElementById('mask-editor');
            if (!editor) return;
            if (!expanded) {
                editor.style.zoom = '';
                workspace.style.alignItems = '';
                return;
            }
            editor.style.zoom = '1';
            workspace.style.alignItems = 'center';
            requestAnimationFrame(() => {
                const width = Math.max(editor.offsetWidth, 1);
                const height = Math.max(editor.offsetHeight, 1);
                const availableWidth = window.innerWidth - 64;
                const availableHeight = window.innerHeight - 100;
                const scale = Math.max(
                    1,
                    Math.min(availableWidth / width, availableHeight / height)
                );
                editor.style.zoom = String(scale);
            });
        }""",
        queue=False,
    )
    export_button.click(
        create_exports,
        inputs=[
            annotation_editor,
            selected_tile_state,
            source_state,
            category_name,
            polygon_simplification,
        ],
        outputs=[download_mask, download_coco, export_summary],
    )
    save_and_next.click(
        save_review_and_advance,
        inputs=[
            annotation_editor,
            batch_session_state,
            tiles_state,
            collection_state,
            needs_later_review,
            working_mask_state,
        ],
        outputs=[
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            needs_later_review,
            latest_prediction_state,
        ],
    )
    clear_later_review.click(
        clear_later_review_flag,
        inputs=[batch_session_state, tiles_state],
        outputs=[
            batch_session_state,
            needs_later_review,
            editor_status,
            tile_session_summary,
        ],
    )
    skip_as_background.click(
        skip_review_as_background,
        inputs=[batch_session_state, tiles_state, collection_state],
        outputs=[
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            needs_later_review,
            latest_prediction_state,
        ],
    )
    segment_skip_background.click(
        skip_review_as_background,
        inputs=[batch_session_state, tiles_state, collection_state],
        outputs=[
            batch_session_state,
            selected_tile_state,
            tile_for_analysis,
            source_state,
            batch_progress,
            workflow,
            needs_later_review,
            latest_prediction_state,
        ],
    )
    final_export_button.click(
        create_batch_exports,
        inputs=[
            batch_session_state,
            tiles_state,
            collection_state,
            final_category_name,
            final_simplification,
        ],
        outputs=[
            download_masks_zip,
            download_batch_coco,
            download_tiles_zip,
            final_export_summary,
        ],
    )

    gr.Markdown(
        "A saída oferece apoio à análise visual e requer validação antes de qualquer decisão agronômica."
    )


if __name__ == "__main__":
    demo.launch(inbrowser=False, css=CSS)
