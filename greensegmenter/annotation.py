import json
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from greensegmenter.tiling import Tile


MASK_COLOR = np.array([255, 0, 255, 180], dtype=np.uint8)
MASK_BRUSH_COLOR = ("#ff00ff", 180 / 255)


def editor_value(image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Monta o editor com arrays, evitando perda de transparência no canvas web."""
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[mask.astype(bool)] = MASK_COLOR
    background = image.astype(np.uint8).copy()
    # Background e composite permanecem limpos. A anotação existe somente na
    # layer editável, evitando uma segunda máscara fixa após apagar pixels.
    composite = np.concatenate(
        [background, np.full((*mask.shape, 1), 255, dtype=np.uint8)], axis=2
    )
    return {
        "background": background,
        "layers": [layer],
        "composite": composite,
    }


def mask_from_editor(value: dict[str, Any] | None) -> np.ndarray:
    """Une pixels não transparentes de todas as camadas editáveis."""
    if not value or not value.get("layers"):
        raise ValueError("O editor não contém uma camada de máscara.")
    result: np.ndarray | None = None
    for raw_layer in value["layers"]:
        layer = np.asarray(raw_layer.convert("RGBA") if isinstance(raw_layer, Image.Image) else raw_layer)
        alpha = layer[..., 3] > 0 if layer.shape[-1] == 4 else layer[..., :3].max(axis=2) > 0
        result = alpha if result is None else np.logical_or(result, alpha)
    assert result is not None
    return result


def clean_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    fill_holes: bool = False,
    max_hole_area: int = 0,
    morphology: str = "Nenhuma",
    kernel_size: int = 3,
) -> np.ndarray:
    """Limpa componentes, buracos e bordas usando operações binárias determinísticas."""
    binary = mask.astype(np.uint8)
    if min_component_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        kept = np.zeros_like(binary)
        for label in range(1, count):
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_component_area:
                kept[labels == label] = 1
        binary = kept

    if fill_holes:
        inverse = 1 - binary
        count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
        border_labels = set(np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if label not in border_labels and (max_hole_area == 0 or area <= max_hole_area):
                binary[labels == label] = 1

    if morphology != "Nenhuma":
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("O kernel morfológico deve ser um número ímpar positivo.")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        operation = cv2.MORPH_OPEN if morphology == "Abertura" else cv2.MORPH_CLOSE
        binary = cv2.morphologyEx(binary, operation, kernel)
    return binary.astype(bool)


def merge_mask_region(
    base_mask: np.ndarray,
    region_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    """Adiciona uma predição binária somente dentro da ROI (x1, y1, x2, y2)."""
    result = base_mask.astype(bool).copy()
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > result.shape[1] or y2 > result.shape[0]:
        raise ValueError("A área selecionada ultrapassa os limites do tile.")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("A área selecionada precisa possuir largura e altura.")
    if region_mask.shape != (y2 - y1, x2 - x1):
        raise ValueError("A segmentação regional não corresponde à área selecionada.")
    result[y1:y2, x1:x2] |= region_mask.astype(bool)
    return result


def mask_statistics(mask: np.ndarray) -> dict[str, int | float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return {
        "pixels_anotados": int(mask.sum()),
        "área_percentual": round(float(mask.mean() * 100), 3),
        "componentes": max(0, count - 1),
    }


def polygons_from_mask(
    mask: np.ndarray,
    tile: Tile,
    simplification: float = 1.5,
    min_polygon_area: float = 10.0,
) -> list[list[float]]:
    """Extrai contornos externos COCO e converte coordenadas locais para globais."""
    valid = mask[: tile.valid_height, : tile.valid_width].astype(np.uint8)
    contours, _ = cv2.findContours(valid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_polygon_area:
            continue
        approximated = cv2.approxPolyDP(contour, float(simplification), True)
        if len(approximated) < 3:
            continue
        points = approximated.reshape(-1, 2).astype(float)
        points[:, 0] += tile.x
        points[:, 1] += tile.y
        polygons.append(points.reshape(-1).tolist())
    return polygons


def export_annotation(
    mask: np.ndarray,
    tile: Tile,
    source_shape: tuple[int, ...],
    category_name: str,
    simplification: float,
) -> tuple[str, str, dict[str, Any]]:
    """Exporta a máscara do tile em PNG e seus contornos externos em COCO JSON."""
    export_dir = Path(tempfile.mkdtemp(prefix="greensegmenter_annotation_"))
    stem = f"tile_x{tile.x}_y{tile.y}"
    png_path = export_dir / f"{stem}_mask.png"
    json_path = export_dir / f"{stem}_coco.json"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(png_path)

    polygons = polygons_from_mask(mask, tile, simplification)
    annotations = []
    for annotation_id, polygon in enumerate(polygons, start=1):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        local = points - np.array([tile.x, tile.y], dtype=np.float32)
        area = float(cv2.contourArea(local.reshape(-1, 1, 2)))
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        annotations.append({
            "id": annotation_id,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [polygon],
            "area": round(area, 3),
            "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
            "iscrowd": 0,
        })
    coco = {
        "info": {"description": "Pré-anotação ExG revisada manualmente"},
        "images": [{
            "id": 1,
            "file_name": "imagem_original",
            "width": int(source_shape[1]),
            "height": int(source_shape[0]),
        }],
        "categories": [{"id": 1, "name": category_name or "vegetacao", "supercategory": "agricultura"}],
        "annotations": annotations,
        "greensegmenter": {
            "tile": {"x": tile.x, "y": tile.y, "width": tile.valid_width, "height": tile.valid_height},
            "mask_source": "ExG + edição manual",
            "polygon_representation": "contornos externos; PNG é a fonte de verdade",
        },
    }
    json_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {**mask_statistics(mask), "polígonos_exportados": len(polygons), "formato": "COCO polygon"}
    return str(png_path), str(json_path), summary


def export_annotation_batch(
    saved_masks: dict[int, np.ndarray],
    tiles: list[Tile],
    sources: tuple[int, ...] | list[dict[str, Any]],
    category_name: str,
    simplification: float,
) -> tuple[str, str, dict[str, Any]]:
    """Exporta máscaras revisadas em ZIP e um COCO global para a imagem original."""
    if not saved_masks:
        raise ValueError("Nenhuma máscara revisada foi salva.")
    export_dir = Path(tempfile.mkdtemp(prefix="greensegmenter_batch_"))
    masks_dir = export_dir / "masks"
    masks_dir.mkdir()
    annotations: list[dict[str, Any]] = []
    if isinstance(sources, tuple):
        images = [{
            "id": 1, "file_name": "imagem_original",
            "width": int(sources[1]), "height": int(sources[0]),
        }]
    else:
        images = [
            {
                "id": index + 1,
                "file_name": item["name"],
                "width": int(item.get("shape", item["image"].shape)[1]),
                "height": int(item.get("shape", item["image"].shape)[0]),
            }
            for index, item in enumerate(sources)
        ]
    annotation_id = 1
    for tile_index in sorted(saved_masks):
        tile = tiles[tile_index]
        mask = saved_masks[tile_index].astype(bool)
        safe_source = Path(tile.source_name).stem.replace(" ", "_")
        name = f"{safe_source}_tile_{tile_index + 1:04d}_x{tile.x}_y{tile.y}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(masks_dir / name)
        for polygon in polygons_from_mask(mask, tile, simplification):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            local = points - np.array([tile.x, tile.y], dtype=np.float32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            annotations.append({
                "id": annotation_id,
                "image_id": tile.source_index + 1,
                "category_id": 1,
                "segmentation": [polygon],
                "area": round(float(cv2.contourArea(local.reshape(-1, 1, 2))), 3),
                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                "iscrowd": 0,
                "tile_index": tile_index,
            })
            annotation_id += 1
    coco = {
        "info": {"description": "Anotações de vegetação revisadas no GreenSegmenter"},
        "images": images,
        "categories": [{"id": 1, "name": category_name or "vegetacao", "supercategory": "agricultura"}],
        "annotations": annotations,
        "greensegmenter": {"tiles_revisados": len(saved_masks), "mask_source": "segmentação binária + edição manual"},
    }
    json_path = export_dir / "annotations_coco.json"
    json_path.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = export_dir / "masks_png.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(masks_dir.iterdir()):
            archive.write(path, arcname=path.name)
    return str(zip_path), str(json_path), {
        "tiles_revisados": len(saved_masks),
        "máscaras": len(saved_masks),
        "polígonos": len(annotations),
    }
