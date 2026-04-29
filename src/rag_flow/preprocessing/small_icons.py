from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.runtime import get_torch_device, require_trusted_remote_code_model


IGNORE_TYPES = {
    "header",
    "footer",
    "image",
    "page_number",
    "aside_text",
    "page_footnote",
    "equation",
    "seal",
    "chart",
}

TEXT_FIELD_MAP = {
    "text": ["text"],
    "list": ["list_items"],
    "table": ["table_caption", "table_footnote", "table_body"],
    "image": ["image_caption", "image_footnote"],
    "code": ["code_caption", "code_footnote"],
}

TEXT_FIELD_KEYS = {
    "text",
    "list_items",
    "table_caption",
    "table_footnote",
    "table_body",
    "image_caption",
    "image_footnote",
    "code_caption",
    "code_footnote",
}

METADATA_KEYS = {
    "type",
    "bbox",
    "page_idx",
    "text_level",
    "img_path",
    "sub_type",
}

CHECKED_FIELDS_KEY = "vlm-small-icon-checked-fields"
PATCHED_FIELDS_KEY = "vlm-small-icon-patched-fields"


@dataclass(frozen=True)
class IconPatchArtifacts:
    artifact_dir: Path
    content_json: Path
    origin_pdf: Path
    output_json: Path


@dataclass
class IconPatchStats:
    blocks_seen: int = 0
    fields_seen: int = 0
    requests_submitted: int = 0
    checked_count: int = 0
    patched_count: int = 0
    no_missing_count: int = 0
    skipped_ignored_blocks: int = 0
    skipped_no_bbox: int = 0
    skipped_no_fields: int = 0
    skipped_empty_fields: int = 0
    table_continuation_blocks: int = 0
    table_continuation_crops: int = 0
    windows_processed: int = 0
    batches_processed: int = 0
    checkpoints_written: int = 0


def _single_candidate(candidates: list[Path], *, label: str, artifact_dir: Path) -> Path:
    if not candidates:
        raise FileNotFoundError(f"Cannot find {label} under MinerU artifact dir: {artifact_dir}")
    if len(candidates) > 1:
        names = ", ".join(str(path.name) for path in candidates)
        raise ValueError(f"Found multiple {label} files under {artifact_dir}: {names}")
    return candidates[0]


def _content_stem(content_json: Path, artifact_dir: Path) -> str:
    if content_json.name.endswith("_content_list.json"):
        return content_json.name[: -len("_content_list.json")]
    if content_json.name == "content_list.json":
        return artifact_dir.name
    return content_json.stem


def resolve_icon_patch_artifacts(
    artifact_dir: str | Path,
    *,
    content_json: str | Path | None = None,
    origin_pdf: str | Path | None = None,
    output_json: str | Path | None = None,
) -> IconPatchArtifacts:
    resolved_dir = Path(artifact_dir).expanduser()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"MinerU artifact dir does not exist: {resolved_dir}")

    if content_json:
        resolved_content = Path(content_json).expanduser()
    else:
        content_candidates = sorted(
            path
            for path in resolved_dir.glob("*_content_list.json")
            if "small-icon" not in path.name and "caption" not in path.name
        )
        if not content_candidates:
            content_candidates = sorted(path for path in resolved_dir.glob("content_list.json"))
        resolved_content = _single_candidate(
            content_candidates,
            label="MinerU content_list JSON",
            artifact_dir=resolved_dir,
        )

    stem = _content_stem(resolved_content, resolved_dir)
    if origin_pdf:
        resolved_pdf = Path(origin_pdf).expanduser()
    else:
        exact_origin = resolved_dir / f"{stem}_origin.pdf"
        if exact_origin.exists():
            resolved_pdf = exact_origin
        else:
            resolved_pdf = _single_candidate(
                sorted(resolved_dir.glob("*_origin.pdf")),
                label="MinerU origin PDF",
                artifact_dir=resolved_dir,
            )

    resolved_output = Path(output_json).expanduser() if output_json else resolved_dir / (
        f"{stem}_content_list_PATCHED.json"
    )
    return IconPatchArtifacts(
        artifact_dir=resolved_dir,
        content_json=resolved_content,
        origin_pdf=resolved_pdf,
        output_json=resolved_output,
    )


def resolve_icon_patch_batch(
    artifact_dir: str | Path,
    *,
    recursive: bool = True,
) -> list[IconPatchArtifacts]:
    root = Path(artifact_dir).expanduser()
    try:
        return [resolve_icon_patch_artifacts(root)]
    except (FileNotFoundError, ValueError):
        pass

    candidates = root.rglob("*_content_list.json") if recursive else root.glob("*_content_list.json")
    artifacts = []
    seen_dirs: set[Path] = set()
    for content_json in sorted(candidates):
        if "small-icon" in content_json.name or "caption" in content_json.name:
            continue
        artifact_parent = content_json.parent
        if artifact_parent in seen_dirs:
            continue
        try:
            artifacts.append(resolve_icon_patch_artifacts(artifact_parent, content_json=content_json))
        except FileNotFoundError:
            continue
        seen_dirs.add(artifact_parent)

    if not artifacts:
        raise FileNotFoundError(f"Cannot find MinerU artifact folders under: {root}")
    return artifacts


def checkpoint_path_for(output_json: str | Path) -> Path:
    output = Path(output_json)
    return output.with_name(f"{output.stem}.checkpoint{output.suffix}")


def _checked_fields(block: dict[str, Any]) -> set[str]:
    value = block.get(CHECKED_FIELDS_KEY, [])
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _mark_checked(block: dict[str, Any], key: str) -> None:
    fields = sorted({*_checked_fields(block), key})
    block[CHECKED_FIELDS_KEY] = fields


def _mark_patched(block: dict[str, Any], key: str) -> None:
    value = block.get(PATCHED_FIELDS_KEY, [])
    fields = set(str(item) for item in value) if isinstance(value, list) else set()
    fields.add(key)
    block[PATCHED_FIELDS_KEY] = sorted(fields)
    block["vlm-small-icon-patched"] = True


def _patch_field_keys(block: dict[str, Any]) -> list[str]:
    block_type = block.get("type")
    if block_type in IGNORE_TYPES or _is_table_continuation_block(block):
        return []

    keys: list[str] = []
    for key in TEXT_FIELD_MAP.get(str(block_type), []):
        if key in block:
            keys.append(key)
    for key, value in block.items():
        if key in keys or key in METADATA_KEYS or key.startswith("vlm-small-icon-"):
            continue
        if key in TEXT_FIELD_KEYS or isinstance(value, str) or _is_text_list(value):
            keys.append(key)
    return keys


def _is_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_table_continuation_block(block: dict[str, Any]) -> bool:
    if block.get("type") != "table":
        return False
    return (
        not _join(block.get("table_body", "")).strip()
        and not _join(block.get("table_caption", "")).strip()
        and not _join(block.get("table_footnote", "")).strip()
        and not str(block.get("img_path", "")).strip()
    )


def _is_table_continuation_for_master(master: dict[str, Any], block: dict[str, Any]) -> bool:
    if not _is_table_continuation_block(block):
        return False
    if "bbox" not in master or "bbox" not in block:
        return False

    master_page = int(master.get("page_idx", 0))
    block_page = int(block.get("page_idx", 0))
    if block_page < master_page:
        return False
    if block_page == master_page:
        return block["bbox"][1] >= master["bbox"][3]
    return block["bbox"][1] <= 180


def build_table_continuation_map(content_data: list[dict[str, Any]]) -> dict[int, list[int]]:
    continuations: dict[int, list[int]] = {}
    current_master_idx: int | None = None

    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or block.get("type") != "table":
            continue

        if _join(block.get("table_body", "")).strip():
            current_master_idx = idx
            continuations.setdefault(idx, [])
            continue

        if current_master_idx is None:
            continue
        master = content_data[current_master_idx]
        if _is_table_continuation_for_master(master, block):
            continuations.setdefault(current_master_idx, []).append(idx)

    return {master_idx: indices for master_idx, indices in continuations.items() if indices}


def _table_continuation_indices(table_continuations: dict[int, list[int]]) -> set[int]:
    return {idx for indices in table_continuations.values() for idx in indices}


def _window_visual_page_end(
    *,
    content_data: list[dict[str, Any]],
    table_continuations: dict[int, list[int]],
    page_start: int,
    page_end: int,
    max_page_idx: int,
) -> int:
    visual_page_end = page_end
    for master_idx, continuation_indices in table_continuations.items():
        master = content_data[master_idx]
        master_page = int(master.get("page_idx", 0))
        if page_start <= master_page <= page_end:
            for continuation_idx in continuation_indices:
                continuation = content_data[continuation_idx]
                visual_page_end = max(visual_page_end, int(continuation.get("page_idx", 0)))
    return min(visual_page_end, max_page_idx)


def _write_json(path: Path, content_data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)


def _strip_checkpoint_fields(content_data: list[dict[str, Any]]) -> None:
    for block in content_data:
        if isinstance(block, dict):
            block.pop(CHECKED_FIELDS_KEY, None)


def _print_stats(stats: IconPatchStats, output_json: Path) -> None:
    print("Icon patching stats:")
    print(f"  blocks seen: {stats.blocks_seen}")
    print(f"  fields seen: {stats.fields_seen}")
    print(f"  requests submitted: {stats.requests_submitted}")
    print(f"  checked: {stats.checked_count}")
    print(f"  patched: {stats.patched_count}")
    print(f"  no missing: {stats.no_missing_count}")
    print(f"  skipped ignored blocks: {stats.skipped_ignored_blocks}")
    print(f"  skipped without bbox: {stats.skipped_no_bbox}")
    print(f"  skipped without text fields: {stats.skipped_no_fields}")
    print(f"  skipped empty fields: {stats.skipped_empty_fields}")
    print(f"  table continuation blocks: {stats.table_continuation_blocks}")
    print(f"  table continuation crops: {stats.table_continuation_crops}")
    print(f"  page windows: {stats.windows_processed}")
    print(f"  VLM batches: {stats.batches_processed}")
    print(f"  checkpoints written: {stats.checkpoints_written}")
    print(f"  output: {output_json}")


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def build_icon_patch_prompt(*, original_text: str, field_key: str) -> str:
    if field_key == "table_body" and "<table" in original_text.lower():
        return (
            "Please inspect the image and determine whether any small icons "
            "(for example plus sign, wrench, gear, arrow, save icon) are embedded "
            "in the table but missing from the extracted HTML. The image may contain "
            "one or more vertically stacked crops from the same table across pages.\n"
            f"Here is the extracted HTML table:\n{original_text}\n\n"
            "If icons are missing, insert `[Icon: shape/name]` into the exact table cells "
            "where they belong. Preserve the full HTML table structure, tags, rows, columns, "
            "and all existing text exactly. Do not translate, summarize, rewrite, fix OCR, "
            "normalize punctuation, delete text, or add explanations. Only insert `[Icon: ...]` "
            "tokens where visual icons are present but absent from the HTML. Do not create "
            "a second table for continuation-page crops. Return only the complete modified "
            "HTML table. If no icons are missing, return exactly `No missing`."
        )

    return (
        "Please inspect the image and determine whether any small icons "
        "(for example plus sign, wrench, gear, arrow, save icon) are embedded "
        "in or around the text.\n"
        f'Here is the extracted text:\n"{original_text}"\n\n'
        "If icons are missing from the text, insert `[Icon: shape/name]` at the exact "
        "corresponding position. Preserve every original character, word, line break, "
        "number, and punctuation mark exactly. Do not translate, summarize, rewrite, "
        "fix OCR, delete text, or add explanations. Only insert `[Icon: ...]` tokens. "
        "Return only the modified complete text. If no icons are missing, return exactly "
        "`No missing`."
    )


def is_no_missing_response(text: str) -> bool:
    normalized = re.sub(r"[\s`\"'.:;!,，。；：！]+", " ", text.strip().lower()).strip()
    return normalized in {"no missing", "no missing icon", "no missing icons"} or normalized.startswith(
        "no missing "
    )


def should_apply_icon_patch(*, original_text: str, patched_text: str, field_key: str) -> bool:
    if is_no_missing_response(patched_text):
        return False
    return bool(re.search(r"\[icon\s*:", patched_text, flags=re.IGNORECASE))


def crop_image_from_block(block: dict[str, Any], pdf_images: list[Any], *, page_offset: int = 0) -> Any | None:
    page_idx = int(block.get("page_idx", 0))
    local_page_idx = page_idx - page_offset
    if local_page_idx < 0 or local_page_idx >= len(pdf_images) or "bbox" not in block:
        return None

    image = pdf_images[local_page_idx]
    norm_x0, norm_y0, norm_x1, norm_y1 = block["bbox"]
    rx0 = (norm_x0 / 1000.0) * image.width
    ry0 = (norm_y0 / 1000.0) * image.height
    rx1 = (norm_x1 / 1000.0) * image.width
    ry1 = (norm_y1 / 1000.0) * image.height
    rx0, ry0 = max(0, rx0), max(0, ry0)
    rx1, ry1 = min(image.width, rx1), min(image.height, ry1)
    return image.crop((rx0, ry0, rx1, ry1))


def concat_images_vertically(images: list[Any]) -> Any | None:
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    from PIL import Image

    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    combined = Image.new("RGB", (width, height), (255, 255, 255))
    y_offset = 0
    for image in images:
        combined.paste(image, (0, y_offset))
        y_offset += image.height
    return combined


def build_table_footnote_crop(
    *,
    content_data: list[dict[str, Any]],
    pdf_images: list[Any],
    block_idx: int,
    page_offset: int = 0,
) -> Any | None:
    block = content_data[block_idx]
    last_idx = block_idx
    lookahead_idx = block_idx + 1

    while lookahead_idx < len(content_data):
        next_block = content_data[lookahead_idx]
        next_type = next_block.get("type")
        if next_type in IGNORE_TYPES:
            lookahead_idx += 1
            continue
        if next_type != block.get("type"):
            break
        next_text = _join(next_block.get("table_footnote", "")).strip()
        if next_text:
            break
        last_idx = lookahead_idx
        lookahead_idx += 1

    last_block = content_data[last_idx]
    page_idx = int(last_block.get("page_idx", 0))
    local_page_idx = page_idx - page_offset
    if local_page_idx < 0 or local_page_idx >= len(pdf_images) or "bbox" not in last_block:
        return None

    image = pdf_images[local_page_idx]
    y0_norm = last_block["bbox"][3]
    y1_norm = 1000

    for idx in range(last_idx + 1, len(content_data)):
        next_block = content_data[idx]
        if next_block.get("page_idx") != page_idx:
            break
        if next_block.get("type") not in {"header", "footer", "page_number"} and "bbox" in next_block:
            next_y0 = next_block["bbox"][1]
            if next_y0 > y0_norm:
                y1_norm = next_y0
                break

    rx0 = 0
    ry0 = (y0_norm / 1000.0) * image.height
    rx1 = image.width
    ry1 = (y1_norm / 1000.0) * image.height
    min_height = int(image.width / 190.0) + 1
    if (ry1 - ry0) < min_height:
        ry1 = ry0 + min_height
        if ry1 > image.height:
            ry1 = image.height
            ry0 = max(0, image.height - min_height)
    return image.crop((rx0, ry0, rx1, ry1))


def build_table_body_crop(
    *,
    content_data: list[dict[str, Any]],
    pdf_images: list[Any],
    block_idx: int,
    continuation_indices: list[int],
    page_offset: int = 0,
) -> Any | None:
    crops = []
    block = content_data[block_idx]
    block_crop = crop_image_from_block(block, pdf_images, page_offset=page_offset)
    if block_crop is not None:
        crops.append(block_crop)

    for continuation_idx in continuation_indices:
        continuation = content_data[continuation_idx]
        continuation_crop = crop_image_from_block(continuation, pdf_images, page_offset=page_offset)
        if continuation_crop is not None:
            crops.append(continuation_crop)

    return concat_images_vertically(crops)


def add_small_icon_text(
    *,
    input_json: str | Path,
    output_json: str | Path,
    pdf_path: str | Path,
    model_name: str,
    dpi: int = 200,
    batch_size: int = 6,
    max_new_tokens: int = 5000,
    model_revision: str = "",
    trusted_remote_code_models: tuple[str, ...] = ("Qwen/Qwen3.5-9B",),
    page_window_size: int = 200,
    checkpoint_interval: int = 1,
    checkpoint_json: str | Path | None = None,
    resume: bool = True,
) -> None:
    import torch
    from modelscope import snapshot_download
    from pdf2image import convert_from_path
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = get_torch_device(feature="Small-icon VLM preprocessing")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    require_trusted_remote_code_model(model_name, allowed_models=trusted_remote_code_models)
    snapshot_kwargs = {"revision": model_revision} if model_revision else {}
    model_dir = snapshot_download(model_name, **snapshot_kwargs)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    input_path = Path(input_json)
    output_path = Path(output_json)
    checkpoint_path = Path(checkpoint_json) if checkpoint_json else checkpoint_path_for(output_path)
    if resume and checkpoint_path.exists():
        print(f"Resuming icon patching from checkpoint: {checkpoint_path}")
        source_json = checkpoint_path
    else:
        source_json = input_path

    with source_json.open("r", encoding="utf-8") as f:
        content_data: list[dict[str, Any]] = json.load(f)

    stats = IconPatchStats()
    max_page_idx = max(
        (int(block.get("page_idx", 0)) for block in content_data if isinstance(block, dict) and "page_idx" in block),
        default=-1,
    )
    table_continuations = build_table_continuation_map(content_data)
    table_continuation_indices = _table_continuation_indices(table_continuations)
    stats.table_continuation_blocks = len(table_continuation_indices)

    def write_checkpoint() -> None:
        _write_json(checkpoint_path, content_data)
        stats.checkpoints_written += 1

    def process_batch(requests: list[dict[str, Any]]) -> None:
        if not requests:
            return
        stats.requests_submitted += len(requests)
        stats.batches_processed += 1

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": req["image"]}, {"type": "text", "text": req["prompt"]}]}]
            for req in requests
        ]
        texts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for message in messages
        ]

        image_inputs_list = []
        video_inputs_list = []
        for message in messages:
            image_inputs, video_inputs = process_vision_info(message)
            image_inputs_list.append(image_inputs)
            video_inputs_list.append(video_inputs)

        flat_images = [item for sublist in image_inputs_list if sublist for item in sublist]
        flat_videos = [item for sublist in video_inputs_list if sublist for item in sublist]
        inputs = processor(
            text=texts,
            images=flat_images if flat_images else None,
            videos=flat_videos if flat_videos else None,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for req, raw_output in zip(requests, outputs):
            output = raw_output.strip()
            if "</think>" in output:
                output = output.split("</think>")[-1].strip()

            idx = req["idx"]
            key = req["key"]
            block = content_data[idx]
            _mark_checked(block, key)
            stats.checked_count += 1
            if is_no_missing_response(output):
                stats.no_missing_count += 1
                continue

            if should_apply_icon_patch(
                original_text=req["original_text"],
                patched_text=output,
                field_key=req["key"],
            ):
                content_data[idx][key] = output.split("\n") if req["is_list"] else output
                _mark_patched(content_data[idx], key)
                stats.patched_count += 1

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if checkpoint_interval > 0 and stats.batches_processed % checkpoint_interval == 0:
            write_checkpoint()

    try:
        if max_page_idx < 0:
            _strip_checkpoint_fields(content_data)
            _write_json(output_path, content_data)
            _print_stats(stats, output_path)
            return

        for page_start in range(0, max_page_idx + 1, max(1, page_window_size)):
            scan_page_end = min(max_page_idx, page_start + max(1, page_window_size) - 1)
            visual_page_end = _window_visual_page_end(
                content_data=content_data,
                table_continuations=table_continuations,
                page_start=page_start,
                page_end=scan_page_end,
                max_page_idx=max_page_idx,
            )
            pdf_images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                first_page=page_start + 1,
                last_page=visual_page_end + 1,
            )
            stats.windows_processed += 1

            batch: list[dict[str, Any]] = []
            for idx in tqdm(range(len(content_data)), desc=f"Scanning pages {page_start + 1}-{scan_page_end + 1}"):
                block = content_data[idx]
                if not isinstance(block, dict):
                    continue
                page_idx = int(block.get("page_idx", 0))
                if page_idx < page_start or page_idx > scan_page_end:
                    continue
                stats.blocks_seen += 1
                if block.get("type") in IGNORE_TYPES or idx in table_continuation_indices:
                    stats.skipped_ignored_blocks += 1
                    continue
                if "bbox" not in block:
                    stats.skipped_no_bbox += 1
                    continue

                field_keys = _patch_field_keys(block)
                if not field_keys:
                    stats.skipped_no_fields += 1
                    continue

                checked_fields = _checked_fields(block)
                for key in field_keys:
                    if key in checked_fields:
                        continue
                    stats.fields_seen += 1
                    value = block.get(key, "")
                    is_list = isinstance(value, list)
                    original_text = _join(value).strip()
                    if not original_text:
                        _mark_checked(block, key)
                        stats.skipped_empty_fields += 1
                        continue

                    if key == "table_body":
                        continuation_indices = table_continuations.get(idx, [])
                        final_image = build_table_body_crop(
                            content_data=content_data,
                            pdf_images=pdf_images,
                            block_idx=idx,
                            continuation_indices=continuation_indices,
                            page_offset=page_start,
                        )
                        stats.table_continuation_crops += len(continuation_indices)
                    elif key == "table_footnote":
                        final_image = build_table_footnote_crop(
                            content_data=content_data,
                            pdf_images=pdf_images,
                            block_idx=idx,
                            page_offset=page_start,
                        )
                    else:
                        final_image = crop_image_from_block(block, pdf_images, page_offset=page_start)
                    if final_image is None:
                        continue

                    prompt = build_icon_patch_prompt(original_text=original_text, field_key=key)
                    batch.append(
                        {
                            "idx": idx,
                            "key": key,
                            "block_type": block.get("type"),
                            "page_idx": page_idx,
                            "original_text": original_text,
                            "is_list": is_list,
                            "image": final_image,
                            "prompt": prompt,
                        }
                    )
                    if len(batch) >= batch_size:
                        process_batch(batch)
                        batch = []

            if batch:
                process_batch(batch)

            del pdf_images
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            write_checkpoint()

    except Exception:
        write_checkpoint()
        print(f"Icon patching checkpoint saved before failure: {checkpoint_path}")
        raise
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _strip_checkpoint_fields(content_data)
    _write_json(output_path, content_data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    _print_stats(stats, output_path)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Use a VLM to patch small icon text missing from MinerU JSON.")
    parser.add_argument("--artifact-dir", help="MinerU output folder containing *_content_list.json and *_origin.pdf.")
    parser.add_argument("--input", default=None, help="Input MinerU content_list JSON.")
    parser.add_argument("--output", default=None, help="Output patched content_list JSON.")
    parser.add_argument("--pdf", default=None, help="PDF used for bbox crops. Defaults to *_origin.pdf in --artifact-dir.")
    parser.add_argument("--model", default=config.models.vlm_model)
    parser.add_argument("--model-revision", default=config.models.vlm_model_revision)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--page-window-size", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved inputs without loading the VLM.")
    args = parser.parse_args(argv)

    if args.artifact_dir:
        if args.input or args.pdf or args.output:
            artifacts_list = [
                resolve_icon_patch_artifacts(
                    args.artifact_dir,
                    content_json=args.input,
                    origin_pdf=args.pdf,
                    output_json=args.output,
                )
            ]
        else:
            artifacts_list = resolve_icon_patch_batch(args.artifact_dir, recursive=not args.no_recursive)
    else:
        artifacts_list = [
            IconPatchArtifacts(
                artifact_dir=config.paths.base_dir,
                content_json=Path(args.input).expanduser() if args.input else config.paths.content_json,
                output_json=Path(args.output).expanduser() if args.output else config.paths.patched_json,
                origin_pdf=Path(args.pdf).expanduser() if args.pdf else config.paths.source_pdf,
            )
        ]

    if len(artifacts_list) > 1 and args.checkpoint_json:
        parser.error("--checkpoint-json can only be used with a single patching job.")

    if args.dry_run:
        print(f"Icon patching jobs: {len(artifacts_list)}")
        for artifacts in artifacts_list:
            print("Icon patching inputs:")
            print(f"  artifact_dir: {artifacts.artifact_dir}")
            print(f"  input_json: {artifacts.content_json}")
            print(f"  pdf: {artifacts.origin_pdf}")
            print(f"  output_json: {artifacts.output_json}")
            print(f"  checkpoint_json: {args.checkpoint_json or checkpoint_path_for(artifacts.output_json)}")
            print(f"  page_window_size: {args.page_window_size}")
            print(f"  batch_size: {args.batch_size}")
        return

    for job_idx, artifacts in enumerate(artifacts_list, start=1):
        print(f"Icon patching job {job_idx}/{len(artifacts_list)}: {artifacts.artifact_dir}")
        add_small_icon_text(
            input_json=artifacts.content_json,
            output_json=artifacts.output_json,
            pdf_path=artifacts.origin_pdf,
            model_name=args.model,
            dpi=args.dpi,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            model_revision=args.model_revision,
            trusted_remote_code_models=config.models.trusted_remote_code_models,
            page_window_size=args.page_window_size,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_json=args.checkpoint_json,
            resume=not args.no_resume,
        )


if __name__ == "__main__":
    main()
