from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
from rag_flow.preprocessing.small_icons import resolve_icon_patch_artifacts, resolve_icon_patch_batch
from rag_flow.runtime import get_torch_device, require_trusted_remote_code_model


TEXT_KEYS = [
    "text",
    "list_items",
    "table_caption",
    "table_footnote",
    "table_body",
    "image_caption",
    "image_footnote",
]

INLINE_ICON_SKIP_KEYS = {
    "vlm-small-icon-inline-icon",
    "vlm-small-icon-inline-candidate",
}


@dataclass(frozen=True)
class ImageDescriptionArtifacts:
    artifact_dir: Path
    base_dir: Path
    input_json: Path
    output_json: Path


@dataclass
class ImageDescriptionStats:
    images_seen: int = 0
    caption_candidates: int = 0
    requests_submitted: int = 0
    captioned_count: int = 0
    skipped_inline_icons: int = 0
    skipped_without_img_path: int = 0
    skipped_existing: int = 0
    missing_image_files: int = 0
    failed_image_reads: int = 0
    batches_processed: int = 0
    checkpoints_written: int = 0


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def captioned_json_path_for(input_json: str | Path) -> Path:
    path = Path(input_json)
    name = path.name
    for suffix in ("_content_list_PATCHED.json", "_content_list.json"):
        if name.endswith(suffix):
            prefix = name[: -len(suffix)]
            return path.with_name(f"{prefix}_content_list_PATCHED_CAPTIONED.json")
    if name.endswith("_content_list_PATCHED_CAPTIONED.json"):
        return path
    return path.with_name(f"{path.stem}_CAPTIONED.json")


def checkpoint_path_for(output_json: str | Path) -> Path:
    output = Path(output_json)
    return output.with_name(f"{output.stem}.checkpoint{output.suffix}")


def _write_json(path: Path, content_data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)


def resolve_image_description_artifacts(
    artifact_dir: str | Path,
    *,
    output_json: str | Path | None = None,
) -> ImageDescriptionArtifacts:
    patched_artifacts = resolve_icon_patch_artifacts(artifact_dir)
    resolved_input = patched_artifacts.output_json
    resolved_output = Path(output_json).expanduser() if output_json else captioned_json_path_for(resolved_input)
    return ImageDescriptionArtifacts(
        artifact_dir=patched_artifacts.artifact_dir,
        base_dir=patched_artifacts.artifact_dir,
        input_json=resolved_input,
        output_json=resolved_output,
    )


def resolve_image_description_batch(
    artifact_dir: str | Path,
    *,
    recursive: bool = True,
) -> list[ImageDescriptionArtifacts]:
    return [
        ImageDescriptionArtifacts(
            artifact_dir=artifacts.artifact_dir,
            base_dir=artifacts.artifact_dir,
            input_json=artifacts.output_json,
            output_json=captioned_json_path_for(artifacts.output_json),
        )
        for artifacts in resolve_icon_patch_batch(artifact_dir, recursive=recursive)
    ]


def build_page_text_map(content_data: list[dict[str, Any]]) -> dict[int, list[str]]:
    page_text_map: dict[int, list[str]] = defaultdict(list)
    for block in content_data:
        page_idx = int(block.get("page_idx", 0))
        block_texts = []
        for key in TEXT_KEYS:
            text = _join(block.get(key, "")).strip()
            if text:
                block_texts.append(text)
        if block_texts:
            page_text_map[page_idx].append("\n".join(block_texts))
    return page_text_map


def _page_context_segment(page_text_map: dict[int, list[str]], page_idx: int) -> str:
    if page_idx not in page_text_map:
        return ""
    return f"\n--- [Text from Page {page_idx}] ---\n" + "\n\n".join(page_text_map[page_idx])


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...[truncated]"
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)].rstrip() + marker


def get_three_page_context(
    page_text_map: dict[int, list[str]],
    target_page_idx: int,
    *,
    max_context_chars: int = 12000,
) -> str:
    segments = {
        page_idx: _page_context_segment(page_text_map, page_idx)
        for page_idx in [target_page_idx - 1, target_page_idx, target_page_idx + 1]
    }
    segments = {page_idx: text for page_idx, text in segments.items() if text}
    full_context = "\n".join(segments[page_idx] for page_idx in sorted(segments))
    if max_context_chars <= 0 or len(full_context) <= max_context_chars:
        return full_context

    current = segments.get(target_page_idx, "")
    side_pages = [page_idx for page_idx in [target_page_idx - 1, target_page_idx + 1] if page_idx in segments]
    current_budget = min(len(current), max(0, int(max_context_chars * 0.6))) if current else 0
    remaining = max(0, max_context_chars - current_budget)
    side_budget = remaining // max(1, len(side_pages)) if side_pages else 0

    truncated = {}
    for page_idx, text in segments.items():
        if page_idx == target_page_idx:
            truncated[page_idx] = _truncate_text(text, current_budget or max_context_chars)
        else:
            truncated[page_idx] = _truncate_text(text, side_budget)

    context = "\n".join(truncated[page_idx] for page_idx in sorted(truncated) if truncated[page_idx])
    return _truncate_text(context, max_context_chars)


def should_caption_image_block(block: dict[str, Any]) -> bool:
    if block.get("type") != "image" or not block.get("img_path"):
        return False
    return not any(block.get(key) for key in INLINE_ICON_SKIP_KEYS)


def collect_image_description_stats(
    content_data: list[dict[str, Any]],
    *,
    base_dir: str | Path | None = None,
    skip_existing: bool = True,
) -> ImageDescriptionStats:
    stats = ImageDescriptionStats()
    base_path = Path(base_dir) if base_dir else None
    for block in content_data:
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        stats.images_seen += 1
        if any(block.get(key) for key in INLINE_ICON_SKIP_KEYS):
            stats.skipped_inline_icons += 1
            continue
        if not block.get("img_path"):
            stats.skipped_without_img_path += 1
            continue
        if skip_existing and str(block.get("image_description_vlm", "")).strip():
            stats.skipped_existing += 1
            continue
        stats.caption_candidates += 1
        if base_path is not None and not (base_path / block["img_path"]).exists():
            stats.missing_image_files += 1
    return stats


def _print_stats(stats: ImageDescriptionStats, output_json: Path) -> None:
    print("Image captioning stats:")
    print(f"  images seen: {stats.images_seen}")
    print(f"  caption candidates: {stats.caption_candidates}")
    print(f"  requests submitted: {stats.requests_submitted}")
    print(f"  captioned: {stats.captioned_count}")
    print(f"  skipped inline icons: {stats.skipped_inline_icons}")
    print(f"  skipped without img_path: {stats.skipped_without_img_path}")
    print(f"  skipped existing descriptions: {stats.skipped_existing}")
    print(f"  missing image files: {stats.missing_image_files}")
    print(f"  failed image reads: {stats.failed_image_reads}")
    print(f"  VLM batches: {stats.batches_processed}")
    print(f"  checkpoints written: {stats.checkpoints_written}")
    print(f"  output: {output_json}")


def add_image_descriptions(
    *,
    base_dir: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    model_name: str,
    max_new_tokens: int = 2000,
    batch_size: int = 4,
    max_context_chars: int = 12000,
    model_revision: str = "",
    trusted_remote_code_models: tuple[str, ...] = ("Qwen/Qwen3.5-9B",),
    checkpoint_interval: int = 1,
    checkpoint_json: str | Path | None = None,
    resume: bool = True,
    skip_existing: bool = True,
) -> None:
    import torch
    from modelscope import snapshot_download
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base_path = Path(base_dir)
    output_path = Path(output_json)
    checkpoint_path = Path(checkpoint_json) if checkpoint_json else checkpoint_path_for(output_path)
    if resume and checkpoint_path.exists():
        print(f"Resuming image captioning from checkpoint: {checkpoint_path}")
        source_json = checkpoint_path
    else:
        source_json = Path(input_json)

    with source_json.open("r", encoding="utf-8") as f:
        content_data: list[dict[str, Any]] = json.load(f)

    stats = collect_image_description_stats(content_data, base_dir=base_path, skip_existing=skip_existing)
    page_text_map = build_page_text_map(content_data)
    device = get_torch_device(feature="Image-description VLM preprocessing")
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
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    def write_checkpoint() -> None:
        _write_json(checkpoint_path, content_data)
        stats.checkpoints_written += 1

    def process_batch(requests: list[dict[str, Any]]) -> None:
        if not requests:
            return
        stats.requests_submitted += len(requests)
        stats.batches_processed += 1

        batch_messages = [
            {"role": "user", "content": [{"type": "image", "image": req["image"]}, {"type": "text", "text": req["prompt"]}]}
            for req in requests
        ]
        texts = [
            processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for message in batch_messages
        ]

        image_inputs_list = []
        video_inputs_list = []
        for message in batch_messages:
            image_inputs, video_inputs = process_vision_info([message])
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
            if output:
                content_data[req["idx"]]["image_description_vlm"] = output
                stats.captioned_count += 1

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if checkpoint_interval > 0 and stats.batches_processed % checkpoint_interval == 0:
            write_checkpoint()

    try:
        batch: list[dict[str, Any]] = []
        for idx in tqdm(range(len(content_data)), desc="Processing images"):
            block = content_data[idx]
            if not isinstance(block, dict) or block.get("type") != "image":
                continue
            if any(block.get(key) for key in INLINE_ICON_SKIP_KEYS):
                continue
            if not block.get("img_path"):
                continue
            if skip_existing and str(block.get("image_description_vlm", "")).strip():
                continue
            if not should_caption_image_block(block):
                continue
            image_path = base_path / block["img_path"]
            if not image_path.exists():
                print(f"Warning: image not found: {image_path}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as exc:
                stats.failed_image_reads += 1
                print(f"Warning: failed to read {image_path}: {exc}")
                continue

            page_idx = int(block.get("page_idx", 0))
            context_text = get_three_page_context(
                page_text_map,
                page_idx,
                max_context_chars=max_context_chars,
            )
            prompt = (
                "You are an expert technical documentation assistant. I will provide an image "
                "extracted from a manual, plus text context from the previous, current, and next pages.\n\n"
                f"### Text Context:\n{context_text}\n\n"
                "### Task:\n"
                "Provide a clear technical description of the image. Explain what the interface, "
                "diagram, or chart shows, what features it highlights, and its purpose according "
                "to the manual. Do not include greetings. If the image is simple, describe it briefly."
            )
            batch.append({"idx": idx, "page_idx": page_idx, "image": image, "prompt": prompt})
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []

        if batch:
            process_batch(batch)

    except Exception:
        write_checkpoint()
        print(f"Image captioning checkpoint saved before failure: {checkpoint_path}")
        raise
    finally:
        del model
        del processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_json(output_path, content_data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    _print_stats(stats, output_path)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Add context-aware image descriptions to MinerU JSON.")
    parser.add_argument("--artifact-dir", help="MinerU output folder containing *_content_list_PATCHED.json.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default=config.models.vlm_model)
    parser.add_argument("--model-revision", default=config.models.vlm_model_revision)
    parser.add_argument("--max-new-tokens", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved inputs and image counts without loading the VLM.")
    args = parser.parse_args(argv)

    if args.artifact_dir:
        if args.base_dir or args.input:
            parser.error("--artifact-dir cannot be combined with --base-dir or --input.")
        if args.output:
            artifacts_list = [resolve_image_description_artifacts(args.artifact_dir, output_json=args.output)]
        else:
            artifacts_list = resolve_image_description_batch(args.artifact_dir, recursive=not args.no_recursive)
    else:
        input_json = Path(args.input).expanduser() if args.input else config.paths.patched_json
        output_json = Path(args.output).expanduser() if args.output else config.paths.captioned_json
        base_dir = Path(args.base_dir).expanduser() if args.base_dir else config.paths.base_dir
        artifacts_list = [
            ImageDescriptionArtifacts(
                artifact_dir=base_dir,
                base_dir=base_dir,
                input_json=input_json,
                output_json=output_json,
            )
        ]

    if len(artifacts_list) > 1 and args.checkpoint_json:
        parser.error("--checkpoint-json can only be used with a single captioning job.")

    if args.dry_run:
        print(f"Image captioning jobs: {len(artifacts_list)}")
        for artifacts in artifacts_list:
            print("Image captioning inputs:")
            print(f"  artifact_dir: {artifacts.artifact_dir}")
            print(f"  base_dir: {artifacts.base_dir}")
            print(f"  input_json: {artifacts.input_json}")
            print(f"  output_json: {artifacts.output_json}")
            print(f"  checkpoint_json: {args.checkpoint_json or checkpoint_path_for(artifacts.output_json)}")
            print(f"  max_context_chars: {args.max_context_chars}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  batch_size: {args.batch_size}")
            if artifacts.input_json.exists():
                with artifacts.input_json.open("r", encoding="utf-8") as f:
                    content_data: list[dict[str, Any]] = json.load(f)
                stats = collect_image_description_stats(
                    content_data,
                    base_dir=artifacts.base_dir,
                    skip_existing=not args.no_skip_existing,
                )
                _print_stats(stats, artifacts.output_json)
            else:
                print("  input_exists: false")
        return

    for job_idx, artifacts in enumerate(artifacts_list, start=1):
        print(f"Image captioning job {job_idx}/{len(artifacts_list)}: {artifacts.artifact_dir}")
        add_image_descriptions(
            base_dir=artifacts.base_dir,
            input_json=artifacts.input_json,
            output_json=artifacts.output_json,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            max_context_chars=args.max_context_chars,
            model_revision=args.model_revision,
            trusted_remote_code_models=config.models.trusted_remote_code_models,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_json=args.checkpoint_json,
            resume=not args.no_resume,
            skip_existing=not args.no_skip_existing,
        )


if __name__ == "__main__":
    main()
