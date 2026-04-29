from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig
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


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


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


def get_three_page_context(page_text_map: dict[int, list[str]], target_page_idx: int) -> str:
    context_lines = []
    for page_idx in [target_page_idx - 1, target_page_idx, target_page_idx + 1]:
        if page_idx in page_text_map:
            context_lines.append(f"\n--- [Text from Page {page_idx}] ---")
            context_lines.append("\n\n".join(page_text_map[page_idx]))
    return "\n".join(context_lines)


def add_image_descriptions(
    *,
    base_dir: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    model_name: str,
    max_new_tokens: int = 10000,
    batch_size: int = 4,
    model_revision: str = "",
    trusted_remote_code_models: tuple[str, ...] = ("Qwen/Qwen3.5-9B",),
) -> None:
    import torch
    from modelscope import snapshot_download
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    base_path = Path(base_dir)
    with Path(input_json).open("r", encoding="utf-8") as f:
        content_data: list[dict[str, Any]] = json.load(f)

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

    captioned_count = 0

    def process_batch(requests: list[dict[str, Any]]) -> None:
        nonlocal captioned_count
        if not requests:
            return

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
                captioned_count += 1

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch: list[dict[str, Any]] = []
    for idx in tqdm(range(len(content_data)), desc="Processing images"):
        block = content_data[idx]
        if block.get("type") != "image" or not block.get("img_path"):
            continue
        image_path = base_path / block["img_path"]
        if not image_path.exists():
            print(f"Warning: image not found: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            print(f"Warning: failed to read {image_path}: {exc}")
            continue

        page_idx = int(block.get("page_idx", 0))
        context_text = get_three_page_context(page_text_map, page_idx)
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

    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(content_data, f, ensure_ascii=False, indent=2)
    print(f"Generated {captioned_count} image descriptions at {output_json}")


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Add context-aware image descriptions to MinerU JSON.")
    parser.add_argument("--base-dir", default=str(config.paths.base_dir))
    parser.add_argument("--input", default=str(config.paths.patched_json))
    parser.add_argument("--output", default=str(config.paths.captioned_json))
    parser.add_argument("--model", default=config.models.vlm_model)
    parser.add_argument("--model-revision", default=config.models.vlm_model_revision)
    parser.add_argument("--max-new-tokens", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    add_image_descriptions(
        base_dir=args.base_dir,
        input_json=args.input,
        output_json=args.output,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        model_revision=args.model_revision,
        trusted_remote_code_models=config.models.trusted_remote_code_models,
    )


if __name__ == "__main__":
    main()
