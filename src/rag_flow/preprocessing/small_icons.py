from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from rag_flow.config import AppConfig


IGNORE_TYPES = {
    "header",
    "footer",
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


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def crop_image_from_block(block: dict[str, Any], pdf_images: list[Any]) -> Any | None:
    page_idx = int(block.get("page_idx", 0))
    if page_idx >= len(pdf_images) or "bbox" not in block:
        return None

    image = pdf_images[page_idx]
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
    mineru_data: list[dict[str, Any]],
    pdf_images: list[Any],
    block_idx: int,
) -> Any | None:
    block = mineru_data[block_idx]
    last_idx = block_idx
    lookahead_idx = block_idx + 1

    while lookahead_idx < len(mineru_data):
        next_block = mineru_data[lookahead_idx]
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

    last_block = mineru_data[last_idx]
    page_idx = int(last_block.get("page_idx", 0))
    if page_idx >= len(pdf_images) or "bbox" not in last_block:
        return None

    image = pdf_images[page_idx]
    y0_norm = last_block["bbox"][3]
    y1_norm = 1000

    for idx in range(last_idx + 1, len(mineru_data)):
        next_block = mineru_data[idx]
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


def add_small_icon_text(
    *,
    input_json: str | Path,
    output_json: str | Path,
    pdf_path: str | Path,
    model_name: str,
    dpi: int = 200,
    batch_size: int = 6,
    max_new_tokens: int = 5000,
) -> None:
    import torch
    from modelscope import snapshot_download
    from pdf2image import convert_from_path
    from qwen_vl_utils import process_vision_info
    from tqdm import tqdm
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_dir = snapshot_download(model_name)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.generation_config.pad_token_id = processor.tokenizer.eos_token_id

    with Path(input_json).open("r", encoding="utf-8") as f:
        mineru_data: list[dict[str, Any]] = json.load(f)
    pdf_images = convert_from_path(str(pdf_path), dpi=dpi)

    patched_count = 0
    visited_blocks: set[tuple[int, str]] = set()

    def process_batch(requests: list[dict[str, Any]]) -> None:
        nonlocal patched_count
        if not requests:
            return

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
        ).to("cuda")

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

            if "No missing" not in output and "[Icon:" in output:
                idx = req["idx"]
                key = req["key"]
                mineru_data[idx][key] = output.split("\n") if req["is_list"] else output
                mineru_data[idx]["vlm-small-icon-patched"] = True
                patched_count += 1

        del inputs
        del generated_ids
        del generated_ids_trimmed
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    batch: list[dict[str, Any]] = []
    for idx in tqdm(range(len(mineru_data)), desc="Scanning bboxes"):
        block = mineru_data[idx]
        block_type = block.get("type")
        if block_type in IGNORE_TYPES or block_type not in TEXT_FIELD_MAP or "bbox" not in block:
            continue

        for key in TEXT_FIELD_MAP[block_type]:
            if (idx, key) in visited_blocks or not block.get(key):
                continue
            is_list = isinstance(block[key], list)
            original_text = _join(block[key]).strip()
            if not original_text:
                continue

            final_image = None
            if key == "table_footnote":
                final_image = build_table_footnote_crop(
                    mineru_data=mineru_data,
                    pdf_images=pdf_images,
                    block_idx=idx,
                )
            else:
                crops = []
                base_image = crop_image_from_block(block, pdf_images)
                if base_image:
                    crops.append(base_image)

                lookahead_idx = idx + 1
                while lookahead_idx < len(mineru_data):
                    next_block = mineru_data[lookahead_idx]
                    next_type = next_block.get("type")
                    if next_type in IGNORE_TYPES:
                        lookahead_idx += 1
                        continue
                    if next_type != block_type:
                        break
                    next_text = _join(next_block.get(key, "")).strip()
                    if next_text:
                        break
                    if "bbox" in next_block:
                        next_image = crop_image_from_block(next_block, pdf_images)
                        if next_image:
                            crops.append(next_image)
                    visited_blocks.add((lookahead_idx, key))
                    lookahead_idx += 1
                if crops:
                    final_image = concat_images_vertically(crops)

            if final_image is None:
                continue

            prompt = (
                "Please inspect the image and determine whether any small icons "
                "(for example plus sign, wrench, gear, arrow, save icon) are embedded "
                "in or around the text.\n"
                f'Here is the extracted text:\n"{original_text}"\n\n'
                "If icons are missing from the text, insert `[Icon: shape/name]` at the exact "
                "corresponding position. Return only the modified complete text."
            )
            batch.append(
                {
                    "idx": idx,
                    "key": key,
                    "block_type": block_type,
                    "page_idx": int(block.get("page_idx", 0)),
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

    with Path(output_json).open("w", encoding="utf-8") as f:
        json.dump(mineru_data, f, ensure_ascii=False, indent=2)
    print(f"Patched {patched_count} missing icon texts at {output_json}")


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Use a VLM to patch small icon text missing from MinerU JSON.")
    parser.add_argument("--input", default=str(config.paths.content_json))
    parser.add_argument("--output", default=str(config.paths.small_icon_json))
    parser.add_argument("--pdf", default=str(config.paths.source_pdf))
    parser.add_argument("--model", default=config.models.vlm_model)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    args = parser.parse_args(argv)

    add_small_icon_text(
        input_json=args.input,
        output_json=args.output,
        pdf_path=args.pdf,
        model_name=args.model,
        dpi=args.dpi,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
