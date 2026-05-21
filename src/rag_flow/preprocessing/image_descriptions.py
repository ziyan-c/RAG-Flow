from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rag_flow.config import AppConfig
from rag_flow.preprocessing.small_icons import (
    image_to_data_url,
    resolve_icon_patch_artifacts,
    resolve_icon_patch_batch,
)
from rag_flow.table_continuations import build_table_continuation_map, table_master_by_continuation


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

DEFAULT_CAPTION_MAX_NEW_TOKENS = 8000
DEFAULT_CAPTION_MAX_CONTEXT_TOKENS = 10000
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
SECTIONED_PATCHED_INPUT_SUFFIX = "_content_list_SECTIONED_PATCHED.json"
SECTIONED_PATCHED_CAPTIONED_SUFFIX = "_content_list_SECTIONED_PATCHED_CAPTIONED.json"
IMAGE_ANSWERING_POLICY_KEY = "image_answering_policy"
IMAGE_ANSWERING_CONFIDENCE_KEY = "image_answering_confidence"
IMAGE_ANSWERING_REASON_KEY = "image_answering_reason"


@dataclass(frozen=True)
class ImageDescriptionArtifacts:
    artifact_dir: Path
    base_dir: Path
    input_json: Path
    output_json: Path
    origin_pdf: Path


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


@dataclass(frozen=True)
class ContextTokenStats:
    contexts: int
    min_tokens: int
    p50_tokens: int
    p90_tokens: int
    p95_tokens: int
    max_tokens: int
    avg_tokens: float
    contexts_at_budget: int


@dataclass(frozen=True)
class ContextCollection:
    text: str
    block_indices: tuple[int, ...]


@dataclass(frozen=True)
class ContextBlockSelection:
    before_indices: tuple[int, ...]
    current_indices: tuple[int, ...]
    after_indices: tuple[int, ...]


class ImageDescriptionLLMOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_description_vlm: str
    image_answering_policy: Literal[
        "caption_only",
        "image_optional",
        "image_recommended",
        "image_required",
    ]
    image_answering_confidence: Literal["high", "medium", "low"]
    image_answering_reason: str


class TextBudgeter(Protocol):
    def count(self, text: str) -> int:
        ...

    def take_head(self, text: str, max_tokens: int) -> str:
        ...

    def take_tail(self, text: str, max_tokens: int) -> str:
        ...


class ApproxTokenBudgeter:
    """Conservative token estimate for dry-run stats without calling the LLM."""

    def count(self, text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(CJK_RE.findall(text))
        non_cjk_chars = len(text) - cjk_chars
        return cjk_chars + math.ceil(non_cjk_chars / 4)

    def take_head(self, text: str, max_tokens: int) -> str:
        return _take_estimated_with_marker(text, max_tokens, from_tail=False, budgeter=self)

    def take_tail(self, text: str, max_tokens: int) -> str:
        return _take_estimated_with_marker(text, max_tokens, from_tail=True, budgeter=self)


class TokenizerBudgeter:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def count(self, text: str) -> int:
        return len(self.encode(text)) if text else 0

    def take_head(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        token_ids = self.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        marker = "\n...[truncated]"
        marker_ids = self.encode(marker)
        if max_tokens <= len(marker_ids):
            return self.decode(token_ids[:max_tokens])
        return self.decode(token_ids[: max_tokens - len(marker_ids)]).rstrip() + marker

    def take_tail(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        token_ids = self.encode(text)
        if len(token_ids) <= max_tokens:
            return text
        marker = "[truncated before]...\n"
        marker_ids = self.encode(marker)
        if max_tokens <= len(marker_ids):
            return self.decode(token_ids[-max_tokens:])
        return marker + self.decode(token_ids[-(max_tokens - len(marker_ids)) :]).lstrip()


def _join(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value or "")


def captioned_json_path_for(input_json: str | Path) -> Path:
    path = Path(input_json)
    name = path.name
    if name.endswith(SECTIONED_PATCHED_CAPTIONED_SUFFIX):
        return path
    if name.endswith(SECTIONED_PATCHED_INPUT_SUFFIX):
        prefix = name[: -len(SECTIONED_PATCHED_INPUT_SUFFIX)]
        return path.with_name(f"{prefix}{SECTIONED_PATCHED_CAPTIONED_SUFFIX}")
    raise ValueError(
        "Captioning requires sectioned patched JSON "
        f"(*{SECTIONED_PATCHED_INPUT_SUFFIX}); old *_content_list_PATCHED.json inputs are not supported."
    )


def require_sectioned_patched_captioning_input(input_json: str | Path) -> Path:
    path = Path(input_json)
    if not path.name.endswith(SECTIONED_PATCHED_INPUT_SUFFIX):
        raise ValueError(
            "Captioning requires sectioned patched JSON "
            f"(*{SECTIONED_PATCHED_INPUT_SUFFIX}); run sectioning and patching first."
        )
    return path


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
        origin_pdf=patched_artifacts.origin_pdf,
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
            origin_pdf=artifacts.origin_pdf,
        )
        for artifacts in resolve_icon_patch_batch(artifact_dir, recursive=recursive)
    ]


def _block_context_text(block: dict[str, Any]) -> str:
    block_texts = []
    for key in TEXT_KEYS:
        text = _join(block.get(key, "")).strip()
        if text:
            block_texts.append(text)
    return "\n".join(block_texts)


def _format_document_context(block: dict[str, Any]) -> str:
    breadcrumb = str(block.get("breadcrumb", "")).strip()
    if not breadcrumb:
        return ""
    return f"### Document Context\nbreadcrumb: {breadcrumb}"


def _format_context_block(
    content_data: list[dict[str, Any]],
    idx: int,
    *,
    continuation_to_master: dict[int, int] | None = None,
) -> str:
    block = content_data[idx]
    source_block = block
    source_idx = idx
    continuation_suffix = ""
    if continuation_to_master and idx in continuation_to_master:
        master_idx = continuation_to_master[idx]
        if 0 <= master_idx < len(content_data) and isinstance(content_data[master_idx], dict):
            source_block = content_data[master_idx]
            source_idx = master_idx
            continuation_suffix = f", continuation of table block {master_idx}"

    text = _block_context_text(source_block)
    if not text:
        return ""
    page_idx = block.get("page_idx", "?")
    block_type = block.get("type", "unknown")
    source_label = f", source block {source_idx}" if source_idx != idx else ""
    return f"--- [Block {idx}, page {page_idx}, type {block_type}{continuation_suffix}{source_label}] ---\n{text}"


def _section_path_key(block: dict[str, Any]) -> tuple[str, ...] | None:
    value = block.get("section_path")
    if isinstance(value, list):
        parts = tuple(str(part).strip() for part in value if str(part).strip())
        return parts or None
    if isinstance(value, str):
        parts = tuple(part.strip() for part in value.split(">") if part.strip())
        return parts or None
    return None


def _context_section_key(
    content_data: list[dict[str, Any]],
    idx: int,
    *,
    continuation_to_master: dict[int, int] | None = None,
) -> tuple[str, ...] | None:
    block = content_data[idx]
    key = _section_path_key(block)
    if key or not continuation_to_master or idx not in continuation_to_master:
        return key
    master_idx = continuation_to_master[idx]
    if 0 <= master_idx < len(content_data) and isinstance(content_data[master_idx], dict):
        return _section_path_key(content_data[master_idx])
    return None


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = "\n...[truncated]"
    if max_chars <= len(marker):
        return text[:max_chars]
    return text[: max_chars - len(marker)].rstrip() + marker


def resize_image_for_captioning(image: Any, max_image_side: int) -> Any:
    if max_image_side <= 0:
        return image
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_image_side:
        return image
    scale = max_image_side / longest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    from PIL import Image as PILImage

    resampling = getattr(getattr(PILImage, "Resampling", None), "LANCZOS", PILImage.LANCZOS)
    return image.resize(new_size, resampling)


def _take_by_estimated_tokens(
    text: str,
    max_tokens: int,
    *,
    from_tail: bool,
    budgeter: TextBudgeter,
) -> str:
    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[-mid:] if from_tail else text[:mid]
        if budgeter.count(candidate) <= max_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _take_estimated_with_marker(
    text: str,
    max_tokens: int,
    *,
    from_tail: bool,
    budgeter: TextBudgeter,
) -> str:
    if max_tokens <= 0 or not text:
        return ""
    if budgeter.count(text) <= max_tokens:
        return text

    if from_tail:
        marker = "[truncated before]...\n"
        marker_tokens = budgeter.count(marker)
        if max_tokens <= marker_tokens:
            return _take_by_estimated_tokens(text, max_tokens, from_tail=True, budgeter=budgeter)
        tail = _take_by_estimated_tokens(
            text,
            max_tokens - marker_tokens,
            from_tail=True,
            budgeter=budgeter,
        )
        return marker + tail.lstrip()

    marker = "\n...[truncated]"
    marker_tokens = budgeter.count(marker)
    if max_tokens <= marker_tokens:
        return _take_by_estimated_tokens(text, max_tokens, from_tail=False, budgeter=budgeter)
    head = _take_by_estimated_tokens(
        text,
        max_tokens - marker_tokens,
        from_tail=False,
        budgeter=budgeter,
    )
    return head.rstrip() + marker


def _collect_nearby_context(
    content_data: list[dict[str, Any]],
    target_idx: int,
    *,
    direction: int,
    max_tokens: int,
    budgeter: TextBudgeter,
    continuation_to_master: dict[int, int] | None = None,
    section_key: tuple[str, ...] | None = None,
) -> ContextCollection:
    if max_tokens <= 0:
        return ContextCollection(text="", block_indices=())
    if direction < 0:
        scan_range = range(target_idx - 1, -1, -1)
    else:
        scan_range = range(target_idx + 1, len(content_data))

    segments = []
    block_indices = []
    used = 0
    separator = "\n\n"
    separator_tokens = budgeter.count(separator)
    for idx in scan_range:
        block = content_data[idx]
        if not isinstance(block, dict):
            continue
        if section_key is not None:
            candidate_section_key = _context_section_key(
                content_data,
                idx,
                continuation_to_master=continuation_to_master,
            )
            if candidate_section_key is None:
                continue
            if candidate_section_key != section_key:
                break
        segment = _format_context_block(
            content_data,
            idx,
            continuation_to_master=continuation_to_master,
        )
        if not segment:
            continue
        separator_budget = separator_tokens if segments else 0
        remaining = max_tokens - used - separator_budget
        if remaining <= 0:
            break
        segment_tokens = budgeter.count(segment)
        if segment_tokens > remaining:
            segment = (
                budgeter.take_tail(segment, remaining)
                if direction < 0
                else budgeter.take_head(segment, remaining)
            )
            segment_tokens = budgeter.count(segment)
        if not segment:
            break
        segments.append(segment)
        block_indices.append(idx)
        used += separator_budget + segment_tokens

    if direction < 0:
        segments.reverse()
        block_indices.reverse()
    return ContextCollection(text=separator.join(segments), block_indices=tuple(block_indices))


def collect_surrounding_context_selection(
    content_data: list[dict[str, Any]],
    target_idx: int,
    *,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    budgeter: TextBudgeter | None = None,
    table_continuations: dict[int, list[int]] | None = None,
) -> tuple[str, ContextBlockSelection]:
    budgeter = budgeter or ApproxTokenBudgeter()
    if max_context_tokens <= 0:
        document_context = ""
        if 0 <= target_idx < len(content_data) and isinstance(content_data[target_idx], dict):
            document_context = _format_document_context(content_data[target_idx])
        return document_context, ContextBlockSelection(before_indices=(), current_indices=(), after_indices=())

    resolved_continuations = (
        build_table_continuation_map(content_data) if table_continuations is None else table_continuations
    )
    continuation_to_master = table_master_by_continuation(resolved_continuations)

    target = ""
    current_indices: tuple[int, ...] = ()
    target_section_key: tuple[str, ...] | None = None
    if 0 <= target_idx < len(content_data) and isinstance(content_data[target_idx], dict):
        target_section_key = _context_section_key(
            content_data,
            target_idx,
            continuation_to_master=continuation_to_master,
        )
        target = _format_context_block(
            content_data,
            target_idx,
            continuation_to_master=continuation_to_master,
        )
        if target:
            current_indices = (target_idx,)

    target_budget = min(budgeter.count(target), max(0, max_context_tokens // 5)) if target else 0
    target_context = budgeter.take_head(target, target_budget)
    remaining = max_context_tokens - budgeter.count(target_context)
    before_budget = max(0, remaining // 2)
    after_budget = max(0, remaining - before_budget)

    before_context = _collect_nearby_context(
        content_data,
        target_idx,
        direction=-1,
        max_tokens=before_budget,
        budgeter=budgeter,
        continuation_to_master=continuation_to_master,
        section_key=target_section_key,
    )
    after_context = _collect_nearby_context(
        content_data,
        target_idx,
        direction=1,
        max_tokens=after_budget,
        budgeter=budgeter,
        continuation_to_master=continuation_to_master,
        section_key=target_section_key,
    )

    before_used = budgeter.count(before_context.text)
    after_used = budgeter.count(after_context.text)
    if before_used < before_budget:
        expanded_after_budget = max(0, remaining - before_used)
        after_context = _collect_nearby_context(
            content_data,
            target_idx,
            direction=1,
            max_tokens=expanded_after_budget,
            budgeter=budgeter,
            continuation_to_master=continuation_to_master,
            section_key=target_section_key,
        )
        after_used = budgeter.count(after_context.text)
    if after_used < after_budget:
        expanded_before_budget = max(0, remaining - after_used)
        before_context = _collect_nearby_context(
            content_data,
            target_idx,
            direction=-1,
            max_tokens=expanded_before_budget,
            budgeter=budgeter,
            continuation_to_master=continuation_to_master,
            section_key=target_section_key,
        )

    document_context = ""
    if 0 <= target_idx < len(content_data) and isinstance(content_data[target_idx], dict):
        document_context = _format_document_context(content_data[target_idx])
    budgeted_sections = []
    if before_context.text:
        budgeted_sections.append("### Nearby Text Before Image\n" + before_context.text)
    if target_context:
        budgeted_sections.append("### Current Image Caption/Footnote\n" + target_context)
    if after_context.text:
        budgeted_sections.append("### Nearby Text After Image\n" + after_context.text)
    budgeted_context = budgeter.take_head("\n\n".join(budgeted_sections), max_context_tokens)
    context = "\n\n".join(part for part in (document_context, budgeted_context) if part)
    selection = ContextBlockSelection(
        before_indices=before_context.block_indices,
        current_indices=current_indices if target_context else (),
        after_indices=after_context.block_indices,
    )
    return context, selection


def get_surrounding_text_context(
    content_data: list[dict[str, Any]],
    target_idx: int,
    *,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    budgeter: TextBudgeter | None = None,
    table_continuations: dict[int, list[int]] | None = None,
) -> str:
    context, _selection = collect_surrounding_context_selection(
        content_data,
        target_idx,
        max_context_tokens=max_context_tokens,
        budgeter=budgeter,
        table_continuations=table_continuations,
    )
    return context


def should_caption_image_block(block: dict[str, Any]) -> bool:
    if block.get("type") != "image" or not block.get("img_path"):
        return False
    return not any(block.get(key) for key in INLINE_ICON_SKIP_KEYS)


def _is_caption_candidate(block: dict[str, Any], *, skip_existing: bool = True) -> bool:
    if not should_caption_image_block(block):
        return False
    if skip_existing and str(block.get("image_description_vlm", "")).strip():
        return False
    return True


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


def _percentile(sorted_values: list[int], percentile: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, math.ceil(len(sorted_values) * percentile) - 1)
    return sorted_values[idx]


def collect_context_token_stats(
    content_data: list[dict[str, Any]],
    *,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    skip_existing: bool = True,
    budgeter: TextBudgeter | None = None,
) -> ContextTokenStats:
    budgeter = budgeter or ApproxTokenBudgeter()
    table_continuations = build_table_continuation_map(content_data)
    token_counts = []
    contexts_at_budget = 0
    for idx, block in enumerate(content_data):
        if not isinstance(block, dict) or not _is_caption_candidate(block, skip_existing=skip_existing):
            continue
        context = get_surrounding_text_context(
            content_data,
            idx,
            max_context_tokens=max_context_tokens,
            budgeter=budgeter,
            table_continuations=table_continuations,
        )
        token_count = budgeter.count(context)
        token_counts.append(token_count)
        if max_context_tokens > 0 and token_count >= int(max_context_tokens * 0.95):
            contexts_at_budget += 1

    if not token_counts:
        return ContextTokenStats(
            contexts=0,
            min_tokens=0,
            p50_tokens=0,
            p90_tokens=0,
            p95_tokens=0,
            max_tokens=0,
            avg_tokens=0.0,
            contexts_at_budget=0,
        )

    sorted_counts = sorted(token_counts)
    return ContextTokenStats(
        contexts=len(sorted_counts),
        min_tokens=sorted_counts[0],
        p50_tokens=_percentile(sorted_counts, 0.50),
        p90_tokens=_percentile(sorted_counts, 0.90),
        p95_tokens=_percentile(sorted_counts, 0.95),
        max_tokens=sorted_counts[-1],
        avg_tokens=sum(sorted_counts) / len(sorted_counts),
        contexts_at_budget=contexts_at_budget,
    )


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
    print(f"  LLM batches: {stats.batches_processed}")
    print(f"  checkpoints written: {stats.checkpoints_written}")
    print(f"  output: {output_json}")


def _print_context_token_stats(stats: ContextTokenStats, *, estimated: bool = False) -> None:
    label = "estimated " if estimated else ""
    print(f"Image captioning {label}context token stats:")
    print(f"  contexts: {stats.contexts}")
    print(
        "  min/p50/p90/p95/max: "
        f"{stats.min_tokens}/{stats.p50_tokens}/{stats.p90_tokens}/{stats.p95_tokens}/{stats.max_tokens}"
    )
    print(f"  average: {stats.avg_tokens:.1f}")
    print(f"  contexts near budget: {stats.contexts_at_budget}")


def make_captioning_llm_client(*, base_url: str, api_key: str, timeout: float) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _emit_metric(metrics_sink: Any | None, kind: str, payload: dict[str, Any]) -> None:
    if metrics_sink is None:
        return
    emit = getattr(metrics_sink, "emit", None)
    if emit is None:
        return
    emit(kind, payload)


def assert_captioning_llm_available(client: Any, *, base_url: str) -> None:
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    try:
        client.models.list()
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            f"Cannot reach the captioning LLM service at {base_url}. "
            "Start it first with `rag-flow serve llm-sglang`."
        ) from exc
    except APIStatusError as exc:
        if exc.status_code in {404, 405}:
            return
        raise RuntimeError(
            f"The captioning LLM service at {base_url} is reachable but not ready "
            f"(HTTP {exc.status_code})."
        ) from exc


def image_description_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "image_description_output",
            "strict": True,
            "schema": ImageDescriptionLLMOutput.model_json_schema(),
        },
    }


def request_image_description_from_llm(
    *,
    client: Any,
    model: str,
    image: Any,
    prompt: str,
    max_tokens: int,
) -> ImageDescriptionLLMOutput:
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ModuleNotFoundError:
        class _OpenAIUnavailableError(Exception):
            pass

        APIConnectionError = APIStatusError = APITimeoutError = _OpenAIUnavailableError

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0,
            response_format=image_description_response_format(),
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "separate_reasoning": True,
            },
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(
            "Cannot reach the captioning LLM service. Start it first with "
            "`rag-flow serve llm-sglang`."
        ) from exc
    except APIStatusError as exc:
        raise RuntimeError(f"Captioning LLM request failed with HTTP {exc.status_code}: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Captioning LLM returned an empty response.")
    return ImageDescriptionLLMOutput.model_validate_json(content)


def _caption_request_metric_base(req: dict[str, Any], *, batch_id: int) -> dict[str, Any]:
    image = req.get("image")
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    pixels = int(width) * int(height) if isinstance(width, int) and isinstance(height, int) else None
    return {
        "batch_id": batch_id,
        "image_id": req.get("idx"),
        "block_idx": req.get("idx"),
        "page_idx": req.get("page_idx"),
        "pdf_page": int(req.get("page_idx", 0)) + 1 if req.get("page_idx") is not None else None,
        "img_path": req.get("img_path"),
        "original_image_width": req.get("original_image_width"),
        "original_image_height": req.get("original_image_height"),
        "input_image_width": width,
        "input_image_height": height,
        "input_image_pixels": pixels,
        "model_context_tokens": req.get("model_context_tokens"),
        "review_context_tokens": req.get("review_context_tokens"),
        "prompt_chars": len(str(req.get("prompt", ""))),
    }


def _run_image_description_request(
    req: dict[str, Any],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    batch_id: int,
    metrics_sink: Any | None,
) -> tuple[ImageDescriptionLLMOutput, dict[str, Any]]:
    started_at = time.time()
    started_perf = time.perf_counter()
    event = _caption_request_metric_base(req, batch_id=batch_id)
    event["started_at"] = started_at
    try:
        output = request_image_description_from_llm(
            client=client,
            model=model,
            image=req["image"],
            prompt=req["prompt"],
            max_tokens=max_tokens,
        )
    except Exception as exc:
        ended_at = time.time()
        event.update(
            {
                "ended_at": ended_at,
                "duration_s": time.perf_counter() - started_perf,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "written": False,
            }
        )
        _emit_metric(metrics_sink, "request", event)
        raise

    ended_at = time.time()
    event.update(
        {
            "ended_at": ended_at,
            "duration_s": time.perf_counter() - started_perf,
            "status": "ok",
            "error_type": None,
            "error": None,
            "output_chars": len(output.model_dump_json()),
        }
    )
    return output, event


def iter_image_description_results(
    requests: list[dict[str, Any]],
    *,
    client: Any,
    model: str,
    max_tokens: int,
    concurrency: int,
    batch_id: int = 0,
    metrics_sink: Any | None = None,
) -> Any:
    if concurrency <= 1 or len(requests) <= 1:
        for req in requests:
            output, event = _run_image_description_request(
                req,
                client=client,
                model=model,
                max_tokens=max_tokens,
                batch_id=batch_id,
                metrics_sink=metrics_sink,
            )
            yield req, output, event
        return

    max_workers = min(concurrency, len(requests))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {
            executor.submit(
                _run_image_description_request,
                req,
                client=client,
                model=model,
                max_tokens=max_tokens,
                batch_id=batch_id,
                metrics_sink=metrics_sink,
            ): req
            for req in requests
        }
        try:
            for future in as_completed(future_to_request):
                output, event = future.result()
                yield future_to_request[future], output, event
        except Exception:
            for future in future_to_request:
                future.cancel()
            raise


def add_image_descriptions(
    *,
    base_dir: str | Path,
    input_json: str | Path,
    output_json: str | Path,
    pdf_path: str | Path | None = None,
    model_name: str,
    max_new_tokens: int = DEFAULT_CAPTION_MAX_NEW_TOKENS,
    batch_size: int = 4,
    concurrency: int = 1,
    max_context_tokens: int = DEFAULT_CAPTION_MAX_CONTEXT_TOKENS,
    max_image_side: int = 0,
    llm_base_url: str = "http://localhost:8080/v1",
    llm_api_key: str = "EMPTY",
    llm_timeout: float = 120.0,
    checkpoint_interval: int = 1,
    checkpoint_json: str | Path | None = None,
    resume: bool = True,
    skip_existing: bool = True,
    write_captioning_view: bool = True,
    captioning_view_pdf: str | Path | None = None,
    review_context_tokens: int | None = None,
    metrics_sink: Any | None = None,
) -> None:
    from PIL import Image
    from tqdm import tqdm

    base_path = Path(base_dir)
    require_sectioned_patched_captioning_input(input_json)
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
    llm_client = make_captioning_llm_client(
        base_url=llm_base_url,
        api_key=llm_api_key,
        timeout=llm_timeout,
    )
    assert_captioning_llm_available(llm_client, base_url=llm_base_url)
    context_budgeter = ApproxTokenBudgeter()
    table_continuations = build_table_continuation_map(content_data)

    def write_checkpoint() -> None:
        started_at = time.time()
        started_perf = time.perf_counter()
        _write_json(checkpoint_path, content_data)
        stats.checkpoints_written += 1
        _emit_metric(
            metrics_sink,
            "checkpoint",
            {
                "reason": "interval",
                "path": str(checkpoint_path),
                "checkpoint_index": stats.checkpoints_written,
                "batches_processed": stats.batches_processed,
                "requests_submitted": stats.requests_submitted,
                "started_at": started_at,
                "ended_at": time.time(),
                "duration_s": time.perf_counter() - started_perf,
                "file_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            },
        )

    def process_batch(requests: list[dict[str, Any]]) -> None:
        if not requests:
            return
        stats.requests_submitted += len(requests)
        stats.batches_processed += 1

        batch_id = stats.batches_processed
        for req, output, request_event in iter_image_description_results(
            requests,
            client=llm_client,
            model=model_name,
            max_tokens=max_new_tokens,
            concurrency=concurrency,
            batch_id=batch_id,
            metrics_sink=metrics_sink,
        ):
            block = content_data[req["idx"]]
            block["image_description_vlm"] = output.image_description_vlm
            block[IMAGE_ANSWERING_POLICY_KEY] = output.image_answering_policy
            block[IMAGE_ANSWERING_CONFIDENCE_KEY] = output.image_answering_confidence
            block[IMAGE_ANSWERING_REASON_KEY] = output.image_answering_reason
            stats.captioned_count += 1
            request_event.update(
                {
                    "decision": "captioned",
                    "written": True,
                    "image_answering_policy": output.image_answering_policy,
                    "image_answering_confidence": output.image_answering_confidence,
                }
            )
            _emit_metric(metrics_sink, "request", request_event)

        if checkpoint_interval > 0 and stats.batches_processed % checkpoint_interval == 0:
            write_checkpoint()

    try:
        batch: list[dict[str, Any]] = []
        candidate_indices = [
            idx
            for idx, block in enumerate(content_data)
            if isinstance(block, dict) and _is_caption_candidate(block, skip_existing=skip_existing)
        ]
        for idx in tqdm(candidate_indices, desc="Processing images"):
            block = content_data[idx]
            image_path = base_path / block["img_path"]
            if not image_path.exists():
                print(f"Warning: image not found: {image_path}")
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                original_width, original_height = image.size
                image = resize_image_for_captioning(image, max_image_side)
            except Exception as exc:
                stats.failed_image_reads += 1
                print(f"Warning: failed to read {image_path}: {exc}")
                continue

            page_idx = int(block.get("page_idx", 0))
            context_text, context_selection = collect_surrounding_context_selection(
                content_data,
                idx,
                max_context_tokens=max_context_tokens,
                budgeter=context_budgeter,
                table_continuations=table_continuations,
            )
            review_context_token_count = None
            if review_context_tokens is not None:
                review_context, _review_selection = collect_surrounding_context_selection(
                    content_data,
                    idx,
                    max_context_tokens=review_context_tokens,
                    budgeter=context_budgeter,
                    table_continuations=table_continuations,
                )
                review_context_token_count = context_budgeter.count(review_context)
            prompt = (
                "You are an expert technical documentation assistant. I will provide an image "
                "extracted from a manual, plus document breadcrumb and nearby text "
                "before and after this image in the same section when section metadata is available.\n\n"
                f"### Text Context:\n{context_text}\n\n"
                "### Task:\n"
                "Describe only what is visible in the image, using the nearby text only to resolve "
                "technical terms, feature names, and purpose. Do not repeat unrelated context or "
                "invent details that are not visible. Explain what the interface, diagram, chart, "
                "or screenshot shows, the visible labels or states that matter, and why it appears "
                "in the manual. Keep the description concise when the image is simple, but be "
                "complete for dense technical diagrams or UI screenshots.\n\n"
                "Also judge whether the generated text description is enough for a future answering "
                "LLM, or whether the original image should be supplied alongside the description. "
                "Use `caption_only` when the text description should be enough, `image_optional` "
                "when the image might help but should not be sent by default, `image_recommended` "
                "when the image should usually be sent if this evidence is retrieved, and "
                "`image_required` when the description cannot reliably replace the image.\n\n"
                "Return a JSON object matching the required response schema:\n"
                "{\n"
                '  "image_description_vlm": "...",\n'
                '  "image_answering_policy": "caption_only|image_optional|image_recommended|image_required",\n'
                '  "image_answering_confidence": "high|medium|low",\n'
                '  "image_answering_reason": "..."\n'
                "}"
            )
            batch.append(
                {
                    "idx": idx,
                    "page_idx": page_idx,
                    "img_path": block["img_path"],
                    "image": image,
                    "prompt": prompt,
                    "original_image_width": original_width,
                    "original_image_height": original_height,
                    "model_context_tokens": context_budgeter.count(context_text),
                    "review_context_tokens": review_context_token_count,
                    "context_before_indices": list(context_selection.before_indices),
                    "context_current_indices": list(context_selection.current_indices),
                    "context_after_indices": list(context_selection.after_indices),
                }
            )
            if len(batch) >= batch_size:
                process_batch(batch)
                batch = []

        if batch:
            process_batch(batch)

    except Exception:
        started_at = time.time()
        started_perf = time.perf_counter()
        _write_json(checkpoint_path, content_data)
        stats.checkpoints_written += 1
        _emit_metric(
            metrics_sink,
            "checkpoint",
            {
                "reason": "failure",
                "path": str(checkpoint_path),
                "checkpoint_index": stats.checkpoints_written,
                "batches_processed": stats.batches_processed,
                "requests_submitted": stats.requests_submitted,
                "started_at": started_at,
                "ended_at": time.time(),
                "duration_s": time.perf_counter() - started_perf,
                "file_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            },
        )
        print(f"Image captioning checkpoint saved before failure: {checkpoint_path}")
        raise
    _write_json(output_path, content_data)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if write_captioning_view and pdf_path:
        from rag_flow.preprocessing.captioning_view import write_captioning_view_pdf

        view_stats = write_captioning_view_pdf(
            content_json=input_json,
            pdf_path=pdf_path,
            output_pdf=captioning_view_pdf,
            max_context_tokens=max_context_tokens,
            budgeter=context_budgeter,
        )
        print(f"Generated captioning view PDF at {view_stats.output_pdf}")
        print(f"  overlays: {view_stats.region_count}")
        print(f"  caption targets: {view_stats.caption_targets}")
    _print_stats(stats, output_path)


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Add context-aware image descriptions to MinerU JSON.")
    parser.add_argument("--artifact-dir", help="MinerU output folder containing patched content_list JSON.")
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--pdf", default=None, help="Source PDF used for the CAPTIONING_VIEW overlay.")
    parser.add_argument("--model", default=config.models.llm_model, help="Model name served by the SGLang API.")
    parser.add_argument("--llm-base-url", default=config.models.llm_base_url)
    parser.add_argument("--llm-api-key", default=config.models.llm_api_key)
    parser.add_argument("--request-timeout", type=float, default=config.captioning.llm_timeout)
    parser.add_argument("--max-new-tokens", type=int, default=config.captioning.max_new_tokens)
    parser.add_argument("--batch-size", type=int, default=config.captioning.batch_size)
    parser.add_argument("--concurrency", type=int, default=config.captioning.concurrency)
    parser.add_argument("--max-context-tokens", type=int, default=config.captioning.max_context_tokens)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=config.captioning.max_image_side,
        help="Resize captioning images so their longest side is at most this many pixels; 0 keeps original size.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=config.captioning.checkpoint_interval)
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--captioning-view-pdf", help="Output PDF that visualizes captioning targets and context blocks.")
    parser.add_argument("--no-captioning-view", action="store_true", help="Do not write the CAPTIONING_VIEW PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved inputs and image counts without calling the LLM.")
    args = parser.parse_args(argv)

    if args.artifact_dir:
        if args.base_dir or args.input:
            parser.error("--artifact-dir cannot be combined with --base-dir or --input.")
        if args.pdf:
            parser.error("--pdf cannot be combined with --artifact-dir; use the artifact folder's *_origin.pdf.")
        if args.output:
            artifacts_list = [resolve_image_description_artifacts(args.artifact_dir, output_json=args.output)]
        else:
            artifacts_list = resolve_image_description_batch(args.artifact_dir, recursive=not args.no_recursive)
    else:
        input_json = Path(args.input).expanduser() if args.input else config.paths.patched_json
        try:
            require_sectioned_patched_captioning_input(input_json)
        except ValueError as exc:
            parser.error(str(exc))
        output_json = Path(args.output).expanduser() if args.output else config.paths.captioned_json
        base_dir = Path(args.base_dir).expanduser() if args.base_dir else config.paths.base_dir
        artifacts_list = [
            ImageDescriptionArtifacts(
                artifact_dir=base_dir,
                base_dir=base_dir,
                input_json=input_json,
                output_json=output_json,
                origin_pdf=Path(args.pdf).expanduser() if args.pdf else config.paths.source_pdf,
            )
        ]

    if len(artifacts_list) > 1 and args.checkpoint_json:
        parser.error("--checkpoint-json can only be used with a single captioning job.")
    if len(artifacts_list) > 1 and args.captioning_view_pdf:
        parser.error("--captioning-view-pdf can only be used with a single captioning job.")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be >= 0.")

    if args.dry_run:
        print(f"Image captioning jobs: {len(artifacts_list)}")
        for artifacts in artifacts_list:
            print("Image captioning inputs:")
            print(f"  artifact_dir: {artifacts.artifact_dir}")
            print(f"  base_dir: {artifacts.base_dir}")
            print(f"  input_json: {artifacts.input_json}")
            print(f"  output_json: {artifacts.output_json}")
            print(f"  input_pdf: {artifacts.origin_pdf}")
            print(f"  checkpoint_json: {args.checkpoint_json or checkpoint_path_for(artifacts.output_json)}")
            if args.no_captioning_view:
                print("  captioning_view_pdf: disabled")
            else:
                from rag_flow.preprocessing.captioning_view import captioning_view_path_for

                print(f"  captioning_view_pdf: {args.captioning_view_pdf or captioning_view_path_for(artifacts.input_json)}")
            print(f"  max_context_tokens: {args.max_context_tokens}")
            print(f"  max_image_side: {args.max_image_side}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  batch_size: {args.batch_size}")
            print(f"  concurrency: {args.concurrency}")
            print(f"  checkpoint_interval: {args.checkpoint_interval}")
            print(f"  llm_base_url: {args.llm_base_url}")
            print(f"  llm_model: {args.model}")
            print(f"  request_timeout: {args.request_timeout}")
            if artifacts.input_json.exists():
                with artifacts.input_json.open("r", encoding="utf-8") as f:
                    content_data: list[dict[str, Any]] = json.load(f)
                stats = collect_image_description_stats(
                    content_data,
                    base_dir=artifacts.base_dir,
                    skip_existing=not args.no_skip_existing,
                )
                _print_stats(stats, artifacts.output_json)
                context_stats = collect_context_token_stats(
                    content_data,
                    max_context_tokens=args.max_context_tokens,
                    skip_existing=not args.no_skip_existing,
                )
                _print_context_token_stats(context_stats, estimated=True)
            else:
                print("  input_exists: false")
        return

    for job_idx, artifacts in enumerate(artifacts_list, start=1):
        print(f"Image captioning job {job_idx}/{len(artifacts_list)}: {artifacts.artifact_dir}")
        add_image_descriptions(
            base_dir=artifacts.base_dir,
            input_json=artifacts.input_json,
            output_json=artifacts.output_json,
            pdf_path=artifacts.origin_pdf,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            max_context_tokens=args.max_context_tokens,
            max_image_side=args.max_image_side,
            llm_base_url=args.llm_base_url,
            llm_api_key=args.llm_api_key,
            llm_timeout=args.request_timeout,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_json=args.checkpoint_json,
            resume=not args.no_resume,
            skip_existing=not args.no_skip_existing,
            write_captioning_view=not args.no_captioning_view,
            captioning_view_pdf=args.captioning_view_pdf,
        )


if __name__ == "__main__":
    main()
