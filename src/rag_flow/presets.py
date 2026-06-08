from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConfigPreset:
    name: str
    summary: str
    env: Mapping[str, str]
    notes: tuple[str, ...] = ()


_TEXT_COMMON: dict[str, str] = {
    "RAG_FLOW_INDEX_MODE": "text",
    "RAG_FLOW_RETRIEVAL_ROUTE_MODE": "text",
    "RAG_FLOW_RETRIEVAL_VISUAL_BONUS": "none",
    "RAG_FLOW_RRF_K": "10",
    "RAG_FLOW_VISUAL_WEIGHT": "2.5",
    "RAG_FLOW_RETRIEVAL_MIN_CANDIDATE_SCORE": "0",
    "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "1.0",
    "RAG_FLOW_RETRIEVAL_FINAL_OUTPUT_IMAGES": "0",
    "RAG_FLOW_ANSWER_MAX_TOKENS": "8000",
    "RAG_FLOW_ANSWER_ENABLE_THINKING": "0",
}


CONFIG_PRESETS: dict[str, ConfigPreset] = {
    "low": ConfigPreset(
        name="low",
        summary="Low-token text-only preset: k150/top10, ratio 0.4, 10k soft cap.",
        env={
            **_TEXT_COMMON,
            "RAG_FLOW_RETRIEVAL_K": "150",
            "RAG_FLOW_FINAL_TOP_K": "10",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
            "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "0.4",
        },
        notes=(
            "Keeps the tested 10k soft cap; the lower average context comes from top10 plus score-ratio pruning.",
            "Useful for previews, batch cost control, or high-volume support flows where a small quality trade-off is acceptable.",
        ),
    ),
    "medium": ConfigPreset(
        name="medium",
        summary="Conservative text baseline: k80/top20 with a 10k retrieved-context cap.",
        env={
            **_TEXT_COMMON,
            "RAG_FLOW_RETRIEVAL_K": "80",
            "RAG_FLOW_FINAL_TOP_K": "20",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
        },
        notes=(
            "Matches the safe text-only preset recommended for normal interactive use.",
            "Keeps ColPali disabled for low latency and simpler deployment.",
        ),
    ),
    "high": ConfigPreset(
        name="high",
        summary="High-recall text preset: k150/top80 with a 16k retrieved-context cap.",
        env={
            **_TEXT_COMMON,
            "RAG_FLOW_RETRIEVAL_K": "150",
            "RAG_FLOW_FINAL_TOP_K": "80",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "16000",
        },
        notes=(
            "Best 200-QA score among the recommended presets, but it costs more context and latency than medium.",
            "The 24k experiment is intentionally not shipped because it triggered many empty answers and missing usage records.",
        ),
    ),
    "low-with-image-input": ConfigPreset(
        name="low-with-image-input",
        summary="Low-token text retrieval plus image_url evidence in the answering payload.",
        env={
            **_TEXT_COMMON,
            "RAG_FLOW_RETRIEVAL_K": "150",
            "RAG_FLOW_FINAL_TOP_K": "10",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
            "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "0.4",
            "RAG_FLOW_RETRIEVAL_FINAL_OUTPUT_IMAGES": "1",
        },
        notes=(
            "Combines the low-token text context policy with retrieval-provided image evidence.",
            "This preset has not been separately validated by a 200-QA run; image tokens remain post-hoc LLM usage.",
        ),
    ),
    "medium-with-image-input": ConfigPreset(
        name="medium-with-image-input",
        summary="Medium text retrieval plus image_url evidence in the answering payload.",
        env={
            **_TEXT_COMMON,
            "RAG_FLOW_RETRIEVAL_K": "80",
            "RAG_FLOW_FINAL_TOP_K": "20",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
            "RAG_FLOW_RETRIEVAL_FINAL_OUTPUT_IMAGES": "1",
        },
        notes=(
            "This is a diagnostic preset for cases where the answer model must see evidence images.",
            "Image tokens are not included in the retrieval context budget; they only appear in post-hoc LLM usage.",
            "It is not the online default because the 200-QA run increased total tokens and latency without improving score.",
        ),
    ),
    "medium-with-visual-recall": ConfigPreset(
        name="medium-with-visual-recall",
        summary="Medium visual recall preset with ColPali route: naive page bonus, k150/top20, 10k cap.",
        env={
            "RAG_FLOW_RETRIEVAL_ROUTE_MODE": "text-visual-naive",
            "RAG_FLOW_RETRIEVAL_VISUAL_BONUS": "page-naive",
            "RAG_FLOW_RETRIEVAL_K": "150",
            "RAG_FLOW_FINAL_TOP_K": "20",
            "RAG_FLOW_RRF_K": "10",
            "RAG_FLOW_VISUAL_WEIGHT": "1.0",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
            "RAG_FLOW_RETRIEVAL_MIN_CANDIDATE_SCORE": "0",
            "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "1.0",
            "RAG_FLOW_RETRIEVAL_FINAL_OUTPUT_IMAGES": "0",
            "RAG_FLOW_RETRIEVAL_DEVICE": "auto",
            "RAG_FLOW_QUANTIZED_COLPALI": "1",
            "RAG_FLOW_INDEX_MODE": "both",
            "RAG_FLOW_INDEX_VISUAL_DPI": "200",
            "RAG_FLOW_INDEX_VISUAL_BATCH_SIZE": "8",
            "RAG_FLOW_ANSWER_MAX_TOKENS": "8000",
            "RAG_FLOW_ANSWER_ENABLE_THINKING": "0",
        },
        notes=(
            "Improves visual/UI evidence recall with a larger candidate pool and optional visual retrieval route.",
            "Requires a built visual index and loads the ColPali query encoder.",
        ),
    ),
}


PRESET_ALIASES: dict[str, str] = {
    "compact": "low",
    "default": "low",
    "high-recall": "high",
    "compact-with-image-input": "low-with-image-input",
    "default-with-image-input": "low-with-image-input",
    "visual-recall": "medium-with-visual-recall",
}


def preset_names() -> tuple[str, ...]:
    return tuple(CONFIG_PRESETS)


def resolve_preset_name(name: str) -> str:
    key = name.strip()
    if key in CONFIG_PRESETS:
        return key
    if key in PRESET_ALIASES:
        return PRESET_ALIASES[key]
    else:
        available = ", ".join(preset_names())
        raise ValueError(f"Unknown RAG Flow preset '{name}'. Available presets: {available}.")


def get_preset(name: str) -> ConfigPreset:
    return CONFIG_PRESETS[resolve_preset_name(name)]
