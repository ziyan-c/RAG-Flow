from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ConfigPreset:
    name: str
    summary: str
    env: Mapping[str, str]
    aliases: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


_TEXT_DIRECT_COMMON: dict[str, str] = {
    "RAG_FLOW_RETRIEVAL_ENABLE_VISUAL": "0",
    "RAG_FLOW_RETRIEVAL_ROUTE_MODE": "text",
    "RAG_FLOW_RETRIEVAL_CANDIDATE_MODE": "direct",
    "RAG_FLOW_RETRIEVAL_K": "150",
    "RAG_FLOW_FINAL_TOP_K": "80",
    "RAG_FLOW_RRF_K": "10",
    "RAG_FLOW_VISUAL_WEIGHT": "2.5",
    "RAG_FLOW_RETRIEVAL_MIN_CANDIDATE_SCORE": "0",
    "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "1.0",
}


CONFIG_PRESETS: dict[str, ConfigPreset] = {
    "default": ConfigPreset(
        name="default",
        summary="Text-only online preset with a 10k retrieved-context cap.",
        env={
            **_TEXT_DIRECT_COMMON,
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "10000",
        },
        aliases=("online", "safe-default"),
        notes=(
            "Matches the safe text-only default recommended for normal interactive use.",
            "Keeps ColPali disabled for low latency and simpler deployment.",
        ),
    ),
    "precise": ConfigPreset(
        name="precise",
        summary="Text-only precise preset with a 5k retrieved-context cap.",
        env={
            **_TEXT_DIRECT_COMMON,
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "5000",
        },
        aliases=("compact", "text-5k"),
        notes=(
            "Uses the same text retrieval backbone as default with a smaller answer context.",
            "Useful when precision, latency, or downstream LLM input cost matters more than broad evidence coverage.",
        ),
    ),
    "tiny": ConfigPreset(
        name="tiny",
        summary="Text-only tiny preset with a 3k retrieved-context cap.",
        env={
            **_TEXT_DIRECT_COMMON,
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "3000",
        },
        aliases=("smoke", "text-3k"),
        notes=(
            "Intended for smoke tests, cheap routing probes, and very short answers.",
            "May miss multi-chunk evidence; use precise/default/enhanced for normal answer quality.",
        ),
    ),
    "enhanced": ConfigPreset(
        name="enhanced",
        summary="Text-only long-answer preset with a 16k retrieved-context cap.",
        env={
            **_TEXT_DIRECT_COMMON,
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "16000",
        },
        aliases=("balanced", "long-answer", "text-16k"),
        notes=(
            "Uses the same text retrieval backbone as default with more answer context.",
            "Useful when Qwen3.6 can consume more evidence and latency should remain low.",
        ),
    ),
    "high-recall": ConfigPreset(
        name="high-recall",
        summary="Offline text-only review preset with a 24k retrieved-context cap.",
        env={
            **_TEXT_DIRECT_COMMON,
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "24000",
        },
        aliases=("offline", "review"),
        notes=(
            "Intended for offline review or difficult queries where context volume is acceptable.",
            "Keeps min_score_ratio=1.0; the soft token cap controls context length without relative pruning.",
        ),
    ),
    "visual-route": ConfigPreset(
        name="visual-route",
        summary="Optional ColPali visual route preset with a 16k retrieved-context cap.",
        env={
            "RAG_FLOW_RETRIEVAL_ENABLE_VISUAL": "1",
            "RAG_FLOW_RETRIEVAL_ROUTE_MODE": "visual-naive",
            "RAG_FLOW_RETRIEVAL_CANDIDATE_MODE": "visual-page-local-naive",
            "RAG_FLOW_RETRIEVAL_K": "150",
            "RAG_FLOW_FINAL_TOP_K": "80",
            "RAG_FLOW_RRF_K": "10",
            "RAG_FLOW_VISUAL_WEIGHT": "2.5",
            "RAG_FLOW_RETRIEVAL_MAX_CONTEXT_TOKENS": "16000",
            "RAG_FLOW_RETRIEVAL_MIN_CANDIDATE_SCORE": "0",
            "RAG_FLOW_RETRIEVAL_MIN_SCORE_RATIO": "1.0",
            "RAG_FLOW_RETRIEVAL_DEVICE": "auto",
            "RAG_FLOW_QUANTIZED_COLPALI": "1",
            "RAG_FLOW_INDEX_VISUAL_DPI": "200",
            "RAG_FLOW_INDEX_VISUAL_BATCH_SIZE": "8",
        },
        aliases=("visualroute", "visual", "colpali"),
        notes=(
            "Matches the latest thesis visual-optional recommendation, not the default.",
            "Requires a built visual index and loads the ColPali query encoder.",
        ),
    ),
}


PRESET_ALIASES: dict[str, str] = {
    alias: name
    for name, preset in CONFIG_PRESETS.items()
    for alias in (name, *preset.aliases)
}


def preset_names() -> tuple[str, ...]:
    return tuple(CONFIG_PRESETS)


def resolve_preset_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    try:
        return PRESET_ALIASES[key]
    except KeyError as exc:
        available = ", ".join(preset_names())
        raise ValueError(f"Unknown RAG Flow preset '{name}'. Available presets: {available}.") from exc


def get_preset(name: str) -> ConfigPreset:
    return CONFIG_PRESETS[resolve_preset_name(name)]
