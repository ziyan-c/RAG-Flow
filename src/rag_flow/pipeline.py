from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import create_page_level_chunks, write_chunks
from .config import AppConfig
from .indexing import upsert_colpali_vectors, upsert_text_vectors
from .mineru import MinerUArtifacts, expected_content_json, find_content_json, infer_artifacts, run_mineru
from .preprocessing.image_descriptions import add_image_descriptions
from .preprocessing.small_icons import add_small_icon_text

STAGES = ("parsing", "patching", "captioning", "chunking", "indexing")


@dataclass(frozen=True)
class Stage:
    name: str
    output_path: Path | None
    action: Callable[[], None]


def _stage_slice(from_stage: str, to_stage: str) -> tuple[str, ...]:
    if from_stage not in STAGES:
        raise ValueError(f"Unknown from-stage {from_stage!r}. Choose one of: {', '.join(STAGES)}")
    if to_stage not in STAGES:
        raise ValueError(f"Unknown to-stage {to_stage!r}. Choose one of: {', '.join(STAGES)}")
    start = STAGES.index(from_stage)
    end = STAGES.index(to_stage)
    if start > end:
        raise ValueError("--from-stage cannot come after --to-stage")
    return STAGES[start : end + 1]


def _print_stage(stage_name: str, output_path: Path | None, *, dry_run: bool, skipped: bool) -> None:
    prefix = "DRY-RUN" if dry_run else "SKIP" if skipped else "RUN"
    if output_path:
        print(f"[{prefix}] {stage_name} -> {output_path}")
    else:
        print(f"[{prefix}] {stage_name}")


def run_ingest(
    config: AppConfig,
    *,
    pdf_path: str | Path | None = None,
    from_stage: str = "parsing",
    to_stage: str = "chunking",
    skip_existing: bool = True,
    force: bool = False,
    dry_run: bool = False,
    write_patching_view: bool = True,
    patching_view_pdf: str | Path | None = None,
    patch_max_new_tokens: int | None = None,
) -> MinerUArtifacts:
    selected_stages = _stage_slice(from_stage, to_stage)
    source_pdf = Path(pdf_path or config.mineru.input_path)
    source_name = config.paths.source_name if source_pdf == config.paths.source_pdf else source_pdf.name
    mineru_ran = False

    content_json = None
    if "parsing" not in selected_stages:
        content_json = infer_artifacts(config, source_pdf=source_pdf).content_json
    elif skip_existing and not force:
        content_json = find_content_json(config, source_pdf=source_pdf)

    if "parsing" in selected_stages and (force or not content_json):
        if dry_run:
            run_mineru(config, pdf_path=source_pdf, dry_run=True)
            content_json = expected_content_json(config, source_pdf=source_pdf)
        else:
            run_mineru(config, pdf_path=source_pdf)
            content_json = infer_artifacts(config, source_pdf=source_pdf).content_json
            mineru_ran = True

    artifacts = infer_artifacts(config, content_json=content_json, source_pdf=source_pdf)

    def patch_icons() -> None:
        add_small_icon_text(
            input_json=artifacts.content_json,
            output_json=artifacts.patched_json,
            pdf_path=source_pdf,
            model_name=config.models.vlm_model,
            model_revision=config.models.vlm_model_revision,
            trusted_remote_code_models=config.models.trusted_remote_code_models,
            max_new_tokens=config.patching.max_new_tokens if patch_max_new_tokens is None else patch_max_new_tokens,
            write_patching_view=write_patching_view,
            patching_view_pdf=patching_view_pdf,
        )

    def caption_images() -> None:
        add_image_descriptions(
            base_dir=artifacts.base_dir,
            input_json=artifacts.patched_json,
            output_json=artifacts.captioned_json,
            model_name=config.models.vlm_model,
            model_revision=config.models.vlm_model_revision,
            trusted_remote_code_models=config.models.trusted_remote_code_models,
        )

    def chunk_pages() -> None:
        chunks = create_page_level_chunks(artifacts.captioned_json, source_name)
        write_chunks(chunks, artifacts.chunks_json)
        print(f"Created {len(chunks)} page-level chunks at {artifacts.chunks_json}")

    stages = {
        "patching": Stage("patching", artifacts.patched_json, patch_icons),
        "captioning": Stage("captioning", artifacts.captioned_json, caption_images),
        "chunking": Stage("chunking", artifacts.chunks_json, chunk_pages),
        "indexing": Stage(
            "indexing",
            None,
            lambda: (
                upsert_text_vectors(config, artifacts.chunks_json),
                upsert_colpali_vectors(config, pdf_path=source_pdf, source_name=source_name),
            ),
        ),
    }

    for name in selected_stages:
        if name == "parsing":
            skipped = bool(content_json) and not force and not mineru_ran and not dry_run
            _print_stage(name, artifacts.content_json, dry_run=dry_run, skipped=skipped)
            continue

        stage = stages[name]
        skipped = bool(stage.output_path and stage.output_path.exists() and skip_existing and not force)
        _print_stage(stage.name, stage.output_path, dry_run=dry_run, skipped=skipped)
        if dry_run or skipped:
            continue
        stage.action()

    return artifacts


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Run the RAG Flow ingestion pipeline.")
    parser.add_argument("--pdf", default=str(config.mineru.input_path), help="Source PDF to parse and index.")
    parser.add_argument("--from-stage", choices=STAGES, default="parsing")
    parser.add_argument("--to-stage", choices=STAGES, default="chunking")
    parser.add_argument("--force", action="store_true", help="Re-run stages even when outputs already exist.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Run stages even when outputs already exist.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=config.patching.max_new_tokens,
        help="Patching VLM generation budget.",
    )
    parser.add_argument("--patching-view-pdf", help="Output PDF that visualizes patching crop regions.")
    parser.add_argument("--no-patching-view", action="store_true", help="Do not write the PATCHING_VIEW PDF.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pipeline without running it.")
    args = parser.parse_args(argv)

    run_ingest(
        config,
        pdf_path=args.pdf,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        skip_existing=not args.no_skip_existing,
        force=args.force,
        dry_run=args.dry_run,
        write_patching_view=not args.no_patching_view,
        patching_view_pdf=args.patching_view_pdf,
        patch_max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
