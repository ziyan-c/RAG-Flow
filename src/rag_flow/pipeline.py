from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .chunking import create_chunks, write_chunks
from .config import AppConfig
from .indexing import upsert_colpali_vectors, upsert_text_vectors
from .mineru import MinerUArtifacts, expected_content_json, find_content_json, infer_artifacts, run_mineru
from .preprocessing.chunking_view import write_chunking_view_pdf
from .preprocessing.image_descriptions import add_image_descriptions
from .preprocessing.small_icons import add_small_icon_text
from .sectioning import write_sectioned_json

STAGES = ("parsing", "sectioning", "patching", "captioning", "chunking", "indexing")


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
    write_chunking_view: bool = True,
    chunking_view_pdf: str | Path | None = None,
    patch_max_new_tokens: int | None = None,
    patch_batch_size: int | None = None,
    patch_concurrency: int | None = None,
    patch_checkpoint_interval: int | None = None,
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

    def recover_sections() -> None:
        result = write_sectioned_json(
            input_json=artifacts.content_json,
            input_pdf=source_pdf,
            output_json=artifacts.sectioned_json,
            audit_json=artifacts.sectioning_audit_json,
        )
        print(
            f"Recovered {result.stats['section_event_count']} PDF outline sections at "
            f"{artifacts.sectioned_json}"
        )

    def patch_icons() -> None:
        add_small_icon_text(
            input_json=artifacts.sectioned_json,
            output_json=artifacts.patched_json,
            pdf_path=source_pdf,
            llm_base_url=config.models.llm_base_url,
            llm_api_key=config.models.llm_api_key,
            llm_model=config.models.llm_model,
            dpi=config.patching.dpi,
            batch_size=config.patching.batch_size if patch_batch_size is None else patch_batch_size,
            concurrency=config.patching.concurrency if patch_concurrency is None else patch_concurrency,
            max_new_tokens=config.patching.max_new_tokens if patch_max_new_tokens is None else patch_max_new_tokens,
            llm_timeout=config.patching.llm_timeout,
            page_window_size=config.patching.page_window_size,
            checkpoint_interval=(
                config.patching.checkpoint_interval
                if patch_checkpoint_interval is None
                else patch_checkpoint_interval
            ),
            write_patching_view=write_patching_view,
            patching_view_pdf=patching_view_pdf,
        )

    def caption_images() -> None:
        add_image_descriptions(
            base_dir=artifacts.base_dir,
            input_json=artifacts.patched_json,
            output_json=artifacts.captioned_json,
            pdf_path=source_pdf,
            model_name=config.models.llm_model,
            max_new_tokens=config.captioning.max_new_tokens,
            batch_size=config.captioning.batch_size,
            concurrency=config.captioning.concurrency,
            max_context_tokens=config.captioning.max_context_tokens,
            max_image_side=config.captioning.max_image_side,
            llm_base_url=config.models.llm_base_url,
            llm_api_key=config.models.llm_api_key,
            llm_timeout=config.captioning.llm_timeout,
            checkpoint_interval=config.captioning.checkpoint_interval,
        )

    def chunk_pages() -> None:
        chunks = create_chunks(
            artifacts.captioned_json,
            source_name,
            mode=config.chunking.mode,
            max_tokens=config.chunking.max_tokens,
            overlap_tokens=config.chunking.overlap_tokens,
            min_tokens=config.chunking.min_tokens,
        )
        write_chunks(chunks, artifacts.chunks_json)
        print(f"Created {len(chunks)} {config.chunking.mode} chunks at {artifacts.chunks_json}")
        if write_chunking_view and source_pdf.exists():
            stats = write_chunking_view_pdf(
                chunks_json=artifacts.chunks_json,
                pdf_path=source_pdf,
                output_pdf=chunking_view_pdf,
            )
            print(f"Generated chunking view PDF at {stats.output_pdf}")
            print(f"  overlays: {stats.region_count}")
            print(f"  chunks with overlays: {stats.chunks_with_regions}/{stats.chunk_count}")
        elif write_chunking_view:
            print(f"Skipping chunking view PDF because source PDF does not exist: {source_pdf}")

    stages = {
        "sectioning": Stage("sectioning", artifacts.sectioned_json, recover_sections),
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
        help="Patching LLM generation budget.",
    )
    parser.add_argument("--patch-batch-size", type=int, default=config.patching.batch_size)
    parser.add_argument("--patch-concurrency", type=int, default=config.patching.concurrency)
    parser.add_argument("--patch-checkpoint-interval", type=int, default=config.patching.checkpoint_interval)
    parser.add_argument("--patching-view-pdf", help="Output PDF that visualizes patching crop regions.")
    parser.add_argument("--no-patching-view", action="store_true", help="Do not write the PATCHING_VIEW PDF.")
    parser.add_argument("--chunking-view-pdf", help="Output PDF that visualizes final chunk regions.")
    parser.add_argument("--no-chunking-view", action="store_true", help="Do not write the CHUNKING_VIEW PDF.")
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
        write_chunking_view=not args.no_chunking_view,
        chunking_view_pdf=args.chunking_view_pdf,
        patch_max_new_tokens=args.max_new_tokens,
        patch_batch_size=args.patch_batch_size,
        patch_concurrency=args.patch_concurrency,
        patch_checkpoint_interval=args.patch_checkpoint_interval,
    )


if __name__ == "__main__":
    main()
