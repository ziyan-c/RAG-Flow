from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "source"
OUTPUT_ROOT = REPO_ROOT / ".local" / "CUSTOM_DATA" / "pdfs" / "output"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_dahua_metadata_layout import (  # noqa: E402
    _metadata_sidecar_path,
    _prune_empty_dirs,
    _rewrite_json_paths,
    _safe_move,
    _target_relpath,
    _write_metadata_sidecar,
)


def _output_dir_for_relpath(relpath: str) -> Path:
    pdf = SOURCE_ROOT / relpath
    return OUTPUT_ROOT / Path(relpath).parent / pdf.stem


def apply_decisions(decisions: list[dict]) -> None:
    for decision in decisions:
        source_relpath = decision["source_relpath"]
        metadata = decision["metadata"]
        metadata["filename"] = Path(source_relpath).name
        target_relpath = decision.get("target_relpath") or _target_relpath(metadata)

        current_pdf = SOURCE_ROOT / source_relpath
        target_pdf = SOURCE_ROOT / target_relpath
        current_sidecar = _metadata_sidecar_path(current_pdf)
        target_sidecar = _metadata_sidecar_path(target_pdf)
        current_output = _output_dir_for_relpath(source_relpath)
        target_output = _output_dir_for_relpath(target_relpath)

        if not current_pdf.exists():
            raise FileNotFoundError(current_pdf)
        if current_output.exists() and current_output != target_output:
            if target_output.exists():
                raise FileExistsError(target_output)
            _safe_move(current_output, target_output)
            for json_path in target_output.rglob("*.json"):
                _rewrite_json_paths(
                    json_path,
                    old_relpath=source_relpath,
                    new_relpath=target_relpath,
                    old_docdir=current_output,
                    new_docdir=target_output,
                )

        if current_pdf != target_pdf:
            _safe_move(current_pdf, target_pdf)
            if current_sidecar.exists() and current_sidecar != target_sidecar:
                if target_sidecar.exists():
                    target_sidecar.unlink()
                _safe_move(current_sidecar, target_sidecar)

        _write_metadata_sidecar(target_sidecar, source_relpath=target_relpath, metadata=metadata)

    _prune_empty_dirs(SOURCE_ROOT)
    _prune_empty_dirs(OUTPUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply human-reviewed Dahua PDF metadata sidecar decisions.")
    parser.add_argument("decision_json")
    args = parser.parse_args()
    decisions = json.loads(Path(args.decision_json).read_text(encoding="utf-8"))
    if not isinstance(decisions, list):
        raise ValueError("decision_json must contain a list")
    apply_decisions(decisions)
    print(f"applied_decisions={len(decisions)}")


if __name__ == "__main__":
    main()
