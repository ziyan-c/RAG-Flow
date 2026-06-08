from __future__ import annotations

from pathlib import Path, PurePosixPath

# Normalized marker for legacy source-pdfs/source_pdfs/sourcepdfs ancestors.
SOURCE_ROOT_MARKERS = {"sourcepdfs"}


def normalize_source_name(source_name: str | Path) -> str:
    normalized = str(source_name).replace("\\", "/").strip("/")
    return normalized or Path(source_name).name


def source_breadcrumb(source_name: str | Path, section_path: list[str] | tuple[str, ...] = ()) -> str:
    source_relpath = normalize_source_name(source_name)
    parts = [part for part in PurePosixPath(source_relpath).parts if part not in {"", "."}]
    parts.extend(str(item).strip() for item in section_path if str(item).strip())
    return " > ".join(parts)


def source_payload_fields(source_name: str | Path) -> dict[str, str]:
    source_relpath = normalize_source_name(source_name)
    path = PurePosixPath(source_relpath)
    fields = {
        "source_relpath": source_relpath,
        "source_filename": path.name,
        "breadcrumb": source_breadcrumb(source_relpath),
    }
    return fields


def source_root_from_input_path(input_path: str | Path) -> Path | None:
    path = Path(input_path).expanduser()
    if path.suffix.lower() == ".pdf":
        return None
    return path


def _relative_to_root(pdf_path: Path, root: Path) -> str | None:
    try:
        relpath = normalize_source_name(pdf_path.relative_to(root))
        return None if relpath == "." else relpath
    except ValueError:
        pass

    try:
        relpath = normalize_source_name(pdf_path.resolve().relative_to(root.resolve()))
        return None if relpath == "." else relpath
    except ValueError:
        return None


def _source_root_ancestors(pdf_path: Path) -> list[Path]:
    roots = []
    for parent in pdf_path.parents:
        marker = parent.name.lower().replace("-", "").replace("_", "")
        parent_marker = parent.parent.name.lower().replace("-", "").replace("_", "")
        if marker in SOURCE_ROOT_MARKERS or (marker == "source" and parent_marker == "pdfs"):
            roots.append(parent)
    return roots


def _same_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def source_name_for_pdf(
    pdf_path: str | Path,
    *,
    configured_source_pdf: str | Path | None = None,
    configured_source_name: str | None = None,
    source_root: str | Path | None = None,
) -> str:
    pdf = Path(pdf_path).expanduser()
    roots = []
    if source_root is not None:
        roots.append(Path(source_root).expanduser())
    roots.extend(_source_root_ancestors(pdf))

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        relpath = _relative_to_root(pdf, root)
        if relpath:
            return relpath

    if configured_source_pdf is not None and _same_path(pdf, Path(configured_source_pdf).expanduser()):
        return normalize_source_name(configured_source_name or pdf.name)

    return normalize_source_name(pdf.name)
