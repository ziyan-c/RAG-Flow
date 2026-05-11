from __future__ import annotations

from pathlib import Path


def _existing_model_dir(candidate: Path) -> Path | None:
    candidate = candidate.expanduser()
    if not candidate.is_dir():
        return None
    if (candidate / "config.json").is_file():
        return candidate

    snapshots_dir = candidate / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if (snapshot / "config.json").is_file():
                return snapshot
        if snapshots:
            return snapshots[0]

    return candidate


def _local_model_candidates(model_id: str, local_root: Path) -> list[Path]:
    basename = model_id.rsplit("/", 1)[-1]
    escaped = model_id.replace("/", "--")
    candidates = [
        local_root / model_id,
        local_root / basename,
        local_root / escaped,
        local_root / f"models--{escaped}",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_model_location(
    model_id_or_path: str,
    *,
    explicit_path: str | Path | None = None,
    local_root: str | Path | None = None,
) -> str:
    """Prefer local model directories, then fall back to the original model id."""
    if explicit_path:
        explicit = Path(explicit_path).expanduser()
        resolved = _existing_model_dir(explicit)
        if resolved is None:
            raise FileNotFoundError(f"Configured model path does not exist: {explicit}")
        return str(resolved)

    configured_path = Path(model_id_or_path).expanduser()
    resolved = _existing_model_dir(configured_path)
    if resolved is not None:
        return str(resolved)

    if local_root:
        root = Path(local_root).expanduser()
        for candidate in _local_model_candidates(model_id_or_path, root):
            resolved = _existing_model_dir(candidate)
            if resolved is not None:
                return str(resolved)

    return model_id_or_path
