from __future__ import annotations

import argparse
import importlib.metadata
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AppConfig

PINNED_MINERU_VERSION = "3.0.9"
MINERU_PYTHON_MIN = (3, 10)
MINERU_PYTHON_MAX = (3, 14)


@dataclass(frozen=True)
class MinerUStatus:
    command: str
    command_path: str | None
    package: str
    package_version: str | None
    python: str
    python_version: str
    python_supported: bool

    @property
    def installed(self) -> bool:
        return bool(self.command_path or self.package_version)


@dataclass(frozen=True)
class MinerUArtifacts:
    base_dir: Path
    content_json: Path
    patched_json: Path
    captioned_json: Path
    chunks_json: Path


@dataclass(frozen=True)
class MinerUBatchItem:
    input_pdf: Path
    output_dir: Path


def mineru_python(config: AppConfig) -> str:
    return config.mineru.python or sys.executable


def mineru_install_spec(config: AppConfig) -> str:
    extra = f"[{config.mineru.extra}]" if config.mineru.extra else ""
    version = f"=={config.mineru.version}" if config.mineru.version else ""
    return f"{config.mineru.package}{extra}{version}"


def _python_version(python: str) -> tuple[str, bool]:
    if python == sys.executable:
        version_info = sys.version_info
        version = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        return version, MINERU_PYTHON_MIN <= version_info[:2] < MINERU_PYTHON_MAX

    try:
        result = subprocess.run(
            [
                python,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown", False

    version = result.stdout.strip()
    try:
        major, minor, *_ = (int(part) for part in version.split("."))
    except ValueError:
        return version or "unknown", False
    return version, MINERU_PYTHON_MIN <= (major, minor) < MINERU_PYTHON_MAX


def _package_version(python: str, package: str) -> str | None:
    if python == sys.executable:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            return None

    code = (
        "import importlib.metadata, sys\n"
        f"package = {package!r}\n"
        "try:\n"
        "    print(importlib.metadata.version(package))\n"
        "except importlib.metadata.PackageNotFoundError:\n"
        "    sys.exit(1)\n"
    )
    try:
        result = subprocess.run(
            [python, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _first_command_token(command: str) -> str:
    try:
        return shlex.split(command)[0]
    except (IndexError, ValueError):
        return command


def _command_path(command: str, python: str) -> str | None:
    path = Path(command).expanduser()
    if path.parent != Path(".") and path.exists():
        return str(path)

    python_bin = str(Path(python).expanduser().parent)
    search_path = os.pathsep.join([python_bin, os.environ.get("PATH", "")])
    return shutil.which(command, path=search_path)


def mineru_status(config: AppConfig) -> MinerUStatus:
    command = _first_command_token(config.mineru.command)
    python = mineru_python(config)
    command_path = _command_path(command, python)
    python_version, python_supported = _python_version(python)
    package_version = _package_version(python, config.mineru.package)

    return MinerUStatus(
        command=command,
        command_path=command_path,
        package=config.mineru.package,
        package_version=package_version,
        python=python,
        python_version=python_version,
        python_supported=python_supported,
    )


def format_status(status: MinerUStatus, config: AppConfig) -> str:
    lines = [
        "MinerU status:",
        f"  command: {status.command_path or status.command + ' (not found)'}",
        f"  package: {status.package} {status.package_version or '(not installed in current Python)'}",
        f"  install spec: {mineru_install_spec(config)}",
        f"  python: {status.python} ({status.python_version})",
        f"  python supported for MinerU pin: {'yes' if status.python_supported else 'no; use Python 3.10-3.13'}",
        f"  input path: {config.mineru.input_path}",
        f"  output dir: {config.mineru.output_dir}",
        f"  backend: {config.mineru.backend or '(default)'}",
        f"  model source: {config.mineru.model_source or '(default)'}",
    ]
    if config.mineru.lang:
        lines.append(f"  lang: {config.mineru.lang}")
    return "\n".join(lines)


def build_mineru_command(
    config: AppConfig,
    *,
    pdf_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> list[str]:
    pdf = Path(pdf_path or config.mineru.input_path)
    output = Path(output_dir or config.mineru.output_dir)
    command = config.mineru.command
    context = {
        "pdf": str(pdf),
        "input_path": str(pdf),
        "source_pdf": str(pdf),
        "output_dir": str(output),
        "backend": config.mineru.backend,
        "model_source": config.mineru.model_source,
        "lang": config.mineru.lang,
    }

    if "{" in command and "}" in command:
        return shlex.split(command.format(**context))

    args = [*shlex.split(command), "-p", str(pdf), "-o", str(output)]
    if config.mineru.backend:
        args.extend(["-b", config.mineru.backend])
    if config.mineru.lang:
        args.extend(["-l", config.mineru.lang])
    if config.mineru.extra_args:
        args.extend(shlex.split(config.mineru.extra_args))
    return args


def iter_input_pdfs(input_path: str | Path, *, recursive: bool = True) -> list[Path]:
    path = Path(input_path).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"MinerU input file must be a PDF: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"MinerU input path does not exist: {path}")

    candidates = path.rglob("*") if recursive else path.glob("*")
    pdfs = [candidate for candidate in candidates if candidate.is_file() and candidate.suffix.lower() == ".pdf"]
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found under MinerU input directory: {path}")
    return sorted(pdfs, key=lambda item: str(item.relative_to(path)).lower())


def mineru_batch_items(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = True,
) -> list[MinerUBatchItem]:
    input_root = Path(input_path).expanduser()
    output_root = Path(output_dir).expanduser()
    pdfs = iter_input_pdfs(input_root, recursive=recursive)
    if input_root.is_file():
        return [MinerUBatchItem(input_pdf=pdfs[0], output_dir=output_root)]

    items = []
    for pdf in pdfs:
        relative_parent = pdf.relative_to(input_root).parent
        items.append(MinerUBatchItem(input_pdf=pdf, output_dir=output_root / relative_parent))
    return items


def _subprocess_env(config: AppConfig) -> dict[str, str] | None:
    if not config.mineru.model_source:
        return None
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = config.mineru.model_source
    return env


def _format_command(command: list[str], config: AppConfig) -> str:
    prefix = []
    if config.mineru.model_source:
        prefix.append(f"MINERU_MODEL_SOURCE={config.mineru.model_source}")
    return " ".join(shlex.quote(part) for part in [*prefix, *command])


def install_mineru(config: AppConfig, *, dry_run: bool = False, force: bool = False) -> list[str]:
    python = mineru_python(config)
    _, python_supported = _python_version(python)
    if not force and not python_supported:
        raise RuntimeError(
            "The pinned MinerU package supports Python >=3.10,<3.14. "
            "Create a Python 3.12 environment and set RAG_FLOW_MINERU_PYTHON to its python executable, "
            "or rerun with --force if you know your install target is compatible."
        )

    command = [python, "-m", "pip", "install", mineru_install_spec(config)]
    if dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return command
    subprocess.run(command, check=True)
    return command


def run_mineru(
    config: AppConfig,
    *,
    pdf_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    dry_run: bool = False,
) -> list[str]:
    status = mineru_status(config)
    if not dry_run and not status.command_path:
        if config.mineru.auto_install:
            install_mineru(config)
            status = mineru_status(config)
            if not status.command_path:
                raise RuntimeError(
                    f"Installed {mineru_install_spec(config)}, but MinerU command "
                    f"{status.command!r} is still not on PATH. Set RAG_FLOW_MINERU_COMMAND explicitly."
                )
        else:
            raise RuntimeError(
                "MinerU command is not available. Run `rag-flow mineru setup` first, "
                "or set RAG_FLOW_MINERU_COMMAND to an existing MinerU CLI."
            )

    command = build_mineru_command(config, pdf_path=pdf_path, output_dir=output_dir)
    if status.command_path:
        command[0] = status.command_path
    if dry_run:
        print(_format_command(command, config))
        return command
    subprocess.run(command, check=True, env=_subprocess_env(config))
    return command


def run_mineru_batch(
    config: AppConfig,
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    recursive: bool = True,
    dry_run: bool = False,
) -> list[list[str]]:
    source = Path(input_path or config.mineru.input_path)
    output_root = Path(output_dir or config.mineru.output_dir)
    items = mineru_batch_items(source, output_root, recursive=recursive)
    commands = []
    for item in items:
        if not dry_run:
            item.output_dir.mkdir(parents=True, exist_ok=True)
        if len(items) > 1:
            print(f"MinerU input: {item.input_pdf}")
            print(f"MinerU output dir: {item.output_dir}")
        commands.append(
            run_mineru(
                config,
                pdf_path=item.input_pdf,
                output_dir=item.output_dir,
                dry_run=dry_run,
            )
        )
    return commands


def expected_content_json(config: AppConfig, *, source_pdf: str | Path | None = None) -> Path:
    stem = Path(source_pdf or config.mineru.input_path).stem
    return Path(config.mineru.output_dir) / stem / "auto" / f"{stem}_content_list.json"


def _content_json_score(path: Path, source_pdf: str | Path | None) -> int | None:
    if source_pdf is None:
        return 0

    stem = Path(source_pdf).stem
    if path.name == f"{stem}_content_list.json":
        return 0
    if path.name == "content_list.json" and stem in path.parts:
        return 1
    return None


def _content_json_candidates(root: Path, *, source_pdf: str | Path | None = None) -> list[Path]:
    if not root.exists():
        return []
    candidates = []
    for path in root.rglob("*content_list.json"):
        if path.name.endswith("_content_list.json") or path.name == "content_list.json":
            if "small-icon" not in path.name and "caption" not in path.name:
                score = _content_json_score(path, source_pdf)
                if score is not None:
                    candidates.append(path)
    return sorted(
        candidates,
        key=lambda item: (_content_json_score(item, source_pdf) or 0, len(item.parts), str(item)),
    )


def find_content_json(
    config: AppConfig,
    *,
    search_root: str | Path | None = None,
    source_pdf: str | Path | None = None,
) -> Path | None:
    target_input = source_pdf or config.mineru.input_path
    if config.paths.content_json.exists() and _content_json_score(config.paths.content_json, target_input) is not None:
        return config.paths.content_json

    roots = []
    if search_root:
        roots.append(Path(search_root))
    roots.extend([config.paths.base_dir, config.mineru.output_dir])

    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates = _content_json_candidates(resolved, source_pdf=target_input)
        if candidates:
            return candidates[0]
    return None


def infer_artifacts(
    config: AppConfig,
    *,
    content_json: str | Path | None = None,
    source_pdf: str | Path | None = None,
) -> MinerUArtifacts:
    resolved_content = Path(content_json) if content_json else find_content_json(config, source_pdf=source_pdf)
    if resolved_content is None:
        raise FileNotFoundError(
            f"Cannot find MinerU content_list JSON. Expected {config.paths.content_json} "
            f"or a *content_list.json under {config.mineru.output_dir}."
        )

    resolved_content = resolved_content.expanduser()
    if resolved_content == config.paths.content_json:
        return MinerUArtifacts(
            base_dir=config.paths.base_dir,
            content_json=config.paths.content_json,
            patched_json=config.paths.patched_json,
            captioned_json=config.paths.captioned_json,
            chunks_json=config.paths.chunks_json,
        )

    base_dir = resolved_content.parent
    if resolved_content.name.endswith("_content_list.json"):
        prefix = resolved_content.name[: -len("_content_list.json")]
    elif resolved_content.name == "content_list.json":
        prefix = Path(source_pdf or config.mineru.input_path).stem
    else:
        prefix = resolved_content.stem

    return MinerUArtifacts(
        base_dir=base_dir,
        content_json=resolved_content,
        patched_json=base_dir / f"{prefix}_content_list_PATCHED.json",
        captioned_json=base_dir / f"{prefix}_content_list_PATCHED_CAPTIONED.json",
        chunks_json=base_dir / f"{prefix}_page_level_chunks.json",
    )


def main(argv: list[str] | None = None) -> None:
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Check, install, or run MinerU for RAG Flow.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="Check MinerU CLI/package availability.")

    setup_parser = subparsers.add_parser("setup", help="Install the pinned MinerU package.")
    setup_parser.add_argument("--dry-run", action="store_true")
    setup_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="Run MinerU on the configured input path.")
    run_parser.add_argument(
        "--input",
        "--pdf",
        dest="input_path",
        default=str(config.mineru.input_path),
        help="Input PDF or folder of PDFs for MinerU.",
    )
    run_parser.add_argument("--output-dir", default=str(config.mineru.output_dir))
    run_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When input is a folder, only parse PDFs directly inside it.",
    )
    run_parser.add_argument("--dry-run", action="store_true")

    locate_parser = subparsers.add_parser("locate", help="Locate MinerU output artifacts.")
    locate_parser.add_argument("--content-json")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        print(format_status(mineru_status(config), config))
    elif args.command == "setup":
        install_mineru(config, dry_run=args.dry_run, force=args.force)
    elif args.command == "run":
        run_mineru_batch(
            config,
            input_path=args.input_path,
            output_dir=args.output_dir,
            recursive=not args.no_recursive,
            dry_run=args.dry_run,
        )
    elif args.command == "locate":
        artifacts = infer_artifacts(config, content_json=args.content_json)
        print(f"base_dir={artifacts.base_dir}")
        print(f"content_json={artifacts.content_json}")
        print(f"patched_json={artifacts.patched_json}")
        print(f"captioned_json={artifacts.captioned_json}")
        print(f"chunks_json={artifacts.chunks_json}")


if __name__ == "__main__":
    main()
