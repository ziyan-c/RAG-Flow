from __future__ import annotations

import argparse
import importlib
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import load_env_file


REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ENV_FILE = REPO_ROOT / ".local" / "rag-flow.env"

MODULE_COMMANDS: dict[str, tuple[str, bool]] = {
    "mineru": ("rag_flow.mineru", True),
    "patch": ("rag_flow.preprocessing.small_icons", True),
    "patch-view": ("rag_flow.preprocessing.patching_view", True),
    "caption": ("rag_flow.preprocessing.image_descriptions", True),
    "caption-view": ("rag_flow.preprocessing.captioning_view", True),
    "ingest": ("rag_flow.pipeline", True),
    "chunk": ("rag_flow.chunking", True),
    "index": ("rag_flow.indexing", True),
    "retriever": ("rag_flow.api", True),
    "test-retriever": ("rag_flow.retrieval_client", True),
    "chat": ("rag_flow.chat_cli", True),
    "agent-demo": ("rag_flow.agentic", False),
}

MODULE_HELP: dict[str, str] = {
    "mineru": "Check, install, locate, or run MinerU.",
    "patch": "Patch small icon text from a MinerU artifact directory.",
    "patch-view": "Draw patching LLM crop regions over a source PDF.",
    "caption": "Generate image descriptions for patched MinerU output.",
    "caption-view": "Draw captioning image targets and context blocks over a source PDF.",
    "ingest": "Run the staged ingestion pipeline.",
    "chunk": "Build page-level chunks from MinerU JSON.",
    "index": "Upsert or inspect Qdrant indexes.",
    "retriever": "Start the retrieval API service.",
    "test-retriever": "Query the retrieval API from the terminal.",
    "chat": "Start the terminal RAG chat client.",
    "agent-demo": "Run the tool-calling demo.",
}

MODULE_ENV_PYTHON: dict[str, str] = {
    "patch": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "patch-view": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "caption": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "caption-view": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "chunk": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "index": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "retriever": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "test-retriever": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "chat": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "agent-demo": "RAG_FLOW_PIPELINE_PYTHON_BIN",
    "ingest": "RAG_FLOW_PIPELINE_PYTHON_BIN",
}

COMMAND_HELP: dict[str, str] = {
    "retriever": """usage: rag-flow retriever [-h] [--host HOST] [--port PORT] [--reload]

Start the RAG Flow retrieval API.

options:
  -h, --help   show this help message and exit
  --host HOST
  --port PORT
  --reload
""",
}

INIT_SCRIPTS: dict[str, tuple[str, ...]] = {
    "china-all": ("scripts", "init", "china-all.sh"),
    "china-sources": ("scripts", "init", "china-source.sh"),
    "soft-links": ("scripts", "init", "soft-links.sh"),
    "cpu-cores": ("scripts", "init", "cpu-cores.sh"),
}

ENV_SCRIPTS = (
    "install-uv",
    "create-mineru",
    "create-pipeline",
    "create-llm",
    "create-all",
)

SERVE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "llm-sglang": ("scripts", "serve-llm-sglang.sh"),
}

DOWNLOAD_SCRIPTS: dict[str, tuple[str, ...]] = {
    "llm": ("scripts", "llm", "download-sglang-model.sh"),
}

REMOTE_SCRIPTS: dict[str, tuple[str, ...]] = {
    "ssh-autodl": ("scripts", "remote", "ssh-autodl.sh"),
}


def _script_path(*parts: str) -> Path:
    path = REPO_ROOT.joinpath(*parts)
    if not path.exists():
        raise SystemExit(f"Cannot find script: {path}")
    return path


def _script_env() -> dict[str, str]:
    env = os.environ.copy()
    env_file = os.environ.get("RAG_FLOW_ENV_FILE") or REPO_ENV_FILE
    if env_file:
        env.setdefault("RAG_FLOW_ENV_FILE", str(env_file))
        for key, value in load_env_file(env_file).items():
            env.setdefault(key, value)
    return env


def _print_command(command: Sequence[str]) -> None:
    print(" ".join(shlex.quote(part) for part in command))


def _run_script(script_parts: Sequence[str], args: Sequence[str], *, dry_run: bool) -> None:
    command = [str(_script_path(*script_parts)), *args]
    if dry_run:
        _print_command(command)
        return
    subprocess.run(command, check=True, env=_script_env())


def _run_module_main(module_name: str, args: Sequence[str], *, accepts_argv: bool) -> None:
    module = importlib.import_module(module_name)
    main = getattr(module, "main")
    if accepts_argv:
        main(list(args))
        return
    if args:
        raise SystemExit(f"{module_name}.main does not accept command arguments: {' '.join(args)}")
    main()


def _truthy(value: str | None) -> bool:
    return (value or "").lower() in {"1", "true", "yes", "on"}


def _same_executable(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _maybe_reexec_module(command_name: str, module_args: Sequence[str]) -> bool:
    if _truthy(os.environ.get("RAG_FLOW_DISABLE_ENV_REEXEC")):
        return False
    if os.environ.get("RAG_FLOW_ENV_REEXECED"):
        return False

    env_key = MODULE_ENV_PYTHON.get(command_name)
    if not env_key:
        return False

    env = _script_env()
    target_python = env.get(env_key, "").strip()
    if not target_python:
        return False

    target_path = Path(target_python).expanduser()
    if not target_path.exists():
        if "--dry-run" in module_args or "--help" in module_args or "-h" in module_args:
            return False
        raise SystemExit(
            f"{env_key} is configured as {target_path}, but that Python does not exist. "
            "Run `rag-flow env create-pipeline` first, then retry this command."
        )
    if not target_path.is_file():
        return False
    if not os.access(target_path, os.X_OK):
        raise SystemExit(f"{env_key} is configured as {target_path}, but it is not executable.")
    if not (target_path.parent / "rag-flow").exists() and command_name != "agent-demo":
        if "--dry-run" in module_args or "--help" in module_args or "-h" in module_args:
            return False
        raise SystemExit(
            f"{target_path} exists, but the pipeline environment does not appear to have rag-flow installed. "
            "Run `rag-flow env create-pipeline` first, then retry this command."
        )
    if _same_executable(target_path, Path(sys.executable)):
        return False

    run_env = env.copy()
    run_env["RAG_FLOW_ENV_REEXECED"] = "1"
    subprocess.run(
        [str(target_path), "-m", "rag_flow.cli", command_name, *module_args],
        check=True,
        env=run_env,
    )
    return True


def _dispatch_module(args: argparse.Namespace) -> None:
    if _maybe_reexec_module(args.command, args.args):
        return
    module_name, accepts_argv = MODULE_COMMANDS[args.command]
    _run_module_main(module_name, args.args, accepts_argv=accepts_argv)


def _dispatch_init(args: argparse.Namespace) -> None:
    _run_script(INIT_SCRIPTS[args.init_command], args.args, dry_run=args.dry_run)


def _dispatch_env(args: argparse.Namespace) -> None:
    _run_script(("scripts", "env", f"{args.env_command}.sh"), args.args, dry_run=args.dry_run)


def _dispatch_serve(args: argparse.Namespace) -> None:
    if args.serve_command == "llm-sglang":
        script_args: list[str] = []
        for attr, flag in (
            ("profile", "--profile"),
            ("model_path", "--model-path"),
            ("served_model_name", "--served-model-name"),
            ("port", "--port"),
            ("context_length", "--context-length"),
            ("mem_fraction_static", "--mem-fraction-static"),
            ("tp_size", "--tp-size"),
        ):
            value = getattr(args, attr, None)
            if value is not None:
                script_args.extend([flag, str(value)])
        passthrough_args = list(args.args)
        if passthrough_args[:1] == ["--"]:
            passthrough_args = passthrough_args[1:]
        script_args.extend(passthrough_args)
        if args.dry_run:
            script_args.insert(0, "--dry-run")
        _run_script(SERVE_SCRIPTS[args.serve_command], script_args, dry_run=False)
        return
    _run_script(SERVE_SCRIPTS[args.serve_command], args.args, dry_run=args.dry_run)


def _dispatch_download(args: argparse.Namespace) -> None:
    if args.download_command == "llm":
        script_args: list[str] = []
        for attr, flag in (
            ("source", "--source"),
            ("profile", "--profile"),
            ("model_id", "--model-id"),
            ("model_path", "--model-path"),
            ("served_model_name", "--served-model-name"),
            ("revision", "--revision"),
            ("python", "--python"),
        ):
            value = getattr(args, attr, None)
            if value is not None:
                script_args.extend([flag, str(value)])
        if args.dry_run:
            script_args.insert(0, "--dry-run")
        _run_script(DOWNLOAD_SCRIPTS[args.download_command], script_args, dry_run=False)
        return
    _run_script(DOWNLOAD_SCRIPTS[args.download_command], args.args, dry_run=args.dry_run)


def _dispatch_remote(args: argparse.Namespace) -> None:
    _run_script(REMOTE_SCRIPTS[args.remote_command], args.args, dry_run=args.dry_run)


def _add_passthrough_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)


def _add_script_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    aliases: Sequence[str] = (),
    help_text: str,
    handler,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, aliases=list(aliases), help=help_text)
    parser.add_argument("--dry-run", action="store_true", help="Print the script command without running it.")
    _add_passthrough_arguments(parser)
    parser.set_defaults(handler=handler)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-flow",
        description="Unified command line entrypoint for RAG Flow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, (module_name, _accepts_argv) in MODULE_COMMANDS.items():
        module_parser = subparsers.add_parser(
            command,
            help=MODULE_HELP[command],
            add_help=False,
        )
        module_parser.add_argument("-h", "--help", action="store_true", help=argparse.SUPPRESS)
        module_parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
        module_parser.set_defaults(handler=_dispatch_module)

    init_parser = subparsers.add_parser("init", help="Run local machine initialization helpers.")
    init_subparsers = init_parser.add_subparsers(dest="init_command", required=True)
    _add_script_command(
        init_subparsers,
        "china-all",
        help_text="Run all China machine initialization steps.",
        handler=_dispatch_init,
    )
    _add_script_command(
        init_subparsers,
        "china-sources",
        help_text="Configure China mirrors, caches, locale, and conda settings.",
        handler=_dispatch_init,
    )
    _add_script_command(
        init_subparsers,
        "soft-links",
        help_text="Move large home/cache directories to the runtime disk and symlink them back.",
        handler=_dispatch_init,
    )
    _add_script_command(
        init_subparsers,
        "cpu-cores",
        help_text="Write CPU build/runtime thread settings to ~/.bashrc.",
        handler=_dispatch_init,
    )

    env_parser = subparsers.add_parser("env", help="Create isolated Python environments.")
    env_subparsers = env_parser.add_subparsers(dest="env_command", required=True)
    for command in ENV_SCRIPTS:
        _add_script_command(
            env_subparsers,
            command,
            help_text=f"Run scripts/env/{command}.sh.",
            handler=_dispatch_env,
        )

    serve_parser = subparsers.add_parser("serve", help="Start long-running services.")
    serve_subparsers = serve_parser.add_subparsers(dest="serve_command", required=True)
    llm_parser = serve_subparsers.add_parser(
        "llm-sglang",
        help="Start the SGLang OpenAI-compatible LLM service.",
    )
    llm_parser.add_argument("--dry-run", action="store_true", help="Print the resolved SGLang command without running it.")
    llm_parser.add_argument("--profile", help="Known model profile, such as qwen3.6-35b-a3b-gptq-int4.")
    llm_parser.add_argument("--model-path", help="Local model path. Overrides the selected profile path.")
    llm_parser.add_argument("--served-model-name", help="Model name exposed by SGLang's OpenAI-compatible API.")
    llm_parser.add_argument("--port", help="SGLang listen port.")
    llm_parser.add_argument("--context-length", help="SGLang context length.")
    llm_parser.add_argument("--mem-fraction-static", help="SGLang static GPU memory fraction.")
    llm_parser.add_argument("--tp-size", help="Tensor parallel size.")
    _add_passthrough_arguments(llm_parser)
    llm_parser.set_defaults(handler=_dispatch_serve)

    download_parser = subparsers.add_parser("download", help="Download local model assets.")
    download_subparsers = download_parser.add_subparsers(dest="download_command", required=True)
    download_llm_parser = download_subparsers.add_parser(
        "llm",
        help="Download the configured SGLang model.",
    )
    download_llm_parser.add_argument("--dry-run", action="store_true", help="Print the download command without running it.")
    download_llm_parser.add_argument(
        "--source",
        choices=("auto", "modelscope", "hf", "huggingface"),
        help="Download source. Defaults to auto, which tries modelscope before Hugging Face.",
    )
    download_llm_parser.add_argument("--profile", help="Known model profile, such as qwen3.6-35b-a3b-gptq-int4.")
    download_llm_parser.add_argument("--model-id", "--model", dest="model_id", help="Model id to download.")
    download_llm_parser.add_argument(
        "--model-path",
        "--local-dir",
        dest="model_path",
        help="Local directory for the model files.",
    )
    download_llm_parser.add_argument("--served-model-name", help="Model name exposed by SGLang after download.")
    download_llm_parser.add_argument("--revision", help="Optional model revision or commit.")
    download_llm_parser.add_argument("--python", help="Python interpreter with, or to install, the downloader package.")
    download_llm_parser.set_defaults(handler=_dispatch_download)

    remote_parser = subparsers.add_parser("remote", help="Remote machine helpers.")
    remote_subparsers = remote_parser.add_subparsers(dest="remote_command", required=True)
    _add_script_command(
        remote_subparsers,
        "ssh-autodl",
        help_text="Open the configured AutoDL SSH session.",
        handler=_dispatch_remote,
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] in MODULE_COMMANDS:
        command = raw_args[0]
        module_args = raw_args[1:]
        if command in COMMAND_HELP and any(arg in {"-h", "--help"} for arg in module_args):
            print(COMMAND_HELP[command], end="")
            return
        if _maybe_reexec_module(command, module_args):
            return
        module_name, accepts_argv = MODULE_COMMANDS[command]
        _run_module_main(module_name, module_args, accepts_argv=accepts_argv)
        return

    parser = build_parser()
    args, unknown = parser.parse_known_args(raw_args)
    if unknown:
        if hasattr(args, "args"):
            args.args.extend(unknown)
        else:
            parser.error(f"unrecognized arguments: {' '.join(unknown)}")
    if getattr(args, "help", False):
        if args.command in COMMAND_HELP:
            print(COMMAND_HELP[args.command], end="")
            return
        args.args.insert(0, "--help")
    args.handler(args)


if __name__ == "__main__":
    main()
