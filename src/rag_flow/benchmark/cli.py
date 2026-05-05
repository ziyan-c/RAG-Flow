from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args[:1] == ["patching"]:
        from rag_flow.benchmark import patching

        patching.main(raw_args[1:])
        return
    if raw_args[:1] == ["captioning"]:
        from rag_flow.benchmark import captioning

        captioning.main(raw_args[1:])
        return

    parser = argparse.ArgumentParser(prog="rag-flow benchmark", description="Run RAG Flow benchmark workflows.")
    subparsers = parser.add_subparsers(dest="benchmark_command", required=True)

    patching_parser = subparsers.add_parser("patching", help="Run patching parameter benchmark stages.")
    patching_parser.add_argument("args", nargs=argparse.REMAINDER)
    captioning_parser = subparsers.add_parser("captioning", help="Run captioning parameter benchmark stages.")
    captioning_parser.add_argument("args", nargs=argparse.REMAINDER)

    args = parser.parse_args(raw_args)
    if args.benchmark_command == "patching":
        from rag_flow.benchmark import patching

        patching.main(list(args.args))
        return
    if args.benchmark_command == "captioning":
        from rag_flow.benchmark import captioning

        captioning.main(list(args.args))
        return

    raise SystemExit(f"Unknown benchmark command: {args.benchmark_command}")
