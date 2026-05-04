from __future__ import annotations

from rag_flow import cli


def test_mineru_command_delegates_to_module_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.mineru

    monkeypatch.setattr(rag_flow.mineru, "main", lambda argv: calls.append(argv))

    cli.main(["mineru", "doctor"])

    assert calls == [["doctor"]]


def test_patch_command_delegates_to_small_icon_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.small_icons

    monkeypatch.setattr(rag_flow.preprocessing.small_icons, "main", lambda argv: calls.append(argv))

    cli.main(["patch", "--artifact-dir", "hybrid_auto", "--dry-run"])

    assert calls == [["--artifact-dir", "hybrid_auto", "--dry-run"]]


def test_patch_command_reexecs_to_pipeline_python_when_configured(tmp_path, monkeypatch):
    env_python = tmp_path / "env" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    env_python.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    env_python.chmod(0o755)
    (env_python.parent / "rag-flow").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    calls = []

    monkeypatch.delenv("RAG_FLOW_ENV_REEXECED", raising=False)
    monkeypatch.setattr(cli, "_script_env", lambda: {"RAG_FLOW_PIPELINE_PYTHON_BIN": str(env_python)})
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    cli.main(["patch", "--dry-run"])

    assert calls
    command = calls[0][0][0]
    assert command == [str(env_python), "-m", "rag_flow.cli", "patch", "--dry-run"]
    assert calls[0][1]["env"]["RAG_FLOW_ENV_REEXECED"] == "1"


def test_patch_command_errors_when_pipeline_python_missing(tmp_path, monkeypatch):
    env_python = tmp_path / "env" / "bin" / "python"

    monkeypatch.delenv("RAG_FLOW_ENV_REEXECED", raising=False)
    monkeypatch.setattr(cli, "_script_env", lambda: {"RAG_FLOW_PIPELINE_PYTHON_BIN": str(env_python)})

    try:
        cli.main(["patch"])
    except SystemExit as exc:
        assert "rag-flow env create-pipeline" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for missing pipeline environment")


def test_script_env_prefers_repo_local_env_over_cwd(tmp_path, monkeypatch):
    repo_env = tmp_path / "repo" / ".local" / "rag-flow.env"
    repo_env.parent.mkdir(parents=True)
    repo_env.write_text("RAG_FLOW_PIPELINE_PYTHON_BIN=/repo/python\n", encoding="utf-8")
    cwd_env = tmp_path / "cwd" / ".local" / "rag-flow.env"
    cwd_env.parent.mkdir(parents=True)
    cwd_env.write_text("RAG_FLOW_PIPELINE_PYTHON_BIN=/cwd/python\n", encoding="utf-8")

    monkeypatch.chdir(cwd_env.parent.parent)
    monkeypatch.delenv("RAG_FLOW_ENV_FILE", raising=False)
    monkeypatch.delenv("RAG_FLOW_PIPELINE_PYTHON_BIN", raising=False)
    monkeypatch.setattr(cli, "REPO_ENV_FILE", repo_env)

    env = cli._script_env()

    assert env["RAG_FLOW_ENV_FILE"] == str(repo_env)
    assert env["RAG_FLOW_PIPELINE_PYTHON_BIN"] == "/repo/python"


def test_caption_command_delegates_to_image_description_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.image_descriptions

    monkeypatch.setattr(rag_flow.preprocessing.image_descriptions, "main", lambda argv: calls.append(argv))

    cli.main(["caption", "--dry-run"])

    assert calls == [["--dry-run"]]


def test_benchmark_command_delegates_to_benchmark_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.benchmark.cli

    monkeypatch.setattr(rag_flow.benchmark.cli, "main", lambda argv: calls.append(argv))

    cli.main(["benchmark", "patching", "dpi", "--dry-run"])

    assert calls == [["patching", "dpi", "--dry-run"]]


def test_caption_view_command_delegates_to_captioning_view_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.captioning_view

    monkeypatch.setattr(rag_flow.preprocessing.captioning_view, "main", lambda argv: calls.append(argv))

    cli.main(
        [
            "caption-view",
            "--input-json",
            "manual_content_list_PATCHED_CAPTIONED.json",
            "--input-pdf",
            "manual.pdf",
            "--dry-run",
        ]
    )

    assert calls == [
        ["--input-json", "manual_content_list_PATCHED_CAPTIONED.json", "--input-pdf", "manual.pdf", "--dry-run"]
    ]


def test_caption_view_artifact_dir_command_delegates_to_captioning_view_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.captioning_view

    monkeypatch.setattr(rag_flow.preprocessing.captioning_view, "main", lambda argv: calls.append(argv))

    cli.main(["caption-view", "--artifact-dir", "hybrid_auto", "--dry-run"])

    assert calls == [["--artifact-dir", "hybrid_auto", "--dry-run"]]


def test_patch_view_command_delegates_to_patching_view_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.patching_view

    monkeypatch.setattr(rag_flow.preprocessing.patching_view, "main", lambda argv: calls.append(argv))

    cli.main(["patch-view", "--input-json", "manual_content_list_PATCHED.json", "--input-pdf", "manual.pdf", "--dry-run"])

    assert calls == [
        ["--input-json", "manual_content_list_PATCHED.json", "--input-pdf", "manual.pdf", "--dry-run"]
    ]


def test_module_help_is_forwarded(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.mineru

    monkeypatch.setattr(rag_flow.mineru, "main", lambda argv: calls.append(argv))

    cli.main(["mineru", "--help"])

    assert calls == [["--help"]]


def test_init_china_sources_dry_run(capsys):
    cli.main(["init", "china-sources", "--dry-run"])

    output = capsys.readouterr().out
    assert "scripts/init/china-source.sh" in output


def test_init_china_all_dry_run(capsys):
    cli.main(["init", "china-all", "--dry-run"])

    output = capsys.readouterr().out
    assert "scripts/init/china-all.sh" in output


def test_env_create_mineru_dry_run(capsys):
    cli.main(["env", "create-mineru", "--dry-run"])

    output = capsys.readouterr().out
    assert "scripts/env/create-mineru.sh" in output


def test_env_install_uv_dry_run(capsys):
    cli.main(["env", "install-uv", "--dry-run"])

    output = capsys.readouterr().out
    assert "scripts/env/install-uv.sh" in output


def test_serve_llm_sglang_options_are_forwarded(monkeypatch):
    calls: list[tuple[tuple[str, ...], list[str], bool]] = []

    def fake_run_script(script_parts, args, *, dry_run):
        calls.append((tuple(script_parts), list(args), dry_run))

    monkeypatch.setattr(cli, "_run_script", fake_run_script)

    cli.main(
        [
            "serve",
            "llm-sglang",
            "--profile",
            "qwen3.6-35b-a3b-gptq-int4",
            "--port",
            "8090",
            "--served-model-name",
            "qwen-local",
            "--dry-run",
            "--",
            "--log-level",
            "warning",
        ]
    )

    assert calls == [
        (
            cli.SERVE_SCRIPTS["llm-sglang"],
            [
                "--dry-run",
                "--profile",
                "qwen3.6-35b-a3b-gptq-int4",
                "--served-model-name",
                "qwen-local",
                "--port",
                "8090",
                "--log-level",
                "warning",
            ],
            False,
        )
    ]


def test_download_llm_options_are_forwarded(monkeypatch):
    calls: list[tuple[tuple[str, ...], list[str], bool]] = []

    def fake_run_script(script_parts, args, *, dry_run):
        calls.append((tuple(script_parts), list(args), dry_run))

    monkeypatch.setattr(cli, "_run_script", fake_run_script)

    cli.main(
        [
            "download",
            "llm",
            "--source",
            "hf",
            "--profile",
            "qwen3.6-35b-a3b-gptq-int4",
            "--model-path",
            "/models/qwen36",
            "--revision",
            "abc123",
            "--dry-run",
        ]
    )

    assert calls == [
        (
            cli.DOWNLOAD_SCRIPTS["llm"],
            [
                "--dry-run",
                "--source",
                "hf",
                "--profile",
                "qwen3.6-35b-a3b-gptq-int4",
                "--model-path",
                "/models/qwen36",
                "--revision",
                "abc123",
            ],
            False,
        )
    ]


def test_llm_group_command_is_removed():
    try:
        cli.main(["llm", "download", "--dry-run"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("llm command group should be removed")


def test_retriever_help_is_local_to_unified_cli(capsys):
    cli.main(["retriever", "--help"])

    output = capsys.readouterr().out
    assert "usage: rag-flow retriever" in output
    assert "--host HOST" in output


def test_preprocess_command_is_removed():
    try:
        cli.main(["preprocess", "icons", "--dry-run"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("preprocess command should be removed")
