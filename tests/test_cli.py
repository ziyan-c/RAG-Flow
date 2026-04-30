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


def test_caption_command_delegates_to_image_description_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.image_descriptions

    monkeypatch.setattr(rag_flow.preprocessing.image_descriptions, "main", lambda argv: calls.append(argv))

    cli.main(["caption", "--dry-run"])

    assert calls == [["--dry-run"]]


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
