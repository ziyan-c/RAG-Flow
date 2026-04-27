from __future__ import annotations

from rag_flow import cli


def test_mineru_command_delegates_to_module_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.mineru

    monkeypatch.setattr(rag_flow.mineru, "main", lambda argv: calls.append(argv))

    cli.main(["mineru", "doctor"])

    assert calls == [["doctor"]]


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


def test_retriever_help_is_local_to_unified_cli(capsys):
    cli.main(["retriever", "--help"])

    output = capsys.readouterr().out
    assert "usage: rag-flow retriever" in output
    assert "--host HOST" in output


def test_preprocess_command_delegates_to_module_main(monkeypatch):
    calls: list[list[str]] = []

    import rag_flow.preprocessing.small_icons

    monkeypatch.setattr(rag_flow.preprocessing.small_icons, "main", lambda argv: calls.append(argv))

    cli.main(["preprocess", "icons", "--dry-run"])

    assert calls == [["--dry-run"]]
