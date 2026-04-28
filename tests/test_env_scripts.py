from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_install_uv_uses_configured_pip_source(tmp_path):
    bin_dir = tmp_path / "bin"
    log_file = tmp_path / "pip.log"
    python_stub = bin_dir / "python3"
    uv_stub = bin_dir / "uv"
    bin_dir.mkdir()
    python_stub.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "printf '%s\\n' \"$PIP_INDEX_URL|$PIP_CACHE_DIR|$*\" >> \"$RAG_FLOW_TEST_PIP_LOG\"",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    uv_stub.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    uv_stub.chmod(0o755)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_PIP_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_FORCE_INSTALL_UV": "1",
        "RAG_FLOW_PIP_INDEX_URL": "https://mirror.example/simple/",
    }

    subprocess.run([shutil.which("bash") or "bash", str(ROOT / "scripts/env/install-uv.sh")], check=True, env=env)

    log_text = log_file.read_text(encoding="utf-8")
    assert "https://mirror.example/simple/" in log_text
    assert "-m pip install -U uv" in log_text


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _prepare_create_mineru_stubs(tmp_path: Path) -> tuple[Path, Path, Path]:
    bin_dir = tmp_path / "bin"
    env_dir = tmp_path / "envs" / "rag-flow-mineru"
    env_bin = env_dir / "bin"
    log_file = tmp_path / "commands.log"
    bin_dir.mkdir()
    env_bin.mkdir(parents=True)
    _write_executable(
        bin_dir / "python3",
        "#!/usr/bin/env bash\n"
        "printf 'python3|%s|%s|%s\\n' \"$PIP_INDEX_URL\" \"$PIP_CACHE_DIR\" \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(
        bin_dir / "micromamba",
        "#!/usr/bin/env bash\n"
        "printf 'micromamba|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(
        bin_dir / "uv",
        "#!/usr/bin/env bash\n"
        "printf 'uv|%s|%s|%s|%s|%s\\n' \"$PIP_INDEX_URL\" \"$UV_INDEX_URL\" \"$PIP_CACHE_DIR\" \"$UV_CACHE_DIR\" \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(
        env_bin / "python",
        "#!/usr/bin/env bash\n"
        "printf 'env-python|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(env_bin / "mineru", "#!/usr/bin/env bash\nexit 0\n")
    return bin_dir, env_dir, log_file


def test_create_mineru_installs_uv_by_default_and_uses_uv_for_pip(tmp_path):
    bin_dir, env_dir, log_file = _prepare_create_mineru_stubs(tmp_path)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_ENV_ROOT": str(tmp_path / "envs"),
        "RAG_FLOW_MINERU_ENV": "rag-flow-mineru",
        "RAG_FLOW_FORCE_INSTALL_UV": "1",
        "RAG_FLOW_UPDATE_ENV_FILE": "0",
        "RAG_FLOW_PIP_INDEX_URL": "https://mirror.example/simple/",
        "RAG_FLOW_UV_INDEX_URL": "https://mirror.example/simple/",
    }

    subprocess.run([shutil.which("bash") or "bash", str(ROOT / "scripts/env/create-mineru.sh")], check=True, env=env)

    log_text = log_file.read_text(encoding="utf-8")
    assert "python3|https://mirror.example/simple/" in log_text
    assert "-m pip install -U uv" in log_text
    assert f"uv|https://mirror.example/simple/|https://mirror.example/simple/" in log_text
    assert f"pip install --python {env_dir / 'bin' / 'python'} -e" in log_text


def test_create_mineru_can_disable_uv(tmp_path):
    bin_dir, env_dir, log_file = _prepare_create_mineru_stubs(tmp_path)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_ENV_ROOT": str(tmp_path / "envs"),
        "RAG_FLOW_MINERU_ENV": "rag-flow-mineru",
        "RAG_FLOW_USE_UV": "0",
        "RAG_FLOW_UPDATE_ENV_FILE": "0",
    }

    subprocess.run([shutil.which("bash") or "bash", str(ROOT / "scripts/env/create-mineru.sh")], check=True, env=env)

    log_text = log_file.read_text(encoding="utf-8")
    assert "python3|" not in log_text
    assert "uv|" not in log_text
    assert f"env-python|-m pip install -e {ROOT}[mineru]" in log_text
    assert env_dir.exists()
