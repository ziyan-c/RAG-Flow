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
