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


def test_create_pipeline_writes_python_path(tmp_path):
    bin_dir = tmp_path / "bin"
    env_dir = tmp_path / "envs" / "rag-flow-pipeline"
    env_bin = env_dir / "bin"
    log_file = tmp_path / "commands.log"
    env_file = tmp_path / "rag-flow.env"
    bin_dir.mkdir()
    env_bin.mkdir(parents=True)
    _write_executable(
        bin_dir / "micromamba",
        "#!/usr/bin/env bash\nprintf 'micromamba|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(
        bin_dir / "uv",
        "#!/usr/bin/env bash\nprintf 'uv|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(env_bin / "python", "#!/usr/bin/env bash\nexit 0\n")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_ENV_ROOT": str(tmp_path / "envs"),
        "RAG_FLOW_PIPELINE_ENV": "rag-flow-pipeline",
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
    }

    subprocess.run([shutil.which("bash") or "bash", str(ROOT / "scripts/env/create-pipeline.sh")], check=True, env=env)

    log_text = log_file.read_text(encoding="utf-8")
    env_text = env_file.read_text(encoding="utf-8")
    assert f"uv|pip install --python {env_dir / 'bin' / 'python'} --index-url" in log_text
    assert f"uv|pip install --python {env_dir / 'bin' / 'python'} -e {ROOT}[retrieval,preprocess]" in log_text
    assert f"RAG_FLOW_PIPELINE_PYTHON_BIN={env_dir / 'bin' / 'python'}" in env_text


def test_create_llm_writes_python_path_for_sglang(tmp_path):
    bin_dir = tmp_path / "bin"
    env_dir = tmp_path / "envs" / "rag-flow-llm"
    env_bin = env_dir / "bin"
    log_file = tmp_path / "commands.log"
    env_file = tmp_path / "rag-flow.env"
    bin_dir.mkdir()
    env_bin.mkdir(parents=True)
    _write_executable(bin_dir / "python3", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        bin_dir / "micromamba",
        "#!/usr/bin/env bash\nprintf 'micromamba|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(
        bin_dir / "uv",
        "#!/usr/bin/env bash\nprintf 'uv|%s\\n' \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    _write_executable(env_bin / "python", "#!/usr/bin/env bash\nexit 0\n")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_ENV_ROOT": str(tmp_path / "envs"),
        "RAG_FLOW_LLM_ENV": "rag-flow-llm",
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
    }

    subprocess.run([shutil.which("bash") or "bash", str(ROOT / "scripts/env/create-llm.sh")], check=True, env=env)

    log_text = log_file.read_text(encoding="utf-8")
    env_text = env_file.read_text(encoding="utf-8")
    assert f"uv|pip install --python {env_dir / 'bin' / 'python'} sglang[all]" in log_text
    assert f"uv|pip install --python {env_dir / 'bin' / 'python'} nvidia-cudnn-cu12==9.16.0.29" in log_text
    assert f"RAG_FLOW_LLM_PYTHON_BIN={env_dir / 'bin' / 'python'}" in env_text
    assert f"RAG_FLOW_SGLANG_PYTHON={env_dir / 'bin' / 'python'}" in env_text


def test_serve_llm_sglang_profile_dry_run_uses_qwen36_path(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_PROFILE": "qwen3.6-35b-a3b-gptq-int4",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/serve-llm-sglang.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "SGLang profile: qwen3.6-35b-a3b-gptq-int4" in result.stdout
    assert "/root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "SGLang served model: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "--served-model-name palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "/envs/llm/bin/python -m sglang.launch_server" in result.stdout


def test_serve_llm_sglang_cli_profile_can_select_qwen35(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/serve-llm-sglang.sh"),
            "--dry-run",
            "--profile",
            "qwen3.5-35b-a3b-gptq-int4",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "SGLang profile: qwen3.5-35b-a3b-gptq-int4" in result.stdout
    assert "/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in result.stdout
    assert "SGLang served model: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in result.stdout


def test_serve_llm_sglang_cli_profile_overrides_stale_env_model_path(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_PATH": "/old/qwen3.5",
        "RAG_FLOW_SGLANG_SERVED_MODEL_NAME": "old-model",
    }

    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/serve-llm-sglang.sh"),
            "--dry-run",
            "--profile",
            "qwen3.6-35b-a3b-gptq-int4",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "/old/qwen3.5" not in result.stdout
    assert "old-model" not in result.stdout
    assert "/root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "SGLang served model: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout


def test_serve_llm_sglang_prefers_manual_model_root(tmp_path):
    manual_model = tmp_path / "models" / "palmfuture" / "Qwen3.6-35B-A3B-GPTQ-Int4"
    manual_model.mkdir(parents=True)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT": str(tmp_path / "models"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/serve-llm-sglang.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert f"SGLang model path: {manual_model}" in result.stdout
    assert f"--model-path {manual_model}" in result.stdout


def test_serve_llm_sglang_accepts_huggingface_cache_layout(tmp_path):
    snapshot = tmp_path / "models" / "models--palmfuture--Qwen3.6-35B-A3B-GPTQ-Int4" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT": str(tmp_path / "models"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/serve-llm-sglang.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert f"SGLang model path: {snapshot}" in result.stdout
    assert f"--model-path {snapshot}" in result.stdout


def test_llm_download_dry_run_uses_qwen36_profile(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_PROFILE": "qwen3.6-35b-a3b-gptq-int4",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/llm/download-sglang-model.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "LLM model profile: qwen3.6-35b-a3b-gptq-int4" in result.stdout
    assert "ModelScope model id: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "Local model path: /root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "/envs/llm/bin/python -c" in result.stdout


def test_llm_download_can_select_huggingface_source(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_PROFILE": "qwen3.6-35b-a3b-gptq-int4",
    }

    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/llm/download-sglang-model.sh"),
            "--dry-run",
            "--source",
            "hf",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Download source: Hugging Face" in result.stdout
    assert "Hugging Face model id: palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "Local model path: /root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout
    assert "from huggingface_hub import snapshot_download" in result.stdout


def test_llm_download_auto_dry_run_shows_modelscope_then_hf(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/llm/download-sglang-model.sh"), "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "Download source: auto" in result.stdout
    assert "Download order: ModelScope, then Hugging Face" in result.stdout
    assert result.stdout.index("ModelScope model id:") < result.stdout.index("Hugging Face model id:")


def test_llm_download_uses_manual_model_root_before_download(tmp_path):
    manual_model = tmp_path / "models" / "palmfuture" / "Qwen3.6-35B-A3B-GPTQ-Int4"
    env_file = tmp_path / "rag-flow.env"
    manual_model.mkdir(parents=True)
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT": str(tmp_path / "models"),
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/llm/download-sglang-model.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    env_text = env_file.read_text(encoding="utf-8")
    assert "Existing model found: Manual local model" in result.stdout
    assert f"Local model path: {manual_model}" in result.stdout
    assert "Trying download source:" not in result.stdout
    assert f"RAG_FLOW_SGLANG_MODEL_PATH={manual_model}" in env_text


def test_llm_download_uses_huggingface_cache_layout_before_download(tmp_path):
    snapshot = tmp_path / "models" / "models--palmfuture--Qwen3.6-35B-A3B-GPTQ-Int4" / "snapshots" / "abc123"
    env_file = tmp_path / "rag-flow.env"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT": str(tmp_path / "models"),
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
    }

    result = subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/llm/download-sglang-model.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    env_text = env_file.read_text(encoding="utf-8")
    assert "Existing model found: Manual local model" in result.stdout
    assert f"Local model path: {snapshot}" in result.stdout
    assert "Trying download source:" not in result.stdout
    assert f"RAG_FLOW_SGLANG_MODEL_PATH={snapshot}" in env_text


def test_llm_download_source_override_uses_source_specific_default_path(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_PATH": "/old/modelscope/path",
    }

    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/llm/download-sglang-model.sh"),
            "--dry-run",
            "--source",
            "hf",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "/old/modelscope/path" not in result.stdout
    assert "Local model path: /root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in result.stdout


def test_llm_download_auto_falls_back_to_huggingface(tmp_path):
    bin_dir = tmp_path / "bin"
    env_file = tmp_path / "rag-flow.env"
    log_file = tmp_path / "download.log"
    python_stub = bin_dir / "python"
    bin_dir.mkdir()
    _write_executable(
        python_stub,
        "#!/usr/bin/env bash\n"
        "if [[ \"$*\" == *\"import modelscope\"* || \"$*\" == *\"import huggingface_hub\"* ]]; then exit 0; fi\n"
        "printf 'download|%s|%s\\n' \"$RAG_FLOW_DOWNLOAD_MODEL_ID\" \"$RAG_FLOW_DOWNLOAD_LOCAL_DIR\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n"
        "if [[ \"$RAG_FLOW_DOWNLOAD_LOCAL_DIR\" == *modelscope* ]]; then exit 42; fi\n"
        "exit 0\n",
    )
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
        "RAG_FLOW_SGLANG_PYTHON": str(python_stub),
    }

    subprocess.run(
        [shutil.which("bash") or "bash", str(ROOT / "scripts/llm/download-sglang-model.sh")],
        check=True,
        env=env,
    )

    log_text = log_file.read_text(encoding="utf-8")
    env_text = env_file.read_text(encoding="utf-8")
    assert "/root/.cache/modelscope/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in log_text
    assert "/root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in log_text
    assert "RAG_FLOW_SGLANG_DOWNLOAD_SOURCE=hf" in env_text
    assert "RAG_FLOW_SGLANG_MODEL_PATH=/root/.cache/huggingface/hub/models/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4" in env_text


def test_llm_download_profile_overrides_stale_env_model_path_and_id(tmp_path):
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": "/usr/bin:/bin",
        "RAG_FLOW_ENV_FILE": str(tmp_path / "rag-flow.env"),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_SGLANG_PYTHON": "/envs/llm/bin/python",
        "RAG_FLOW_SGLANG_MODEL_ID": "old/model",
        "RAG_FLOW_SGLANG_MODEL_PATH": "/old/model",
        "RAG_FLOW_SGLANG_SERVED_MODEL_NAME": "old-served",
    }

    result = subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/llm/download-sglang-model.sh"),
            "--dry-run",
            "--profile",
            "qwen3.5-35b-a3b-gptq-int4",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "old/model" not in result.stdout
    assert "/old/model" not in result.stdout
    assert "old-served" not in result.stdout
    assert "ModelScope model id: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in result.stdout
    assert "Local model path: /root/.cache/modelscope/hub/models/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in result.stdout
    assert "Served model name: Qwen/Qwen3.5-35B-A3B-GPTQ-Int4" in result.stdout


def test_llm_download_writes_resolved_env_values(tmp_path):
    bin_dir = tmp_path / "bin"
    env_file = tmp_path / "rag-flow.env"
    log_file = tmp_path / "download.log"
    python_stub = bin_dir / "python"
    bin_dir.mkdir()
    _write_executable(
        python_stub,
        "#!/usr/bin/env bash\n"
        "printf 'python|%s|%s|%s\\n' \"$RAG_FLOW_DOWNLOAD_MODEL_ID\" \"$RAG_FLOW_DOWNLOAD_LOCAL_DIR\" \"$*\" >> \"$RAG_FLOW_TEST_COMMAND_LOG\"\n",
    )
    env = {
        "HOME": str(tmp_path / "home"),
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "RAG_FLOW_TEST_COMMAND_LOG": str(log_file),
        "RAG_FLOW_ENV_FILE": str(env_file),
        "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
        "RAG_FLOW_UPDATE_ENV_FILE": "1",
        "RAG_FLOW_SGLANG_PYTHON": str(python_stub),
    }

    subprocess.run(
        [
            shutil.which("bash") or "bash",
            str(ROOT / "scripts/llm/download-sglang-model.sh"),
            "--model-id",
            "owner/model",
            "--model-path",
            str(tmp_path / "models" / "owner" / "model"),
        ],
        check=True,
        env=env,
    )

    log_text = log_file.read_text(encoding="utf-8")
    env_text = env_file.read_text(encoding="utf-8")
    assert "python|owner/model|" in log_text
    assert "RAG_FLOW_SGLANG_DOWNLOAD_SOURCE=modelscope" in env_text
    assert f"RAG_FLOW_SGLANG_MODEL_ID=owner/model" in env_text
    assert f"RAG_FLOW_SGLANG_MODEL_PATH={tmp_path / 'models' / 'owner' / 'model'}" in env_text
    assert "RAG_FLOW_SGLANG_SERVED_MODEL_NAME=owner/model" in env_text
    assert "RAG_FLOW_LLM_MODEL=owner/model" in env_text
