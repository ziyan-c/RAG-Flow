from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from .config import AppConfig, ModelServerConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _models_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _available_model_ids(base_url: str, api_key: str, *, timeout: float = 2.0) -> tuple[str, ...] | None:
    request = urllib.request.Request(_models_url(base_url))
    if api_key and api_key != "EMPTY":
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            import json

            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(f"Model endpoint {_models_url(base_url)} rejected the configured API key.") from exc
        return None
    except (OSError, urllib.error.URLError, ValueError):
        return None

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return ()
    model_ids = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))
    return tuple(model_ids)


def _base_url_port(base_url: str) -> int:
    parsed = urlparse(base_url)
    if parsed.port:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def _set_env(env: dict[str, str], key: str, value: str | Path | None) -> None:
    if value is None:
        return
    string_value = str(value)
    if string_value:
        env[key] = string_value


class _StartedServer:
    def __init__(
        self,
        *,
        kind: str,
        base_url: str,
        api_key: str,
        model: str,
        server_config: ModelServerConfig,
        process: subprocess.Popen[str],
        log_handle,
    ):
        self.kind = kind
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.server_config = server_config
        self.process = process
        self.log_handle = log_handle

    def wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.server_config.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"{self.kind.upper()} model server exited before serving {self.model}. "
                    f"Check log: {self.server_config.log_path or 'stdout'}"
                )
            available = _available_model_ids(self.base_url, self.api_key, timeout=2.0)
            if available is not None:
                if self.model in available:
                    return
                if available:
                    raise RuntimeError(
                        f"{self.kind.upper()} endpoint {self.base_url} is serving {', '.join(available)}, "
                        f"but this stage needs {self.model}."
                    )
            time.sleep(max(0.1, self.server_config.poll_interval))
        raise TimeoutError(
            f"Timed out waiting for {self.kind.upper()} model server to serve {self.model} at {self.base_url}."
        )

    def stop(self) -> None:
        if not self.server_config.stop_after or self.process.poll() is not None:
            self._close_log()
            return
        print(f"Stopping {self.kind.upper()} model server to free VRAM.")
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            self._close_log()
            return
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
        self._close_log()

    def _close_log(self) -> None:
        if self.log_handle:
            self.log_handle.close()
            self.log_handle = None


def _build_sglang_env(
    *,
    base_url: str,
    model: str,
    server_config: ModelServerConfig,
) -> dict[str, str]:
    env = os.environ.copy()
    env["RAG_FLOW_SGLANG_PORT"] = str(_base_url_port(base_url))
    env["RAG_FLOW_LLM_MODEL"] = model
    _set_env(env, "RAG_FLOW_SGLANG_MODEL_PROFILE", server_config.sglang_model_profile)
    _set_env(env, "RAG_FLOW_SGLANG_MODEL_ID", server_config.sglang_model_id or model)
    _set_env(env, "RAG_FLOW_SGLANG_MODEL_PATH", server_config.sglang_model_path)
    _set_env(env, "RAG_FLOW_SGLANG_SERVED_MODEL_NAME", server_config.sglang_served_model_name or model)
    _set_env(env, "RAG_FLOW_SGLANG_PYTHON", server_config.sglang_python)
    _set_env(env, "RAG_FLOW_SGLANG_LOCAL_MODEL_ROOT", server_config.sglang_local_model_root)
    _set_env(env, "RAG_FLOW_SGLANG_MEM_FRACTION_STATIC", server_config.sglang_mem_fraction_static)
    _set_env(env, "RAG_FLOW_SGLANG_CONTEXT_LENGTH", server_config.sglang_context_length)
    _set_env(env, "RAG_FLOW_SGLANG_TP_SIZE", server_config.sglang_tp_size)
    _set_env(env, "RAG_FLOW_SGLANG_QUANTIZATION", server_config.sglang_quantization)
    _set_env(env, "RAG_FLOW_SGLANG_REASONING_PARSER", server_config.sglang_reasoning_parser)
    _set_env(env, "RAG_FLOW_SGLANG_ATTENTION_BACKEND", server_config.sglang_attention_backend)
    _set_env(env, "RAG_FLOW_SGLANG_KV_CACHE_DTYPE", server_config.sglang_kv_cache_dtype)
    _set_env(env, "RAG_FLOW_SGLANG_EXTRA_ARGS", server_config.sglang_extra_args)
    return env


def _start_server(
    *,
    kind: str,
    base_url: str,
    api_key: str,
    model: str,
    server_config: ModelServerConfig,
) -> _StartedServer:
    repo_root = _repo_root()
    env = _build_sglang_env(base_url=base_url, model=model, server_config=server_config)
    log_handle = None
    stdout = None
    if server_config.log_path:
        server_config.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = server_config.log_path.open("a", encoding="utf-8")
        stdout = log_handle

    if server_config.command:
        command: str | list[str] = server_config.command
        shell = True
    else:
        command = [str(repo_root / "scripts" / "serve-llm-sglang.sh")]
        shell = False

    print(f"Starting {kind.upper()} model server for {model} at {base_url}.")
    if server_config.log_path:
        print(f"  log: {server_config.log_path}")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=env,
        shell=shell,
        stdout=stdout,
        stderr=subprocess.STDOUT if stdout is not None else None,
        text=True,
        start_new_session=True,
    )
    server = _StartedServer(
        kind=kind,
        base_url=base_url,
        api_key=api_key,
        model=model,
        server_config=server_config,
        process=process,
        log_handle=log_handle,
    )
    server.wait_until_ready()
    return server


@contextmanager
def managed_model_server(
    config: AppConfig,
    kind: str,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    enabled: bool = True,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    normalized_kind = kind.lower()
    if normalized_kind == "vlm":
        server_config = config.vlm_server
        resolved_base_url = base_url or config.models.vlm_base_url
        resolved_api_key = api_key if api_key is not None else config.models.vlm_api_key
        resolved_model = model or config.models.vlm_model
    elif normalized_kind == "llm":
        server_config = config.llm_server
        resolved_base_url = base_url or config.models.llm_base_url
        resolved_api_key = api_key if api_key is not None else config.models.llm_api_key
        resolved_model = model or config.models.llm_model
    else:
        raise ValueError("kind must be either 'vlm' or 'llm'")

    available = _available_model_ids(resolved_base_url, resolved_api_key, timeout=2.0)
    if available is not None:
        if resolved_model in available:
            yield
            return
        if server_config.auto_start:
            running = ", ".join(available) if available else "no models"
            raise RuntimeError(
                f"{normalized_kind.upper()} endpoint {resolved_base_url} is already reachable with {running}, "
                f"but this stage needs {resolved_model}."
            )
        yield
        return

    if not server_config.auto_start:
        yield
        return

    started = _start_server(
        kind=normalized_kind,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        model=resolved_model,
        server_config=server_config,
    )
    try:
        yield
    finally:
        started.stop()
