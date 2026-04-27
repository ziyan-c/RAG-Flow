from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_soft_links_can_run_repeatedly(tmp_path):
    home = tmp_path / "home"
    data_disk = tmp_path / "data"
    cache = home / ".cache"
    local = home / ".local"
    cache.mkdir(parents=True)
    local.mkdir()
    (cache / "marker.txt").write_text("cache", encoding="utf-8")
    (local / "marker.txt").write_text("local", encoding="utf-8")

    env = os.environ.copy()
    env.update({"HOME": str(home), "RAG_FLOW_DATA_DISK": str(data_disk)})

    subprocess.run(["bash", str(ROOT / "scripts/init/soft-links.sh")], check=True, env=env)
    subprocess.run(["bash", str(ROOT / "scripts/init/soft-links.sh")], check=True, env=env)

    assert cache.is_symlink()
    assert local.is_symlink()
    assert (data_disk / ".cache" / "marker.txt").read_text(encoding="utf-8") == "cache"
    assert (data_disk / ".local" / "marker.txt").read_text(encoding="utf-8") == "local"


def test_cpu_cores_rewrites_single_bashrc_block(tmp_path):
    bashrc = tmp_path / ".bashrc"
    env = os.environ.copy()
    env.update(
        {
            "RAG_FLOW_INIT_BASHRC": str(bashrc),
            "RAG_FLOW_COMPILE_JOBS": "4",
            "RAG_FLOW_RUNTIME_THREADS": "2",
        }
    )
    subprocess.run(["bash", str(ROOT / "scripts/init/cpu-cores.sh")], check=True, env=env)

    env["RAG_FLOW_COMPILE_JOBS"] = "8"
    env["RAG_FLOW_RUNTIME_THREADS"] = "3"
    subprocess.run(["bash", str(ROOT / "scripts/init/cpu-cores.sh")], check=True, env=env)

    text = bashrc.read_text(encoding="utf-8")
    assert text.count("RAG Flow CPU threading") == 1
    assert "export MAX_JOBS=8" in text
    assert "export OMP_NUM_THREADS=3" in text
    assert "export MAX_JOBS=4" not in text


def test_china_source_rewrites_single_bashrc_block(tmp_path):
    bashrc = tmp_path / ".bashrc"
    env = os.environ.copy()
    env.update(
        {
            "RAG_FLOW_ENV_FILE": str(tmp_path / "missing.env"),
            "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "RAG_FLOW_INIT_REWRITE_APT": "0",
            "RAG_FLOW_INIT_CONFIGURE_LOCALE": "0",
            "RAG_FLOW_INIT_WRITE_CONDARC": "0",
            "RAG_FLOW_INIT_CONDA_CLEAN_INDEX": "0",
            "RAG_FLOW_INIT_MIRROR_PROBE": "0",
            "RAG_FLOW_INIT_WRITE_BASHRC": "1",
            "RAG_FLOW_INIT_BASHRC": str(bashrc),
            "RAG_FLOW_INIT_HF_ENDPOINT": "https://mirror-a.example",
        }
    )
    subprocess.run(["bash", str(ROOT / "scripts/init/china-source.sh")], check=True, env=env)

    env["RAG_FLOW_INIT_HF_ENDPOINT"] = "https://mirror-b.example"
    subprocess.run(["bash", str(ROOT / "scripts/init/china-source.sh")], check=True, env=env)

    text = bashrc.read_text(encoding="utf-8")
    assert text.count("RAG Flow AutoDL Environment") == 1
    assert "export HF_ENDPOINT=https://mirror-b.example" in text
    assert "export HF_ENDPOINT=https://mirror-a.example" not in text


def test_china_source_falls_back_to_tencent_for_managed_defaults(tmp_path):
    env_file = tmp_path / "rag-flow.env"
    condarc = tmp_path / ".condarc"
    env_file.write_text(
        "\n".join(
            [
                "RAG_FLOW_INIT_CONDA_MAIN_CHANNEL=https://mirrors.aliyun.com/anaconda/pkgs/main",
                "RAG_FLOW_INIT_CONDA_R_CHANNEL=https://mirrors.aliyun.com/anaconda/pkgs/r",
                "RAG_FLOW_PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "RAG_FLOW_ENV_FILE": str(env_file),
            "RAG_FLOW_RUNTIME_ROOT": str(tmp_path / "runtime"),
            "RAG_FLOW_INIT_REWRITE_APT": "0",
            "RAG_FLOW_INIT_CONFIGURE_LOCALE": "0",
            "RAG_FLOW_INIT_WRITE_BASHRC": "0",
            "RAG_FLOW_INIT_WRITE_CONDARC": "1",
            "RAG_FLOW_INIT_CONDARC": str(condarc),
            "RAG_FLOW_INIT_CONDA_CLEAN_INDEX": "0",
            "RAG_FLOW_INIT_MIRROR_ORDER": "aliyun,tencent,tuna",
            "RAG_FLOW_INIT_MIRROR_PROBE": "0",
            "RAG_FLOW_INIT_MIRROR_FAIL_PROFILES": "aliyun",
        }
    )

    subprocess.run(["bash", str(ROOT / "scripts/init/china-source.sh")], check=True, env=env)

    condarc_text = condarc.read_text(encoding="utf-8")
    env_text = env_file.read_text(encoding="utf-8")
    assert "https://mirrors.cloud.tencent.com/anaconda/pkgs/main" in condarc_text
    assert "https://mirrors.aliyun.com/anaconda" not in condarc_text
    assert "RAG_FLOW_INIT_CONDA_MAIN_CHANNEL=https://mirrors.cloud.tencent.com/anaconda/pkgs/main" in env_text
    assert "RAG_FLOW_PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple/" in env_text
