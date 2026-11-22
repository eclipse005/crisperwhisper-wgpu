#!/usr/bin/env python3
"""Download official non-Pro CrisperWhisper 2.0 safetensors into a local cache.

Weights are NOT committed to the git repo.  Download is for local testing only;
the checkpoints remain under Nyra's non-commercial research license.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SIZES = ("large", "turbo", "medium", "small")
REPO = "nyralabs/CrisperWhisper2.0_{size}"


def main() -> int:
    cache = Path(os.environ.get(
        "HF_HOME",
        os.environ.get("CRISPERWHISPER_HF_HOME", "/workspace/crisperwhisper-run/hf-cache"),
    ))
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    from huggingface_hub import snapshot_download

    only = sys.argv[1:] or list(SIZES)
    for size in only:
        if size.endswith("_pro") or size not in SIZES:
            print(f"skip unsupported size: {size}", file=sys.stderr)
            continue
        repo_id = REPO.format(size=size)
        print(f"==> downloading {repo_id} into {cache}", flush=True)
        path = snapshot_download(repo_id=repo_id, cache_dir=str(cache / "hub") if False else None)
        # snapshot_download uses HF_HOME/hub by default
        print(f"    ok: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
