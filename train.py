"""Canonical config-driven training entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path():
    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    sys.path.insert(0, str(src_dir))


def main():
    _bootstrap_src_path()
    from train_vstrong import main as train_main

    train_main()


if __name__ == "__main__":
    main()
