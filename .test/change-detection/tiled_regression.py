"""Run the capture regression suite without inherited Python installations."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cases = ("exact_tiles", "tiled_capture", "capture_paths", "dirty_rects", "hdr_capture",
             "native_duplicator", "failure_paths", "memory_pool")
    sources = ["tests/"] if args.full else [f"tests/test_{name}.py" for name in cases]
    command = [sys.executable, "-E", "-X", "utf8", "-B", "-m", "pytest", "-q", "--tb=short",
               "-p", "no:cacheprovider", *sources]
    env = {key: value for key, value in os.environ.items() if key.upper() not in ("PYTHONHOME", "PYTHONPATH")}
    log = args.log
    with log.open("w", encoding="utf8") as out:
        result = subprocess.run(command, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=480)
    print(log.read_text(encoding="utf8", errors="replace"), end="")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
