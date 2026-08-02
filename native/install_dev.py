"""Copy the built native extension into the package for local development.

Cargo emits a `.dll` on Windows; Python only imports native modules named
`.pyd`. This copies and renames the release artifact into `rapidshot/` so
`import rapidshot.native` picks it up without installing anything.

    cd native && cargo build --release
    python native/install_dev.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / "rapidshot"

ARTIFACTS = {
    "win32": ("_rapidshot_native.dll", "_rapidshot_native.pyd"),
    "linux": ("lib_rapidshot_native.so", "_rapidshot_native.so"),
    "darwin": ("lib_rapidshot_native.dylib", "_rapidshot_native.so"),
}


def main() -> int:
    platform = "win32" if sys.platform.startswith("win") else sys.platform
    if platform not in ARTIFACTS:
        print(f"unsupported platform: {sys.platform}")
        return 1
    built_name, install_name = ARTIFACTS[platform]

    source = HERE / "target" / "release" / built_name
    if not source.exists():
        print(f"not found: {source}")
        print("build it first:  cd native && cargo build --release")
        return 1

    destination = PACKAGE / install_name
    shutil.copy2(source, destination)
    print(f"installed {destination.relative_to(PACKAGE.parent)} "
          f"({destination.stat().st_size / 1024:.0f} KB)")

    # Confirm it actually imports, rather than reporting success on a copy that
    # Python cannot load.
    sys.path.insert(0, str(PACKAGE.parent))
    try:
        from rapidshot import native
        if not native.is_available():
            print("copied, but the module still does not import:")
            print(f"  {native._import_error}")
            return 1
        print(f"verified: {native.build_info()}")
    except Exception as exc:  # pragma: no cover
        print(f"copied, but importing failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
