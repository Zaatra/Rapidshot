"""``python -m rapidshot <command>``, also installed as the ``rapidshot`` command.

Commands:

    benchmark   measure this machine the way the README's tables were measured
    diagnose    print what RapidShot can see: adapters, outputs, the native extension
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv else ""
    if command == "benchmark":
        # Imported here so that `import rapidshot` never pays for the harness.
        from rapidshot._bench.cli import main as benchmark
        return benchmark(argv)
    if command == "diagnose":
        import rapidshot
        print(rapidshot.diagnose())
        return 0
    print(__doc__.strip(), file=sys.stderr if command else sys.stdout)
    return 0 if command in ("", "-h", "--help", "help") else 2


if __name__ == "__main__":
    raise SystemExit(main())
