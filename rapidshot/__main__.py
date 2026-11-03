"""``python -m rapidshot <command>``, also installed as the ``rapidshot`` command.

Commands:

    benchmark   measure this machine the way the README's tables were measured
    diagnose    print what RapidShot can see: adapters, outputs, the native extension
    plan        say which backend would capture a source here, and why
                  [--calibrate] [--monitor N | --window TITLE]
                  [--consumer cpu|torch|cupy] [--seconds S]
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
    if command == "plan":
        return _plan(argv)
    print(__doc__.strip(), file=sys.stderr if command else sys.stdout)
    return 0 if command in ("", "-h", "--help", "help") else 2


def _plan(argv) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="rapidshot plan")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--monitor", type=int, help="display index (default: primary)")
    target.add_argument("--window", help="capture the window whose title contains this")
    parser.add_argument("--consumer", choices=("cpu", "torch", "cupy"))
    parser.add_argument("--calibrate", action="store_true",
                        help="measure every backend now and cache the choice")
    parser.add_argument("--seconds", type=float, default=2.0,
                        help="how long to measure each backend")
    args = parser.parse_args(argv)

    import rapidshot

    source = (rapidshot.Window.find(args.window) if args.window
              else rapidshot.Monitor(args.monitor))
    print(rapidshot.plan(source, consumer=args.consumer, calibrate=args.calibrate,
                         seconds=args.seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
