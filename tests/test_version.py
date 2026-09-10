"""The version number is declared once, and everything else derives from it.

It used to be written out in three places -- `pyproject.toml`, `setup.py` and
`rapidshot/__init__.py`. The release workflow checked each against the git tag
but never against each other, so two could agree while the third drifted, and
nothing failed until a release was already being cut.

`rapidshot/_version.py` is now the only declaration. These tests fail if a
second one reappears, because the drift they guard against is invisible until
the moment it is expensive.
"""

import ast
import pathlib
import re

import rapidshot

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _declared_version() -> str:
    """Read `_version.py` the way the build backend does -- parsed, not imported."""
    tree = ast.parse((ROOT / "rapidshot" / "_version.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__"
                for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no __version__ assignment in rapidshot/_version.py")


def test_package_exports_the_declared_version():
    assert rapidshot.__version__ == _declared_version()


def test_version_module_stays_importable_without_the_package():
    """`_version.py` must not grow imports.

    `pyproject.toml` reads it through `[tool.setuptools.dynamic]`, which parses
    the file rather than importing it. An import here would still work at
    runtime and still build -- until the day the build machine cannot satisfy
    it, which is a release-day failure with no local reproduction.
    """
    tree = ast.parse((ROOT / "rapidshot" / "_version.py").read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not imports, "rapidshot/_version.py must stay import-free"


def test_pyproject_derives_the_version_rather_than_repeating_it():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in text
    assert 'attr = "rapidshot._version.__version__"' in text
    # A literal `version = "x.y.z"` under [project] is the drift this prevents.
    project = text.split("[project]", 1)[1].split("\n[", 1)[0]
    assert not re.search(r'^version\s*=\s*"', project, re.M)


def test_setup_py_derives_the_version_rather_than_repeating_it():
    text = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "version=_version()" in text
    assert not re.search(r'version\s*=\s*"\d+\.\d+', text)


def test_no_second_declaration_anywhere_in_the_package():
    """Exactly one file in `rapidshot/` may assign `__version__`."""
    declaring = [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "rapidshot").rglob("*.py")
        if re.search(r'^__version__\s*=\s*["\']', path.read_text(encoding="utf-8"), re.M)
    ]
    assert declaring == ["rapidshot/_version.py"], declaring
