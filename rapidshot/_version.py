"""The one place the version number is written.

It used to live in three: `pyproject.toml`, `setup.py` and
`rapidshot/__init__.py`. The release workflow checked each against the git tag
but never against each other, so two could agree and the third drift without
anything failing until a release was already being cut.

Deliberately free of imports and of anything but a literal assignment.
`pyproject.toml` reads it through `[tool.setuptools.dynamic]`, which parses this
file rather than importing it -- so the version stays readable at build time on
a machine where `import rapidshot` would fail for want of Windows COM.
"""

__version__ = "2.4.0"
