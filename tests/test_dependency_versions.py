"""Dependency version checks must compare numbers, not strings.

They compared strings, so `"12.3.0" < "9.0.0"` was True and every import on
Pillow 12 printed a warning that Pillow was too old.
"""

import pytest

from rapidshot.processor.base import version_below


@pytest.mark.parametrize("version, minimum, below", [
    ("12.3.0", "9.0.0", False),     # the Pillow 12 false alarm
    ("10.0.0", "4.5.0", False),     # OpenCV 10 would have warned the same way
    ("2.5.3", "1.20.0", False),
    ("14.2.0", "10.0.0", False),
    ("1.19.5", "1.20.0", True),
    ("8.4.0", "9.0.0", True),
    ("4.5.0", "4.5.0", False),      # equal is not below
    ("5.0.0.93", "4.5.0", False),   # opencv-python's four-part versions
    ("2.0.0rc1", "1.20.0", False),  # pre-release suffix ignored
    ("1.26.4+cpu", "1.20.0", False),  # local suffix ignored
    ("4.5", "4.5.0", False),        # two parts: 4.5 is 4.5.0, not older
    ("1.20", "1.20.0", False),
    ("4.5.0", "4.5", False),
    ("4", "4.5.0", True),
    ("4.4", "4.5.0", True),
])
def test_version_below_compares_numerically(version, minimum, below):
    assert version_below(version, minimum) is below


def test_installed_dependencies_get_no_false_warning(capsys):
    """Any warning printed must name a version genuinely below its minimum."""
    import re

    import rapidshot.processor.base as base

    # Call the check on a bare instance of whichever class owns it, without
    # building a full processor.
    owner = next(v for v in vars(base).values()
                 if isinstance(v, type) and "_check_dependencies" in vars(v))
    owner._check_dependencies(object.__new__(owner))
    for line in capsys.readouterr().out.splitlines():
        if "or higher is recommended" in line:
            found, minimum = re.findall(r"\d+(?:\.\d+)+", line)[:2]
            assert version_below(found, minimum), line
