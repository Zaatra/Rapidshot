"""How `rapidshot.native` finds the extension, and in what order.

There are two ways to have it: built in place by `native/install_dev.py`, or
installed as the `rapidshot-native` wheel. The order matters and is not
arbitrary -- a developer who has just rebuilt expects to be testing that build,
and silently preferring an installed wheel would make `cargo build` appear to do
nothing at all.

These tests exercise the resolution logic by reloading the module against a
patched import system, so they run identically whether or not either route is
actually present on the machine.
"""

from __future__ import annotations

import builtins
import importlib
import sys
import types

import pytest

import rapidshot.native


def _reload_with(available):
    """Reload `rapidshot.native` with only `available` importable.

    `available` maps module name -> module object. Anything else raises
    ImportError, which is what an absent extension looks like.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ("rapidshot", "rapidshot_native") and fromlist:
            for item in fromlist:
                key = f"{name}.{item}"
                if key in available:
                    module = types.ModuleType(name)
                    setattr(module, item, available[key])
                    return module
            if name == "rapidshot_native":
                raise ImportError(f"No module named '{name}'")
            raise ImportError(f"cannot import name '{fromlist[0]}' from '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    saved = sys.modules.pop("rapidshot.native", None)
    builtins.__import__ = fake_import
    try:
        return importlib.import_module("rapidshot.native")
    finally:
        builtins.__import__ = real_import
        sys.modules["rapidshot.native"] = saved if saved else sys.modules["rapidshot.native"]


def _fake_ext(tag):
    ext = types.ModuleType(f"_rapidshot_native_{tag}")
    ext.build_info = lambda: {"version": "9.9.9", "stage": tag}
    return ext


class TestExtensionDiscovery:
    def test_in_package_build_is_found(self):
        mod = _reload_with({"rapidshot._rapidshot_native": _fake_ext("dev")})
        assert mod.is_available()
        assert "development build" in mod.build_info()["source"]

    def test_wheel_is_found_when_no_local_build(self):
        mod = _reload_with({"rapidshot_native._rapidshot_native": _fake_ext("wheel")})
        assert mod.is_available()
        assert mod.build_info()["source"] == "rapidshot-native wheel"

    def test_local_build_wins_over_wheel(self):
        """Rebuilding must not be shadowed by an installed wheel.

        Both present is the normal state for a contributor who installed the
        wheel first and later built from source. If the wheel won, every
        `cargo build` would appear to have no effect, which is a debugging
        session nobody should have to have.
        """
        mod = _reload_with({
            "rapidshot._rapidshot_native": _fake_ext("dev"),
            "rapidshot_native._rapidshot_native": _fake_ext("wheel"),
        })
        assert "development build" in mod.build_info()["source"]
        assert mod.build_info()["stage"] == "dev"

    def test_neither_present_is_not_an_error(self):
        """Absence is the normal case for `pip install rapidshot`."""
        mod = _reload_with({})
        assert not mod.is_available()
        assert mod.build_info() is None

    def test_hint_leads_with_the_wheel(self):
        """The hint is the only place most users learn the wheel exists.

        It used to offer only "install Rust and the MSVC build tools", which is
        the reason the extension went unused: the easy route was not mentioned
        because it did not exist yet.
        """
        hint = rapidshot.native.BUILD_HINT
        assert "pip install rapidshot-native" in hint
        assert hint.index("pip install rapidshot-native") < hint.index("cargo build")

    def test_require_raises_with_the_hint(self):
        mod = _reload_with({})
        with pytest.raises(RuntimeError) as excinfo:
            mod.require()
        assert "rapidshot-native" in str(excinfo.value)
