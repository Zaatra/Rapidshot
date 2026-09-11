"""probe_onnxruntime() must never hand LoadLibraryW a bare DLL name.

It used to default to "onnxruntime.dll", which walks the DLL search path: on
Windows 11 that found the OS's own System32 copy rather than the installed
package's, and elsewhere could reach the working directory and PATH. And
rapidshot.capabilities() / diagnose() call it with no argument.
"""

from pathlib import Path

import pytest

from rapidshot import native


class RecordingExtension:
    def __init__(self):
        self.loaded = []

    def probe_onnxruntime(self, dll_path):
        self.loaded.append(dll_path)
        return {"dll": dll_path, "loaded": True}


@pytest.fixture
def ext(monkeypatch):
    fake = RecordingExtension()
    monkeypatch.setattr(native, "require", lambda: fake)
    return fake


def test_default_probes_the_installed_packages_dll(ext, monkeypatch, tmp_path):
    dll = tmp_path / "capi" / "onnxruntime.dll"
    dll.parent.mkdir()
    dll.write_bytes(b"MZ")
    monkeypatch.setattr(native, "onnxruntime_dll_path", lambda: str(dll))
    result = native.probe_onnxruntime()
    assert ext.loaded == [str(dll.resolve())]
    assert result["loaded"] is True


def test_no_package_means_nothing_is_loaded(ext, monkeypatch):
    monkeypatch.setattr(native, "onnxruntime_dll_path", lambda: None)
    result = native.probe_onnxruntime()
    assert ext.loaded == [], "fell back to a name search"
    assert result["loaded"] is False and "not installed" in result["error"]


@pytest.mark.parametrize("given", ["onnxruntime.dll", "does/not/exist.dll"])
def test_a_name_or_missing_file_is_refused_not_searched(ext, given, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)   # an empty directory: the bare name resolves to nothing
    result = native.probe_onnxruntime(given)
    assert ext.loaded == []
    assert result["loaded"] is False and "no such file" in result["error"]


def test_an_explicit_existing_path_is_loaded_absolute(ext, monkeypatch, tmp_path):
    dll = tmp_path / "onnxruntime.dll"
    dll.write_bytes(b"MZ")
    monkeypatch.chdir(tmp_path)
    native.probe_onnxruntime("onnxruntime.dll")   # relative, but a real file here
    assert ext.loaded == [str(dll.resolve())]
    assert Path(ext.loaded[0]).is_absolute()
