"""Record isolated browser, worksheet and game fixtures on the HP benchmark display."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import ctypes as ct
from ctypes import wintypes as wt
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener
import uuid

from monitor_target import select_monitor
from real_trace import bind


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "trace-fixtures"
CHROME = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
NODE = Path(r"C:\Program Files\nodejs\node.exe")
APPLICATIONS = ("browser", "spreadsheet", "game")
CDP_BRIDGE = r"""
const readline = require('node:readline');
const ws = new WebSocket(process.argv[1]);
ws.addEventListener('open', () => {
  console.log('ready');
  const input = readline.createInterface({input: process.stdin});
  input.on('line', line => ws.send(line));
  input.on('close', () => ws.close());
});
ws.addEventListener('message', event => {
  const object = JSON.parse(event.data);
  if (object.id !== undefined) console.log(JSON.stringify(object));
});
ws.addEventListener('error', () => process.exit(1));
ws.addEventListener('close', () => process.exit(0));
"""


def exited(process, *, timeout=5):
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        return process.wait(timeout=3)


class Output:
    def __init__(self, process, *, mirror=False):
        self.lines = queue.Queue()
        self.mirror = mirror
        self.log = []
        self.reader = threading.Thread(target=self._read, args=(process.stdout,), daemon=True)
        self.reader.start()

    def _read(self, stream):
        with stream:
            for line in stream:
                line = line.rstrip()
                self.log.append(line)
                if self.mirror:
                    print(line, flush=True)
                self.lines.put(line)
        self.lines.put(None)

    def read(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty as error:
            raise RuntimeError("The fixture helper did not respond in time") from error
        if line is None:
            raise RuntimeError("The fixture helper exited: " + " | ".join(self.log[-5:]))
        return line


class DevTools:
    def __init__(self, endpoint):
        target = urlsplit(endpoint)
        if target.scheme != "ws" or target.hostname != "127.0.0.1" or not target.path.startswith("/devtools/"):
            raise ValueError("Only the controlled browser's loopback DevTools endpoint is allowed")
        self.process = subprocess.Popen([str(NODE), "-e", CDP_BRIDGE, endpoint], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
        self.output = Output(self.process)
        self.sequence = 0
        try:
            if self.output.read(10) != "ready":
                raise RuntimeError("The isolated browser's DevTools connection did not open")
        except BaseException:
            self.close()
            raise

    def send(self, method, **params):
        self.sequence += 1
        request = dict(id=self.sequence, method=method, params=params)
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        answer = json.loads(self.output.read(8))
        if answer.get("id") != request["id"] or "error" in answer:
            raise RuntimeError(f"DevTools {method} failed: {answer.get('error', 'out-of-order reply')}")
        return answer.get("result", {})

    def evaluate(self, expression):
        result = self.send("Runtime.evaluate", expression=expression, returnByValue=True)
        if result.get("exceptionDetails"):
            raise RuntimeError(f"The controlled page rejected an action: {result['exceptionDetails'].get('text')}")
        return result.get("result", {}).get("value")

    def close(self):
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        exited(self.process, timeout=2)
        self.output.reader.join(timeout=2)


def endpoint(profile, expected):
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + 12
    file = profile / "DevToolsActivePort"
    while time.monotonic() < deadline:
        try:
            port = int(file.read_text(encoding="ascii").splitlines()[0])
            if not 0 < port <= 65535:
                raise ValueError("Invalid private DevTools port")
            with opener.open(f"http://127.0.0.1:{port}/json/list", timeout=1) as response:
                for candidate in json.load(response):
                    target_url = candidate.get("url", "").rstrip("/")
                    path = urlsplit(candidate.get("webSocketDebuggerUrl", "")).path
                    if candidate.get("type") == "page" and target_url == expected.rstrip("/") and path:
                        return f"ws://127.0.0.1:{port}{path}"
        except (OSError, ValueError, IndexError):
            pass
        time.sleep(0.1)
    raise RuntimeError("The isolated fixture URL did not appear in its DevTools target list")


@contextmanager
def controlled_chrome(app, target):
    if not CHROME.is_file() or not NODE.is_file():
        raise RuntimeError("The installed Chrome and Node runtimes are required")
    url = (ASSETS / "browser.html").as_uri()
    temporary = tempfile.TemporaryDirectory(prefix=f"rapidshot-{app}-", dir=ROOT,
                                           ignore_cleanup_errors=True)
    profile = Path(temporary.name).resolve()
    if profile.parent != ROOT.resolve():
        raise RuntimeError("The isolated browser profile was created outside the scratch workspace")
    l, t, r, b = target["bounds"]
    cmd = [str(CHROME), f"--user-data-dir={profile}", "--remote-debugging-port=0",
           "--remote-debugging-address=127.0.0.1", "--no-first-run", "--no-default-browser-check",
           "--disable-background-networking", "--disable-extensions",
           f"--window-position={l + 18},{t + 25}", f"--window-size={r - l - 36},{b - t - 70}"]
    cmd += [f"--app={url}"] if app == "browser" else ["--new-window", url]
    process = None
    pipe = None
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pipe = DevTools(endpoint(profile, url))
        window = pipe.send("Browser.getWindowForTarget")["windowId"]
        pipe.send("Browser.setWindowBounds", windowId=window, bounds={"windowState": "normal"})
        pipe.send("Browser.setWindowBounds", windowId=window,
                  bounds=dict(left=l + 18, top=t + 25, width=r - l - 36, height=b - t - 70))
        pipe.send("Page.bringToFront")
        if app == "game":
            # Dino is an intentional Chromium network-error page. Attach before navigating.
            navigation = pipe.send("Page.navigate", url="chrome://dino/")
            if navigation.get("errorText") not in (None, "net::ERR_INTERNET_DISCONNECTED"):
                raise RuntimeError(f"Dino navigation failed: {navigation['errorText']}")
        control_title = f"RapidShot controlled {app} {uuid.uuid4().hex[:8]}"
        selector = "#start" if app == "browser" else ".runner-container canvas"
        expiry = time.monotonic() + 4
        while True:
            try:
                if pipe.evaluate(f"Boolean(document.querySelector({json.dumps(selector)}))"):
                    break
            except RuntimeError:
                # Navigation can briefly invalidate the execution context.
                pass
            if time.monotonic() >= expiry:
                raise RuntimeError(f"The controlled {app} fixture did not load")
            time.sleep(0.1)
        observed = pipe.evaluate(f"document.title = {json.dumps(control_title)}; document.title")
        if observed != control_title:
            raise RuntimeError("The controlled browser title could not be verified")
        yield pipe, control_title, process.pid
    finally:
        try:
            if pipe is not None:
                try:
                    pipe.send("Browser.close")
                except (RuntimeError, OSError):
                    pass
                finally:
                    pipe.close()
        finally:
            try:
                if process is not None:
                    exited(process)
            finally:
                temporary.cleanup()


def process_id(hwnd):
    user = ct.WinDLL("user32", use_last_error=True)
    get_pid = user.GetWindowThreadProcessId
    get_pid.argtypes = [wt.HWND, ct.POINTER(wt.DWORD)]
    get_pid.restype = wt.DWORD
    value = wt.DWORD()
    if not get_pid(hwnd, ct.byref(value)):
        raise ct.WinError(ct.get_last_error())
    return int(value.value)


def place_fixture(title, pid, bounds):
    user = ct.WinDLL("user32", use_last_error=True)
    callback = ct.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    enumerate_windows = bind(user, "EnumWindows", wt.BOOL, callback, wt.LPARAM)
    get_title = bind(user, "GetWindowTextW", ct.c_int, wt.HWND, wt.LPWSTR, ct.c_int)
    is_visible = bind(user, "IsWindowVisible", wt.BOOL, wt.HWND)
    matches = []

    @callback
    def visit(hwnd, _):
        name = ct.create_unicode_buffer(256)
        if is_visible(hwnd) and get_title(hwnd, name, len(name)) and title in name.value:
            if process_id(hwnd) == pid:
                matches.append(hwnd)
        return True

    deadline = time.monotonic() + 4
    while not matches and time.monotonic() < deadline:
        enumerate_windows(visit, 0)
        if not matches:
            time.sleep(0.08)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible fixture window owned by the launched process, found {len(matches)}")
    hwnd = matches[0]
    set_position = bind(user, "SetWindowPos", wt.BOOL, wt.HWND, wt.HWND,
                        ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.UINT)
    set_foreground = bind(user, "SetForegroundWindow", wt.BOOL, wt.HWND)
    l, t, r, b = bounds
    swp_noactivate, swp_showwindow = 0x0010, 0x0040
    if not set_position(hwnd, wt.HWND(-1), l + 18, t + 25, r - l - 36, b - t - 70,
                        swp_noactivate | swp_showwindow):
        raise ct.WinError(ct.get_last_error())
    set_foreground(hwnd)
    return hwnd


def excel_pids():
    class ProcessEntry(ct.Structure):
        _fields_ = [("size", wt.DWORD), ("usage", wt.DWORD), ("pid", wt.DWORD),
                    ("heap", ct.c_size_t), ("module", wt.DWORD), ("threads", wt.DWORD),
                    ("parent", wt.DWORD), ("priority", wt.LONG), ("flags", wt.DWORD),
                    ("name", wt.WCHAR * 260)]

    kernel = ct.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel.CreateToolhelp32Snapshot
    snapshot.argtypes = [wt.DWORD, wt.DWORD]
    snapshot.restype = wt.HANDLE
    close = kernel.CloseHandle
    close.argtypes = [wt.HANDLE]
    handle = snapshot(2, 0)
    if not handle or handle == ct.c_void_p(-1).value:
        raise ct.WinError(ct.get_last_error())
    pids = set()
    try:
        first, subsequent = kernel.Process32FirstW, kernel.Process32NextW
        for function in (first, subsequent):
            function.argtypes = [wt.HANDLE, ct.POINTER(ProcessEntry)]
            function.restype = wt.BOOL
        item = ProcessEntry(size=ct.sizeof(ProcessEntry))
        valid = first(handle, ct.byref(item))
        if not valid:
            raise ct.WinError(ct.get_last_error())
        while valid:
            if item.name.casefold() == "excel.exe":
                pids.add(item.pid)
            valid = subsequent(handle, ct.byref(item))
    finally:
        close(handle)
    return pids


@contextmanager
def controlled_excel(target):
    from comtypes.client import CreateObject

    existing = excel_pids()
    app = CreateObject("Excel.Application", dynamic=True)
    pid = process_id(app.Hwnd)
    if pid in existing:
        raise RuntimeError("Excel automation attached to a pre-existing process")
    workbook = None
    try:
        app.DisplayAlerts = False
        app.EnableEvents = False
        workbook = app.Workbooks.Add(-4167)
        sheet = workbook.ActiveSheet
        sheet.Name = "Controlled trace"
        with (ASSETS / "spreadsheet.csv").open(encoding="utf-8", newline="") as file:
            rows = list(csv.reader(file))
        for row, values in enumerate(rows, 1):
            for column, value in zip("ABCDEFG", values):
                sheet.Range[f"{column}{row}"].Value2 = int(value) if value.isdecimal() else value
        sheet.Range["A1:G1"].Font.Bold = True
        sheet.Range["A:G"].EntireColumn.AutoFit()
        win = workbook.Windows.Item(1)
        title = f"RapidShot controlled spreadsheet {uuid.uuid4().hex[:8]}"
        win.Caption = title
        l, t, r, b = target["bounds"]
        app.WindowState = -4143
        app.Left = (l + 18) * 0.75
        app.Top = (t + 30) * 0.75
        app.Width = (r - l - 36) * 0.75
        app.Height = (b - t - 70) * 0.75
        app.Visible = True
        win.Activate()
        win.ScrollRow = 1
        yield sheet, win, title, pid
    finally:
        try:
            if workbook is not None:
                workbook.Close(False)
        finally:
            app.Quit()


def game_key(pipe, kind):
    pipe.send("Input.dispatchKeyEvent", type=kind, code="Space", key=" ",
              windowsVirtualKeyCode=32, nativeVirtualKeyCode=32)


def browser_motion(pipe, app, seconds):
    if app == "browser":
        pipe.evaluate("document.querySelector('#start').click()")
        return
    deadline = time.monotonic() + max(0.5, seconds - 0.6)
    visuals = set()
    while time.monotonic() < deadline:
        game_key(pipe, "rawKeyDown")
        time.sleep(0.08)
        game_key(pipe, "keyUp")
        visuals.add(pipe.evaluate("document.querySelector('.runner-container canvas').toDataURL()"))
        time.sleep(0.58)
    if len(visuals) < 2:
        raise RuntimeError("The Chrome Dino canvas did not advance during the recording")


def spreadsheet_motion(sheet, win, seconds):
    deadline = time.monotonic() + max(0.5, seconds - 0.55)
    step = 0
    while time.monotonic() < deadline:
        row = 2 + step * 2
        sheet.Range[f"D{row + 4}"].Value2 = 200 + step
        win.ScrollRow = row
        step += 1
        time.sleep(0.085)


def record(args, target, app, title, action, *, pid, region=None):
    path = args.out / f"{app}.npz"
    hwnd = place_fixture(title, pid, target["bounds"])
    command = [sys.executable, "-X", "utf8", "-B", str(ROOT / "real_trace.py"), "record", "--app", app,
               "--title-contains", title, "--device", str(target["device_index"]),
               "--output-index", str(target["output_index"]), "--bounds", *map(str, target["bounds"]),
               "--seconds", str(args.seconds), "--hz", str(args.record_hz), "--wait", "10",
               "--max-mib", str(args.max_mib), "--out", str(path), "--pid", str(pid),
               "--hwnd", str(hwnd)]
    if region is not None:
        command += ["--region", *map(str, region)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    output = Output(process, mirror=True)
    try:
        status = output.read(15)
        if not status.startswith("Waiting for the controlled fixture"):
            raise RuntimeError("The guarded trace recorder did not arm")
        if not output.read(15).startswith("Recording the verified fixture"):
            raise RuntimeError("The guarded trace recorder did not verify the fixture")
        time.sleep(0.3)
        if process.poll() is not None:
            raise RuntimeError("The guarded trace recorder stopped before the motion")
        action()
        status = process.wait(timeout=65)
        if status or not path.is_file():
            raise RuntimeError(f"{app} trace recording failed")
        return path
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=3)
        output.reader.join(timeout=4)


def dpi_context():
    user = ct.WinDLL("user32", use_last_error=True)
    set_dpi = user.SetThreadDpiAwarenessContext
    set_dpi.argtypes = [ct.c_void_p]
    set_dpi.restype = ct.c_void_p
    if not set_dpi(ct.c_void_p(-4)):
        raise ct.WinError(ct.get_last_error())


def launch(args, target, app):
    if app == "spreadsheet":
        with controlled_excel(target) as (sheet, win, title, pid):
            l, t, r, b = target["bounds"]
            # Record worksheet body only, away from the Office identity toolbar.
            crop = (l + 70, t + 230, r - 175, b - 135)
            return record(args, target, app, title, lambda: spreadsheet_motion(sheet, win, args.seconds),
                          pid=pid, region=crop)
    with controlled_chrome(app, target) as (pipe, title, pid):
        return record(args, target, app, title, lambda: browser_motion(pipe, app, args.seconds), pid=pid)


def main():
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=(*APPLICATIONS, "all"), default="all")
    parser.add_argument("--monitor", default="HP 324pe")
    parser.add_argument("--out", type=Path, default=ROOT / "results-downstream-hp324pe" / "traces")
    parser.add_argument("--seconds", type=float, default=2.4)
    parser.add_argument("--record-hz", type=int, default=100)
    parser.add_argument("--max-mib", type=int, default=1664)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--rates", type=int, nargs="+", default=tuple(range(10, 101, 10)))
    args = parser.parse_args()
    if not 1.5 <= args.seconds <= 3 or not 1 <= args.record_hz <= 100 or args.max_mib <= 0:
        parser.error("choose 1.5–3 seconds, 1–100 Hz and a positive memory budget")
    dpi_context()
    target = select_monitor(args.monitor)
    print(f"Recording controlled fixtures on {target['monitor']} {target['bounds']} at {target['physical_refresh_hz']} Hz", flush=True)
    failures = []
    for app in APPLICATIONS if args.app == "all" else (args.app,):
        try:
            path = launch(args, target, app)
            if args.replay:
                subprocess.run([sys.executable, "-X", "utf8", "-B", str(ROOT / "real_trace.py"), "replay", "--trace", str(path),
                                "--rates", *map(str, args.rates), "--out", str(args.out)], check=True,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as error:
            failures.append(app)
            print(f"{app} fixture failed: {error}", file=sys.stderr, flush=True)
    if failures:
        raise SystemExit("Trace recording or replay failed for " + ", ".join(failures))


if __name__ == "__main__":
    main()
