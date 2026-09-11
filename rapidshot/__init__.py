import threading
import weakref
import time
from rapidshot.util.logging import get_logger
import platform
import sys
from typing import Optional, Tuple, Dict, Any
# Screen capture relies on Windows-specific COM technology. Attempt to import it lazily so
# that other modules (e.g., memory pools) remain usable on non-Windows platforms.
try:
    from rapidshot.capture import ScreenCapture  # Updated import: now from rapidshot.capture
    _capture_import_error = None
except ImportError as exc:
    ScreenCapture = None
    _capture_import_error = exc
# DXGI device discovery also relies on COM, so guard those imports as well.
try:
    from rapidshot.core import Output, Device
    _core_import_error = None
except ImportError as exc:
    Output = Device = None
    _core_import_error = exc

try:
    from rapidshot.util.io import (
        enum_dxgi_adapters,
        get_output_metadata,
    )
    _io_import_error = None
except ImportError as exc:
    enum_dxgi_adapters = get_output_metadata = None
    _io_import_error = exc
from rapidshot.util.logging import setup_logging, get_logger
from rapidshot.util.topology import (
    AdapterInfo,
    GpuTopology,
    classify,
    probe_topology,
)

# Pure NumPy, so it imports on any platform even though capture does not.
from rapidshot.preprocess import to_nchw

# Initialize logging
logger = get_logger("init")

# Define explicitly what's exposed from this module
__all__ = [
    "capabilities",
    "diagnose",
    "create", "device_info", "output_info", "topology_info",
    "clean_up", "reset", "ScreenCapture",
    "RapidshotError", "HeadlessError", "get_version_info",
    "probe_topology", "GpuTopology", "AdapterInfo",
    "to_nchw",
]

class RapidshotError(Exception):
    """Base exception for Rapidshot errors."""
    pass

class DeviceError(RapidshotError):
    """Exception raised for errors related to device operations."""
    pass

class HeadlessError(DeviceError):
    """Raised when no adapter drives a display, so there is nothing to duplicate.

    Carries the probed topology so a caller can report which adapters exist.
    Subclasses DeviceError: code that already handles "no usable device" keeps
    working, it just gets a message it can act on.
    """
    def __init__(self, message, topology=None):
        super().__init__(message)
        self.topology = topology


class OutputError(RapidshotError):
    """Exception raised for errors related to output operations."""
    pass

class ConfigurationError(RapidshotError):
    """Exception raised for errors related to configuration."""
    pass

class Singleton(type):
    """
    Singleton metaclass to ensure only one instance of RapidshotFactory exists.

    Constructing the factory enumerates DXGI adapters and opens a D3D11 device
    per adapter, so a check-then-act race here does not merely waste work: two
    threads calling :func:`create` at once both built a factory, one of them was
    thrown away, and the devices it had opened stayed open. Reentrant because
    the constructor runs while the lock is held.
    """
    _instances = {}
    _lock = threading.RLock()

    def __call__(cls, *args, **kwargs):
        instance = cls._instances.get(cls)
        if instance is None:
            with cls._lock:
                instance = cls._instances.get(cls)
                if instance is None:
                    instance = super(Singleton, cls).__call__(*args, **kwargs)
                    cls._instances[cls] = instance
                    return instance
        logger.debug(f"Using existing instance of {cls.__name__}")
        return instance

class RapidshotFactory(metaclass=Singleton):
    """
    Factory class for creating ScreenCapture instances.
    Maintains a registry of created screencapture instances to avoid duplicates.
    """
    _screencapture_instances = weakref.WeakValueDictionary()

    def __init__(self) -> None:
        """
        Initialize the factory by enumerating all available devices and outputs.
        """
        logger.info("Initializing RapidshotFactory")
        if _core_import_error is not None or _io_import_error is not None:
            raise RapidshotError(
                "DXGI device enumeration is not available on this platform."
            ) from (_core_import_error or _io_import_error)
        try:
            # Probe the topology before opening any device. An adapter with no
            # outputs is dropped below, but it is exactly the signal that
            # distinguishes a headless machine from a hybrid GPU one — so the
            # classification has to happen while that information still exists.
            self.topology = probe_topology()

            p_adapters = enum_dxgi_adapters()
            if not p_adapters:
                logger.error("No DXGI adapters found")
                raise HeadlessError(self.topology.help_text(), topology=self.topology)

            self.devices, self.outputs = [], []
            self.device_failures = []
            # Every adapter that opened, whether or not it owns an output.
            #
            # These used to be dropped, on the assumption that the adapter
            # owning the display is the one that can duplicate it. That is not
            # true on a hybrid system: which adapter DDA will accept depends on
            # where the desktop is actually composed, not on which adapter
            # enumerates the output. Keeping render-only adapters gives
            # _select_duplication_device something to fall back to, and is also
            # what makes `prefer_integrated` reachable -- on the machine this
            # was found on, the iGPU had zero outputs, so it was discarded here
            # and `prefer_integrated=True` had nothing left to select.
            self.all_devices = []

            for p_adapter in p_adapters:
                try:
                    device = Device(p_adapter)
                    p_outputs = device.enum_outputs()
                    self.all_devices.append(device)
                    if len(p_outputs) != 0:
                        self.devices.append(device)
                        self.outputs.append([Output(p_output) for p_output in p_outputs])
                except Exception as e:
                    logger.warning(f"Failed to initialize device: {e}")
                    self.device_failures.append(str(e))

            if not self.devices:
                # Every adapter either has no output or refused to open. The
                # topology says which, and only one of the two is fixable by
                # the user.
                logger.error(f"No capture-capable device ({self.topology.kind})")
                raise HeadlessError(
                    self._no_device_message(), topology=self.topology
                )

            self.output_metadata = get_output_metadata()
            if self.topology.is_hybrid:
                logger.warning(
                    "Hybrid GPU system: capture is bound to the display adapter. "
                    "See rapidshot.topology_info() for detail."
                )
            logger.info(f"RapidshotFactory initialized with {len(self.devices)} devices")
        except RapidshotError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize RapidshotFactory: {e}"
            logger.error(error_msg)
            raise RapidshotError(error_msg) from e

    def _no_device_message(self) -> str:
        """Explain why no adapter could be used, without guessing."""
        help_text = self.topology.help_text()
        if not help_text:
            # Outputs exist, so this is not a headless machine: every device
            # creation failed instead. Report that, not a display problem.
            help_text = (
                "No usable graphics device. Adapters with displays attached were "
                "found, but none could be opened as a Direct3D 11 device."
            )
        if self.device_failures:
            failures = "\n".join(f"  {f}" for f in self.device_failures)
            help_text += f"\n\nDevice creation errors:\n{failures}"
        return help_text

    @staticmethod
    def _is_integrated(device) -> bool:
        """Best-effort "is this an iGPU", for candidate ordering only.

        Deliberately not a capability check -- see util/topology.py: vendor is
        not a capability and nothing branches capture behaviour on it. This
        only decides which adapter to *try* first, and trying is what settles
        the question.
        """
        desc = getattr(getattr(device, "desc", None), "Description", "") or ""
        return "intel" in str(desc).lower()

    def _duplication_candidates(self, chosen, prefer_integrated: bool):
        """Every adapter that could duplicate this output, in preference order.

        Includes `chosen` -- the caller does not prepend it separately. That
        matters for `prefer_integrated`: on a hybrid laptop the iGPU usually
        owns no output, so it is not in `self.devices` and cannot be `chosen`.
        Ordering it merely ahead of the *other* fallbacks would put it behind
        the display-owning adapter, which on a working system duplicates
        successfully -- so the integrated adapter would never be tried and the
        flag would do nothing in exactly the topology it exists for.

        Includes adapters with no outputs of their own: which adapter Desktop
        Duplication accepts depends on where the desktop is composed, not on
        which adapter enumerates the output.
        """
        candidates = [chosen] + [d for d in self.all_devices if d is not chosen]
        if prefer_integrated:
            # Stable, so this reorders without dropping or shuffling anything.
            candidates.sort(key=lambda d: not self._is_integrated(d))
        return candidates

    def create(
        self,
        device_idx: int = 0,
        output_idx: int = None,
        region: tuple = None,
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len: int = 64,
        prefer_integrated: bool = False,  # New parameter to force integrated GPU
        pool_output: bool = True,
        timeout_ms: int = 10,
        pool_size_frames: int = 4,
    ) -> "ScreenCapture":
        """
        Create a ScreenCapture instance.

        Args:
            device_idx: Device index
            output_idx: Output index (None for primary)
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for capture
            prefer_integrated: If True, will search for an integrated GPU (e.g., Intel) and select it.
            pool_output: Reuse converted-frame buffers instead of allocating one
                per frame (default since 2.0). grab() then returns a
                PooledBuffer the caller must release. Pass False for the
                pre-2.0 behaviour.
            timeout_ms: How long each acquire waits for a new frame. 0 polls,
                costing ~4x the CPU for ~7% more frames; the default blocks.
                See ScreenCapture.timeout_ms for the measured curve.
            pool_size_frames: Buffers kept for grab() to hand out. Each is a
                full frame, so this is the main tunable part of the process
                footprint. Raise it only if you hold several frames at once.

        Returns:
            ScreenCapture instance
        """
        if ScreenCapture is None:
            raise RapidshotError(
                "ScreenCapture is not available on this platform."
            ) from _capture_import_error
        logger.debug(f"Creating ScreenCapture with device_idx={device_idx}, output_idx={output_idx}, nvidia_gpu={nvidia_gpu}, prefer_integrated={prefer_integrated}")
        
        # If the user prefers an integrated GPU, try to find one automatically.
        if prefer_integrated:
            for idx, device in enumerate(self.devices):
                desc = device.desc.Description if device.desc and hasattr(device.desc, "Description") else ""
                if "intel" in desc.lower():
                    device_idx = idx
                    logger.info(f"Selecting integrated GPU: {desc} at index {idx}")
                    break
            else:
                # Not necessarily absent -- on a hybrid system the iGPU often
                # owns no output, so it is not in self.devices at all. It is
                # still a valid duplication target, and the candidate ordering
                # below puts it first so the pairing search reaches it before
                # the discrete adapter.
                logger.info(
                    "No integrated GPU owns an output; keeping the default "
                    "device index and trying integrated adapters first when "
                    "duplication is set up."
                )
        
        # Validate device index
        if device_idx >= len(self.devices):
            error_msg = f"Invalid device index: {device_idx}, max index is {len(self.devices)-1}"
            logger.error(error_msg)
            raise DeviceError(error_msg)
            
        device = self.devices[device_idx]
        
        # Auto-select primary output if not specified
        if output_idx is None:
            output_idx_list = []
            for idx, output in enumerate(self.outputs[device_idx]):
                metadata = self.output_metadata.get(output.devicename)
                if metadata and metadata[1]:  # Is primary
                    output_idx_list.append(idx)
            if not output_idx_list:
                output_idx = 0
                logger.info("No primary monitor found, using first available output.")
            else:
                output_idx = output_idx_list[0]
                logger.info(f"Using primary monitor (output index {output_idx})")
        elif output_idx >= len(self.outputs[device_idx]):
            error_msg = f"Invalid output index: {output_idx}, max index is {len(self.outputs[device_idx])-1}"
            logger.error(error_msg)
            raise OutputError(error_msg)
        
        # Validate color format
        valid_color_formats = ["RGB", "RGBA", "BGR", "BGRA", "GRAY"]
        if output_color not in valid_color_formats:
            error_msg = f"Invalid color format: {output_color}. Must be one of {valid_color_formats}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        # Check if instance already exists
        # The duplication preference is part of the capture configuration, not
        # merely a hint used during construction. On an Optimus system whose
        # iGPU owns no output, both calls below have the same public device and
        # output indices but deliberately reverse the adapter candidate order.
        # Reusing the first instance would silently make the second call's
        # preference inert.
        instance_key = (device_idx, output_idx, bool(prefer_integrated))
        # Everything else the caller chose, as requested -- recorded before the
        # CuPy fallback below can rewrite nvidia_gpu, so the same request always
        # compares equal. The key above picks the output; this decides whether
        # the camera already on it is the one being asked for.
        requested = {
            "region": tuple(region) if region is not None else None,
            "output_color": output_color,
            "nvidia_gpu": bool(nvidia_gpu),
            "max_buffer_len": max_buffer_len,
            "pool_output": bool(pool_output),
            "timeout_ms": timeout_ms,
            "pool_size_frames": pool_size_frames,
        }
        existing = self._screencapture_instances.get(instance_key)
        # A released camera stays in this weak cache for as long as anything
        # still references it -- including the variable about to be rebound in
        # `camera.release(); camera = rapidshot.create()`. Returning it would
        # hand back a camera that silently yields None forever.
        if existing is not None and not getattr(existing, "released", False):
            built_with = getattr(existing, "_factory_config", None)
            if built_with is not None and built_with != requested:
                # Returning it anyway handed the caller frames in the wrong
                # format, region or processor, with nothing to say so.
                differing = ", ".join(
                    f"{name}={built_with.get(name)!r} (asked for {value!r})"
                    for name, value in requested.items()
                    if built_with.get(name) != value)
                raise ConfigurationError(
                    f"A camera for device {device_idx}, output {output_idx} "
                    f"already exists with different settings: {differing}. "
                    "RapidShot keeps one camera per output; call release() on "
                    "the existing camera before creating one with other settings."
                )
            logger.info(f"Found existing ScreenCapture instance for Device {device_idx}--Output {output_idx}")
            return existing

        try:
            output = self.outputs[device_idx][output_idx]
            output.update_desc()
            
            if nvidia_gpu:
                try:
                    import cupy  # type: ignore[import-not-found]
                    logger.info("Using NVIDIA GPU acceleration with CuPy")
                except ImportError:
                    nvidia_gpu = False
                    logger.warning("NVIDIA GPU acceleration requested but CuPy not available. Falling back to CPU mode.")
            
            screencapture = ScreenCapture(
                output=output,
                device=device,
                candidate_devices=self._duplication_candidates(
                    device, prefer_integrated
                ),
                region=region,
                output_color=output_color,
                nvidia_gpu=nvidia_gpu,
                max_buffer_len=max_buffer_len,
                pool_output=pool_output,
                timeout_ms=timeout_ms,
                pool_size_frames=pool_size_frames,
            )
            screencapture._factory_config = requested
            self._screencapture_instances[instance_key] = screencapture

            # Small delay to ensure initialization is complete
            time.sleep(0.1)
            logger.info(f"Created new ScreenCapture instance for Device {device_idx}--Output {output_idx}")
            return screencapture
        except Exception as e:
            error_msg = f"Failed to create ScreenCapture instance: {e}"
            logger.error(error_msg)
            raise RapidshotError(error_msg) from e

    def device_info(self) -> str:
        """
        Get information about available devices.
        
        Returns:
            String with device information
        """
        ret = ""
        for idx, device in enumerate(self.devices):
            ret += f"Device[{idx}]:{device}\n"
        # Devices only cover adapters that drive a display. Adapters that do
        # not are invisible here otherwise, which is what makes a hybrid
        # system look like a plain single-GPU one.
        ret += self.topology.describe() + "\n"
        return ret

    def topology_info(self) -> str:
        """
        Get the GPU/display topology: which adapters exist, which drive a
        display, and what that implies for capture.

        Returns:
            Multi-line string
        """
        return self.topology.describe()

    def output_info(self) -> str:
        """
        Get information about available outputs.
        
        Returns:
            String with output information
        """
        ret = ""
        for didx, outputs in enumerate(self.outputs):
            for idx, output in enumerate(outputs):
                ret += f"Device[{didx}] Output[{idx}]: "
                ret += f"Resolution:{output.resolution} Rotation:{output.rotation_angle} "
                ret += f"Primary:{self.output_metadata.get(output.devicename)[1]}\n"
        return ret

    def clean_up(self) -> None:
        """
        Release all created screencapture instances.
        """
        logger.info("Cleaning up all ScreenCapture instances")
        for _, screencapture in self._screencapture_instances.items():
            try:
                screencapture.release()
            except Exception as e:
                logger.warning(f"Error releasing ScreenCapture instance: {e}")

    def reset(self) -> None:
        """
        Reset the factory, releasing all resources.
        """
        logger.info("Resetting RapidshotFactory")
        self.clean_up()
        self._screencapture_instances.clear()
        Singleton._instances.clear()


# Global factory instance
__factory = None
# Guards __factory. Separate from Singleton._lock: this one also covers reset(),
# which has to clear the global and the instance registry together.
__factory_lock = threading.RLock()

def get_factory() -> "RapidshotFactory":
    """
    Get the global factory instance, initializing it if necessary.

    Returns:
        RapidshotFactory instance
    """
    global __factory
    if __factory is not None:
        return __factory
    with __factory_lock:
        if __factory is None:
            try:
                __factory = RapidshotFactory()
            except Exception as e:
                logger.error(f"Failed to initialize RapidshotFactory: {e}")
                raise
        return __factory

def create(
    device_idx: int = 0,
    output_idx: int = None,
    region: tuple = None,
    output_color: str = "RGB",
    nvidia_gpu: bool = False,
    max_buffer_len: int = 64,
    prefer_integrated: bool = False,  # New parameter passed to factory
    pool_output: bool = True,
    timeout_ms: int = 10,
    pool_size_frames: int = 4,
) -> "ScreenCapture":
    """
    Create a ScreenCapture instance.
    
    Args:
        device_idx: Device index
        output_idx: Output index (None for primary)
        region: Region to capture (left, top, right, bottom)
        output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        nvidia_gpu: Whether to use NVIDIA GPU acceleration
        max_buffer_len: Maximum buffer length for capture
        prefer_integrated: If True, forces selection of an integrated GPU if available.
        pool_output: Reuse converted-frame buffers (default since 2.0); grab()
            then returns a PooledBuffer the caller must release. Pass False
            for the pre-2.0 behaviour of a freshly allocated array.
        
    Returns:
        ScreenCapture instance
    """
    factory = get_factory()
    return factory.create(
        device_idx=device_idx,
        output_idx=output_idx,
        region=region,
        output_color=output_color,
        nvidia_gpu=nvidia_gpu,
        max_buffer_len=max_buffer_len,
        prefer_integrated=prefer_integrated,
        pool_output=pool_output,
        timeout_ms=timeout_ms,
        pool_size_frames=pool_size_frames,
    )

def device_info() -> str:
    """
    Get information about available devices.
    
    Returns:
        String with device information
    """
    factory = get_factory()
    return factory.device_info()

def output_info() -> str:
    """
    Get information about available outputs.

    Returns:
        String with output information
    """
    factory = get_factory()
    return factory.output_info()

def topology_info() -> str:
    """
    Get the GPU/display topology.

    Unlike device_info(), this reports adapters that cannot capture too — a
    render-only dGPU on a hybrid laptop, or a software adapter. Safe to call on
    a machine where capture itself is unavailable: it probes DXGI directly
    rather than going through the factory.

    Returns:
        String describing the topology
    """
    global __factory
    if __factory is not None:
        return __factory.topology_info()
    return probe_topology().describe()

def clean_up() -> None:
    """
    Release all created screencapture instances.
    """
    with __factory_lock:
        factory = __factory
    # Outside the lock: releasing cameras is slow, and holding the lock across
    # it would block every concurrent create() for the duration.
    if factory is not None:
        factory.clean_up()

def reset() -> None:
    """
    Reset the library, releasing all resources.
    """
    global __factory
    # Held across the whole teardown, not just the swap. `RapidshotFactory.reset`
    # also clears `Singleton._instances`, and between clearing the global and
    # clearing that registry a concurrent get_factory() would be handed back the
    # very factory being torn down. reset() is not on any hot path, and it calls
    # nothing that reaches get_factory().
    with __factory_lock:
        factory, __factory = __factory, None
        if factory is not None:
            factory.reset()

def capabilities(probe_gpu: bool = False) -> Dict[str, Any]:
    """Everything about this machine that decides what RapidShot can do.

    One call, one shape, never raises. Every section is independent and a
    failure in one is reported inside that section rather than aborting the
    report -- the machine where this matters most is the one where something is
    already broken.

    ``probe_gpu`` additionally runs the native cross-adapter probe, which
    creates D3D devices and allocates a shared heap. That is real GPU work and
    is off by default so a diagnostic cannot itself destabilise the thing being
    diagnosed.

    Sections:

    ``rapidshot``
        Version, and whether the optional native extension is loaded and from
        where (a local build and the wheel can both be present and differ).
    ``platform``
        OS, Python, and the packages whose versions change behaviour.
    ``capture``
        Adapter topology: how many adapters, which drive displays, and whether
        this is a hybrid system where the GPU tensor needs a cross-adapter hop.
    ``gpu``
        What the native extension can actually do here, rather than what it was
        compiled with.
    ``dependencies``
        Optional consumers -- CuPy, OpenCV, PIL, ONNX Runtime -- and their
        versions where importable.

    For a printable version see :func:`diagnose`.
    """
    import platform as _platform
    import sys as _sys

    report: Dict[str, Any] = {}

    report["rapidshot"] = {"version": __version__}
    try:
        from rapidshot import native as _native
        report["rapidshot"]["native_extension"] = _native.is_available()
        report["rapidshot"]["native_build"] = _native.build_info()
    except Exception as exc:
        report["rapidshot"]["native_extension"] = False
        report["rapidshot"]["error"] = f"{type(exc).__name__}: {exc}"

    report["platform"] = {
        "os": _platform.platform(),
        "python": _platform.python_version(),
        "machine": _platform.machine(),
        "processor": _platform.processor(),
    }

    # Topology, as text and as counts. The text is what a user pastes into an
    # issue; the counts are what code branches on.
    capture: Dict[str, Any] = {}
    try:
        from rapidshot.util.topology import probe_topology

        topology = probe_topology()
        capture["kind"] = getattr(topology, "kind", None)
        adapters = getattr(topology, "adapters", []) or []
        capture["adapter_count"] = len(adapters)
        capture["display_adapters"] = sum(
            1 for a in adapters if getattr(a, "output_count", 0))
        capture["adapters"] = [
            {"description": getattr(a, "description", None),
             "vendor": getattr(a, "vendor", None),
             "outputs": getattr(a, "output_count", None),
             "dedicated_vram_mb": (getattr(a, "dedicated_video_memory", 0) or 0) // (1024 * 1024),
             "software": getattr(a, "is_software", None)}
            for a in adapters
        ]
        capture["cross_adapter_required_for_gpu_tensor"] = (
            getattr(topology, "kind", None) == "hybrid")
    except Exception as exc:
        capture["error"] = f"{type(exc).__name__}: {exc}"
    report["capture"] = capture

    gpu: Dict[str, Any] = {}
    try:
        from rapidshot import native as _native

        gpu["available"] = _native.is_available()
        if _native.is_available():
            for name, probe in (("shareable_buffers", _native.probe_shareable_buffers),
                                ("onnxruntime", _native.probe_onnxruntime)):
                try:
                    gpu[name] = probe()
                except Exception as exc:
                    gpu[name] = {"error": f"{type(exc).__name__}: {exc}"}
            if probe_gpu:
                try:
                    gpu["cross_adapter"] = _native.probe_cross_adapter()
                except Exception as exc:
                    gpu["cross_adapter"] = {"error": f"{type(exc).__name__}: {exc}"}
            else:
                gpu["cross_adapter"] = "not probed (pass probe_gpu=True)"
    except Exception as exc:
        gpu["error"] = f"{type(exc).__name__}: {exc}"
    report["gpu"] = gpu

    dependencies: Dict[str, Any] = {}
    for module in ("numpy", "comtypes", "cupy", "cv2", "PIL", "onnxruntime"):
        try:
            imported = __import__(module)
            dependencies[module] = getattr(imported, "__version__", "unknown")
        except Exception:
            dependencies[module] = None
    report["dependencies"] = dependencies

    return report


def diagnose(probe_gpu: bool = False) -> str:
    """:func:`capabilities` rendered for a human, and for an issue report.

    The text is the point: most "it does not work" reports are answerable from
    this output alone, and asking a user to run one command beats asking them
    to run six and paste the results in the right order.
    """
    report = capabilities(probe_gpu=probe_gpu)
    lines = ["RapidShot diagnostics", "=" * 21, ""]

    shot = report["rapidshot"]
    lines.append(f"rapidshot {shot.get('version')}")
    build = shot.get("native_build") or {}
    if shot.get("native_extension"):
        lines.append(f"  native extension : yes ({build.get('source', 'unknown source')})")
        lines.append(f"  native version   : {build.get('version')} / {build.get('stage')}")
    else:
        lines.append("  native extension : NO -- GPU tensor and cross-adapter "
                     "transfer unavailable")
        lines.append("                     install with: pip install rapidshot-native")

    plat = report["platform"]
    lines += ["", f"platform : {plat.get('os')}",
              f"python   : {plat.get('python')}",
              f"cpu      : {plat.get('processor')}"]

    capture = report["capture"]
    lines += ["", "capture"]
    if "error" in capture:
        lines.append(f"  topology unavailable: {capture['error']}")
    else:
        lines.append(f"  topology  : {capture.get('kind')} "
                     f"({capture.get('adapter_count')} adapters, "
                     f"{capture.get('display_adapters')} driving a display)")
        for adapter in capture.get("adapters", []):
            outputs = adapter.get("outputs")
            lines.append(f"    - {adapter.get('description')} "
                         f"({adapter.get('vendor')}, {outputs} output"
                         f"{'' if outputs == 1 else 's'})")
        if capture.get("cross_adapter_required_for_gpu_tensor"):
            lines.append("  NOTE: hybrid system. Capture runs on the display "
                         "adapter, so a GPU tensor")
            lines.append("        must cross adapters before CUDA can import it.")

    deps = report["dependencies"]
    lines += ["", "optional dependencies"]
    for name, version in deps.items():
        lines.append(f"  {name:12} {version if version else '-- not installed'}")

    gpu = report["gpu"]
    if isinstance(gpu.get("cross_adapter"), str):
        lines += ["", f"cross-adapter    : {gpu['cross_adapter']}"]
    elif isinstance(gpu.get("cross_adapter"), dict):
        probe = gpu["cross_adapter"]
        lines += ["", f"cross-adapter    : {probe.get('source')} -> "
                      f"{probe.get('destination')} "
                      f"(representative={probe.get('representative')})"]
    return "\n".join(lines)


def get_version_info() -> Dict[str, Any]:
    """
    Get version information about RapidShot and its dependencies.
    
    Returns:
        Dictionary with version information
    """
    info = {
        "rapidshot": {
            "version": __version__,
            "author": __author__,
            "description": __description__,
        },
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "dependencies": {}
    }
    
    # Check numpy
    try:
        import numpy
        info["dependencies"]["numpy"] = numpy.__version__
    except ImportError:
        info["dependencies"]["numpy"] = "not installed"
    
    # Check cupy
    try:
        import cupy  # type: ignore[import-not-found]
        info["dependencies"]["cupy"] = cupy.__version__
    except ImportError:
        info["dependencies"]["cupy"] = "not installed"
    
    # Check pillow
    try:
        from PIL import __version__ as pil_version  # type: ignore[import-not-found]
        info["dependencies"]["pillow"] = pil_version
    except ImportError:
        info["dependencies"]["pillow"] = "not installed"
    
    # Check opencv
    try:
        import cv2  # type: ignore[import-not-found]
        info["dependencies"]["opencv"] = cv2.__version__
    except ImportError:
        info["dependencies"]["opencv"] = "not installed"
    
    # Check comtypes
    try:
        import comtypes  # type: ignore[import-untyped]
        info["dependencies"]["comtypes"] = comtypes.__version__
    except (ImportError, AttributeError):
        info["dependencies"]["comtypes"] = "version unknown"
    
    return info

# Version information. Single source in rapidshot/_version.py -- see the note
# there for why it is not written out again here.
from rapidshot._version import __version__
__author__ = "Rapidshot Contributors"
__description__ = "High-performance screencapture library for Windows using Desktop Duplication API"

