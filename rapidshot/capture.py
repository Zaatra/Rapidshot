import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock, RLock, current_thread
import comtypes  # type: ignore[import-untyped]
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from rapidshot.memory_pool import (
    NumpyMemoryPool,
    CupyMemoryPool,
    PooledBuffer,
    PoolExhaustedError,
)
from rapidshot.util.errors import ( # Added for Phase 2
    RapidShotError,
    RapidShotDXGIError,
    RapidShotReinitError,
    RapidShotDeviceError,
    RapidShotConfigError,
    RapidShotProtectedContentError,
)
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.core.stagesurf import StageSurface
from rapidshot.core.duplicator import Duplicator
from rapidshot._libs.dxgi import (
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_INVALID_CALL,
)
from rapidshot.util.topology import probe_topology
from rapidshot._libs.d3d11 import D3D11_BOX
from rapidshot.processor import Processor
from rapidshot.util.ctypes_helpers import describe_destination
import collections # Added for deque
from rapidshot.util.timer import (
    create_high_resolution_timer,
    set_periodic_timer,
    wait_for_timer,
    cancel_timer,
    close_timer,
    INFINITE,
    WAIT_FAILED,
)

# Set up logger
logger = logging.getLogger(__name__)

# CuPy is imported on first use, not at import time.
#
# Importing it costs **178.8 MB** resident (measured 2026-09-14, ROADMAP § 7.0)
# and every caller paid that merely for `import rapidshot` -- including on
# machines with no NVIDIA GPU, and for callers who only ever touch `grab()`.
# It was 185 of the 217 MB the package cost before its first camera existed;
# the native extension, by comparison, is 1.3 MB.
#
# Nothing here needs CuPy unless `nvidia_gpu=True`, so every use below is
# reached through `_require_cupy()` behind that flag. `CUPY_AVAILABLE` and `cp`
# remain readable as module attributes for anything outside that imported them,
# and resolve the same way -- lazily, via `__getattr__` at the end of this
# module.
_cupy = None
_cupy_import_attempted = False


def _require_cupy():
    """The CuPy module, or None if it is not installed. Imported once."""
    global _cupy, _cupy_import_attempted
    if not _cupy_import_attempted:
        _cupy_import_attempted = True
        try:
            import cupy  # type: ignore[import-not-found]
            _cupy = cupy
        except ImportError:
            _cupy = None
    return _cupy


def _require_positive_int(name: str, value) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def cupy_available() -> bool:
    """Whether CuPy can be imported. Imports it to find out, so call it behind
    ``nvidia_gpu`` rather than as a general capability probe."""
    return _require_cupy() is not None

class ScreenCapture:
    #: How long stop() waits for the capture thread before giving up on it.
    #: An abandoned thread still owns its timer handle and the frame queue, so
    #: this is also the point past which stop() stops tidying up after it.
    _stop_join_timeout_s = 10.0

    def __init__(
        self,
        output: Output,
        device: Device,
        region: Optional[Tuple[int, int, int, int]] = None,
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len: int = 64, # This is for the continuous mode ring buffer
        pool_size_frames: int = 2,
        pool_output: bool = True,
        timeout_ms: int = 10,
        candidate_devices: Optional[List[Device]] = None,
    ) -> None:
        """
        Initialize a ScreenCapture instance.

        Args:
            output: Output device to capture from
            device: Device interface
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for continuous mode capture
            pool_size_frames: Buffers kept for ``grab()`` to hand out. Each one
                is a full frame (8.3 MB at 1080p BGRA), which makes this the
                main tunable part of the process footprint: measured per-camera
                cost is 114 MB at 10 buffers, 85 MB at 4 and 81 MB at 2, against
                75 MB with no pool at all. The default dropped 10 -> 4 for 29 MB
                with no measurable change in frame rate, and **4 -> 2 on
                2026-09-14** for the same reason: at 2560x1600 the process was
                196.3 MB at 4 and 171.7 MB at 2 -- one 12.3 MB RGB buffer per
                step, exactly linear -- at 134.4 and 139.1 fps, a difference
                inside the run-to-run spread. Holding a rolling window of 1, 3
                and 6 frames did not separate them either (140.8/134.5/132.6 at
                2 against 136.5/130.7/134.9 at 4), and neither size could be
                made to exhaust: 25 frames were held at both. Raise it only if
                you genuinely hold several frames at once. Running dry is not an
                error for a converting mode: it falls back to allocating, which
                is slower but always correct.
                ``BGRA`` has no such fallback -- the frame is the staging
                buffer -- so ``grab()`` returns None until a buffer is
                released.
            pool_output: Reuse buffers for the converted frame instead of
                allocating one per frame. Saves ~1.6 ms on a 1080p RGB frame --
                the page faults on first touch cost more than the conversion.
                ``grab()`` then returns a ``PooledBuffer``, which behaves like
                the array for indexing and ``np.asarray`` but **must** be
                released when done. Pass False for the pre-2.0 behaviour of
                returning a freshly allocated array that needs no release.
            timeout_ms: How long each acquire waits for the compositor to
                present a new frame. 0 polls, which costs roughly 4x the CPU for
                about 7% more frames; the default blocks. See
                :attr:`timeout_ms` for the measured curve.

        Raises:
            ValueError: If timeout_ms is negative.
        """
        if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
            raise ValueError(
                f"timeout_ms must be a non-negative int, got {timeout_ms!r}"
            )
        if (not isinstance(pool_size_frames, int) or isinstance(pool_size_frames, bool)
                or pool_size_frames < 1):
            raise ValueError(
                f"pool_size_frames must be a positive int, got {pool_size_frames!r}"
            )
        _require_positive_int("max_buffer_len", max_buffer_len)

        # Initialize basic attributes first to prevent errors during cleanup if initialization fails
        self._output = output
        self._device = device
        self._init_error: Optional[Exception] = None
        # Every adapter that could duplicate this output, in preference order
        # as the factory ranked it. On a hybrid system the adapter that owns
        # the output is not necessarily the one Desktop Duplication accepts,
        # and with prefer_integrated the iGPU may deliberately rank ahead of
        # `device` -- so this order is honoured rather than overridden.
        #
        # `device` is included even when the caller omits it: _build_duplicator
        # reassigns self._device to whichever adapter wins, so a set missing one
        # would shrink on every fallback and strand that adapter.
        candidates = list(candidate_devices or [])
        if not any(d is device for d in candidates):
            candidates.insert(0, device)
        self._all_devices = candidates
        self._timeout_ms = timeout_ms
        self._duplicator = None
        self._stagesurf = None
        self._processor = None
        self._pool_output = pool_output
        self._output_pool = None
        self._output_pool_size = pool_size_frames
        # Outstanding GPU frame handed out by grab_frame(). Capture cannot
        # acquire again until it is released, so this is tracked explicitly to
        # give a clear error instead of an opaque DXGI_ERROR_INVALID_CALL.
        self._live_frame = None
        self.is_capturing = False
        self._capture_thread = None
        self._capture_lock = Lock()
        # Serialises every use of the duplicator, staging surface and immediate
        # context: acquire, copy, map, release, rebuild. `_capture_lock` guards
        # only the continuous-mode deque, so two threads could otherwise call
        # AcquireNextFrame on one duplication (DXGI_ERROR_INVALID_CALL), or
        # have a rebuild null the context mid-copy. Re-entrant because a grab
        # can trigger a rebuild on the same thread.
        self._duplication_lock = RLock()
        self._stop_capture_event = Event() 
        self._frame_available_event = Event() 
        
        # For continuous mode buffer using PooledBuffer wrappers
        self._pooled_frames_deque: Optional[collections.deque] = None 
        self.max_buffer_len = max_buffer_len 
        # The newest queued frame, which video_mode duplicates while the screen
        # is idle. Shared rather than local to the capture thread because
        # get_latest_frame_buffer() hands frames out, and the producer must not
        # copy from one it no longer owns. Guarded by _capture_lock.
        self._last_dup_source = None

        self._timer_handle = None 
        self._frame_count = 0 
        self._capture_start_time = 0 
        self.rotation_angle = 0
        self.width = 0
        self.height = 0
        self.region = None
        self._region_set_by_user = False
        self._sourceRegion = None
        self.shot_w = 0
        self.shot_h = 0
        self.max_buffer_len = max_buffer_len
        self.buffer = False
        self._buffer_lock = Lock() 
        self.cursor = False
        self.memory_pool = None 
        # Reused staging buffer for off-pool grab(region=...) shapes. See
        # _scratch_staging_buffer().
        self._scratch_staging = None
        
        # Phase 2: Re-initialization state variables
        self._is_initialized = False
        self._needs_reinit = False
        self._released = False
        # Recovery is observable, not just survivable. A consumer caching
        # anything derived from a frame -- a GPU preprocessor, a cross-adapter
        # transfer, a resize table -- needs to know the duplicator underneath it
        # was replaced, because the replacement may differ in size, rotation or
        # format. `Frame.generation` stamps each frame with the value current
        # when it was taken.
        self._generation = 0
        self._sequence = 0
        self._recovery_count = 0
        self._last_recovery_reason = None
        self._reinit_attempts = 0
        self._max_reinit_attempts = 5
        self._reinit_backoff_seconds = [0.5, 1.0, 2.0, 3.0, 5.0] # Or generate dynamically
        # Exclusive-fullscreen / mode-switch transitions can refuse duplication
        # for a few hundred ms. Bound the retries so a permanent refusal fails
        # loudly instead of spinning forever.
        self._max_output_change_retries = 12
        self._capture_permanently_failed = False
        self._last_capture_error_message = ""
        
        # "Nothing is arriving" is a question about elapsed time, not about how
        # many times we asked. Counting consecutive empty acquires made the
        # answer depend on `timeout_ms`: with the 10 ms default, 100 misses mean
        # a second of genuinely still screen, but with a polling timeout of 0 the
        # same 100 misses take under 20 ms. That fired the warning seven times
        # while capture was running at 117 fps -- a message that is not merely
        # noisy but the opposite of true.
        self._last_frame_time = None            # set on the first successful grab
        self._quiet_warning_after_s = 2.0
        self._last_quiet_warning = 0.0
        
        # Store initial constructor arguments for re-initialization
        self._init_args = {
            "output": output, # This is an object, direct use might be tricky if it becomes invalid
            "device": device, # Same as output
            "region": region, # Value type, safe
            "output_color": output_color, # Value type, safe
            "nvidia_gpu": nvidia_gpu, # Value type, safe
            "pool_size_frames": pool_size_frames # Value type, safe
        }
        # For re-creating device and output, we might need display_idx/output_idx if original objects become stale.
        # This part needs careful thought if Device/Output objects themselves can become invalid.
        # Assuming for now that the passed device/output objects are stable or re-creatable from stored indices.
        # Storing original indices if available from device/output objects:
        self._display_idx = device.display_idx if hasattr(device, 'display_idx') else 0 # Example
        self._output_idx = output.output_idx if hasattr(output, 'output_idx') else 0 # Example
        
        try:
            if not self._initialize_resources():
                # Prefer the specific cause _initialize_resources recorded; the
                # generic message is only for the case where nothing was.
                raise self._init_error or RapidShotError(
                    "Initial resource initialization failed. Check logs for details.")

        except Exception as e: # Catch errors from _initialize_resources or other __init__ steps
            logger.error(f"Critical error during ScreenCapture __init__: {e}")
            # Ensure cleanup of any partially initialized resources
            self.release() # Call release to clean up whatever was set up
            raise # Re-raise the exception to signal construction failure
            
    def _device_label(self, device: Device) -> str:
        """Adapter description, for messages. Never raises."""
        try:
            return str(device.desc.Description)
        except Exception:
            return repr(device)

    def _build_duplicator(self) -> Duplicator:
        """Create a Duplicator, trying each candidate adapter in turn.

        Which adapter Desktop Duplication will accept cannot be deduced from
        which adapter enumerates the output. On a hybrid system the adapter
        owning the display can refuse while another succeeds, so the working
        pairing is found by trying rather than by assuming -- previously the
        display-owning adapter was assumed and a refusal ended capture.

        ``self._device`` is updated to whichever adapter worked, because the
        stage surface must be built on the same device as the duplicated
        texture; the two cannot be chosen independently.

        Raises:
            RapidShotConfigError: every candidate refused. The message names
                the likely system-level cause, which the raw HRESULT did not.
        """
        # Try in the stored preference order. Not "current device first":
        # `self._device` starts as the adapter that owns the output, and with
        # prefer_integrated the iGPU is deliberately ranked ahead of it, so
        # promoting the current device would silently override the request.
        # The winner is moved to the front *after* it succeeds, which keeps
        # rebuilds cheap without pre-empting the preference on the first
        # attempt. The set is never reduced -- an adapter that refuses once may
        # be the only valid one after a MUX or output change.
        candidates = list(self._all_devices)
        refusals = []
        for device in candidates:
            try:
                duplicator = Duplicator(
                    output=self._output, device=device,
                    timeout_ms=self._timeout_ms,
                )
            except RapidShotConfigError as e:
                # Only an adapter-specific refusal is worth retrying elsewhere.
                # A desktop refusal is also a RapidShotConfigError, but applies
                # to every adapter equally and already carries an actionable
                # message, so it propagates untouched instead of being retried
                # against adapters that will refuse it for the same reason.
                if getattr(e, "hresult", None) not in (
                    DXGI_ERROR_UNSUPPORTED,
                    DXGI_ERROR_INVALID_CALL,
                ):
                    raise
                label = self._device_label(device)
                refusals.append(f"  {label}: {e}")
                logger.info(
                    f"{label} refused to duplicate {self._output.devicename}; "
                    "trying the next adapter."
                )
                continue
            if device is not self._device:
                logger.warning(
                    f"Duplication is running on {self._device_label(device)}, "
                    f"which does not own {self._output.devicename} "
                    f"({self._device_label(self._device)} does)."
                )
                self._device = device
            # Winner first for the next rebuild: cheap, and it can only
            # reorder, never hide an adapter. Identity, not equality -- Device
            # defines no __eq__ and two adapters must never compare equal here.
            self._all_devices = [device] + [
                d for d in self._all_devices if d is not device
            ]
            return duplicator

        raise RapidShotConfigError(
            probe_topology().duplication_failure_help()
            + "\n\nWhat each adapter reported:\n"
            + "\n".join(refusals)
        )

    def _initialize_resources(self, is_reinit=False) -> bool:
        """
        Initializes or re-initializes DXGI/D3D resources (Device, Output, Duplicator).
        Also re-initializes StageSurface, Processor, and MemoryPool if needed.

        **Not safe to call from outside the capture thread while capture is
        running.** It sets ``_duplicator`` to None part-way through, and a
        concurrent ``_grab_locked`` dereferences it: calling this from another
        thread mid-capture raises ``AttributeError: 'NoneType' object has no
        attribute '_frame_acquired'`` in the capture thread. Nothing in the
        library does that -- the capture thread reaches this through ``_grab``
        via ``_needs_reinit``, and ``_on_output_change`` holds the duplication
        lock -- but it is not a constraint the code states anywhere else.
        """
        logger.info(f"{'Re-initializing' if is_reinit else 'Initializing'} capture resources...")
        
        # 1. Clean up existing resources (if any)
        if hasattr(self, '_duplicator') and self._duplicator:
            self._duplicator.release()
            self._duplicator = None
        if hasattr(self, '_stagesurf') and self._stagesurf:
            self._stagesurf.release()
            self._stagesurf = None
        # Device and Output are more complex. If they are passed in, re-getting them might be needed.
        # For now, assume self._device and self._output are either still valid or are re-created.
        # If they are from initial args, and can become stale, this needs more robust handling
        # (e.g. re-calling rapidshot.get_device, rapidshot.get_output based on stored indices).
        
        # For simplicity in this phase, let's assume self._device and self._output are either:
        # a) The initially provided valid objects (if not is_reinit)
        # b) Re-acquired if is_reinit (this part is complex if original handles are stale)
        # Let's simulate re-acquiring for reinit, assuming we have stored indices.
        if is_reinit:
            try:
                logger.debug(f"Re-creating device and output for display {self._display_idx}, output {self._output_idx}")
                # These get_device/get_output calls might not exist in this class directly.
                # This implies ScreenCapture needs access to the global factory functions.
                # For now, this is a placeholder for how Device/Output might be refreshed.
                # If the original device/output objects are stateful and become invalid,
                # they MUST be recreated.
                # Let's assume for now the stored self._device and self._output are updated externally or are robust.
                # If not, this is a major point of failure for re-initialization.
                # For now, we'll proceed assuming self._device and self._output are valid/refreshed.
                # This part of re-initialization (Device/Output) might need to live higher up,
                # e.g. in a factory that creates ScreenCapture, or ScreenCapture needs display_idx.
                
                # A pragmatic approach for now: if re-init, we trust the existing self._device, self._output
                # have been externally managed/updated or are somehow still valid for re-creating Duplicator.
                # This is a known simplification.
                self._output.update_desc() # Try to update the existing output object
                self.width, self.height = self._output.resolution
                logger.info(f"Output description updated. New resolution: {self.width}x{self.height}")

            except Exception as e:
                logger.error(f"Failed to re-acquire/update device/output during re-initialization: {e}")
                self._is_initialized = False
                return False

        try:
            # Use init_args for properties that don't change or are value types
            current_region = self._init_args['region']
            output_color = self._init_args['output_color']
            nvidia_gpu = self._init_args['nvidia_gpu'] # self.nvidia_gpu should be set from this
            pool_size_frames = self._init_args['pool_size_frames']

            self.nvidia_gpu = nvidia_gpu # Ensure it's set before processor/pool

            # Check if GPU acceleration is requested but CuPy is not available
            if self.nvidia_gpu and not cupy_available():
                logger.warning("NVIDIA GPU acceleration requested but CuPy is not available. Falling back to CPU mode for re-init.")
                self.nvidia_gpu = False # Fallback for this attempt

            self.width, self.height = self._output.resolution # Get current resolution
            
            # A rebuild keeps the region the caller asked for -- including one
            # given to start(), which is not in _init_args. This used to reset
            # to the constructor's region, so continuous capture of a region
            # came back from a device loss capturing the whole screen.
            if not is_reinit:
                self._requested_region = current_region
            self.region = self._fit_requested_region()
            self._validate_region(self.region) # This updates self.region and shot_w, shot_h

            logger.debug(f"Creating Duplicator for output: {self._output.devicename}")
            self._duplicator = self._build_duplicator()
            
            logger.debug(f"Creating StageSurface for output: {self._output.devicename}")
            self._stagesurf = StageSurface(output=self._output, device=self._device)
            
            logger.debug(f"Creating Processor with color: {output_color}, GPU: {self.nvidia_gpu}")
            self._processor = Processor(output_color=output_color, nvidia_gpu=self.nvidia_gpu)
            
            self._sourceRegion = D3D11_BOX(
                left=0, top=0, right=self.width, bottom=self.height, front=0, back=1
            )
            self.rotation_angle = self._output.rotation_angle
            self.output_color = output_color 

            # Re-initialize Memory Pool
            if self.memory_pool: # Destroy existing pool before creating a new one
                # Queued frames go back *first*. They are sized for the old
                # resolution and belong to the pool about to be destroyed, and a
                # wrapper that outlives its pool releases into a dead one, which
                # refuses the check-in and drops the buffer instead.
                #
                # This used to happen after the new pool was already in place,
                # and only when `continuous_mode` was True -- a flag nothing
                # ever set. So it never ran: frames from before the display
                # change stayed queued, and get_latest_frame() handed them out
                # as though they were current.
                self._drain_frame_queue()
                logger.debug("Destroying existing memory pool before re-initialization.")
                self.memory_pool.destroy_pool()
            
            buffer_shape = self._staging_shape(self.region)
            dtype = np.uint8

            staging_buffers = self._staging_pool_size(pool_size_frames)
            logger.debug(f"Initializing new memory pool with shape {buffer_shape}, {staging_buffers} buffers.")
            if self.nvidia_gpu:
                self.memory_pool = CupyMemoryPool(buffer_shape, dtype, staging_buffers)
            else:
                self.memory_pool = NumpyMemoryPool(buffer_shape, dtype, staging_buffers)
            
            self._is_initialized = True
            self._needs_reinit = False # Successfully re-initialized (or initialized)
            if is_reinit: # Only reset attempts if this was a re-initialization
                self._reinit_attempts = 0
                # Counted only on success. A failed attempt that will be retried
                # has not replaced anything, so stamping frames with a new
                # generation for it would invalidate consumers' caches for no
                # reason.
                self._generation += 1
                self._recovery_count += 1
                logger.info(
                    f"Capture recovered (generation {self._generation}, "
                    f"reason: {self._last_recovery_reason or 'unspecified'})")
            logger.info("Capture resources successfully initialized.")
            return True

        except (RapidShotConfigError, RapidShotDeviceError, RapidShotDXGIError, RapidShotError) as e:
            logger.error(f"Failed to {'re-initialize' if is_reinit else 'initialize'} resources: {e}")
            self._is_initialized = False
            # Keep the real cause. This used to be logged and dropped, and the
            # caller then raised "Check logs for details" -- which discarded a
            # diagnosis that had already been made, in the one situation where
            # the caller most needs it.
            self._init_error = e
            # self.release() # Clean up anything that might have been created
            return False
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error during resource {'re-initialization' if is_reinit else 'initialization'}: {e}")
            self._is_initialized = False
            self._init_error = e
            # self.release()
            return False

    def _attempt_reinitialization(self) -> bool:
        """
        Attempts to re-initialize capture resources after a recoverable error.
        Manages retries and backoff periods.
        """
        if self._capture_permanently_failed:
            logger.warning("Re-initialization attempt skipped: Capture is permanently failed.")
            return False

        self._reinit_attempts += 1
        logger.warning(f"Re-initialization attempt {self._reinit_attempts} of {self._max_reinit_attempts} scheduled.")
        # No "attempts exceeded" check here: the failure of the last permitted
        # attempt marks capture permanently failed below, and that is tested
        # first, so the count can never pass the maximum on this path.

        backoff_idx = min(self._reinit_attempts - 1, len(self._reinit_backoff_seconds) - 1)
        wait_time = self._reinit_backoff_seconds[backoff_idx]
        logger.info(f"Waiting for {wait_time:.1f} seconds before re-initialization attempt...")
        time.sleep(wait_time)

        logger.info(f"Attempting re-initialization (attempt {self._reinit_attempts}/{self._max_reinit_attempts})...")
        if self._initialize_resources(is_reinit=True):
            logger.info("Re-initialization successful.")
            self._needs_reinit = False # Clear the flag as we succeeded
            return True
        else:
            logger.warning(f"Re-initialization attempt {self._reinit_attempts} failed.")
            # If this was the last attempt, mark as permanently failed
            if self._reinit_attempts == self._max_reinit_attempts:
                self._capture_permanently_failed = True
                self._last_capture_error_message = f"Re-initialization failed after {self._max_reinit_attempts} attempts."
                logger.error(self._last_capture_error_message)
            return False

    def region_to_memory_region(self, region: Tuple[int, int, int, int], rotation_angle: int, output: Output):
        """
        Convert a screen region to memory region based on rotation angle.
        
        Args:
            region: Region to convert (left, top, right, bottom)
            rotation_angle: Rotation angle (0, 90, 180, 270)
            output: Output device
            
        Returns:
            Converted region
        """
        left, top, right, bottom = region

        if rotation_angle != output.rotation_angle:
            raise AssertionError(
                f"Rotation mismatch: capture reports {rotation_angle} but output is {output.rotation_angle}"
            )

        # `region` is in desktop coordinates, which follow the rotation; the
        # duplicated texture is in the panel's native orientation and is
        # `surface_size` big. At 90/270 the two are transposed, so each formula
        # must reverse an axis by that axis's own extent: at 90 the texture's
        # rows run along the desktop's width (= surface_height), at 270 its
        # columns run along the desktop's height (= surface_width).
        #
        # These used to subtract the other dimension. On any non-square panel
        # a full-screen region then mapped outside the texture -- 1080x1920
        # desktop at 270 gave left = -840 -- and CopySubresourceRegion silently
        # skipped the copy, handing back a stale frame.
        surface_width, surface_height = output.surface_size

        if rotation_angle == 0:
            return (left, top, right, bottom)
        if rotation_angle == 90:
            return (top, surface_height - right, bottom, surface_height - left)
        if rotation_angle == 180:
            return (
                surface_width - right,
                surface_height - bottom,
                surface_width - left,
                surface_height - top,
            )
        if rotation_angle == 270:
            return (
                surface_width - bottom,
                left,
                surface_width - top,
                right,
            )

        raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")

    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Union[np.ndarray, Any]]: # Any can be PooledBuffer
        """
        Grab a single frame from the screen.
        Uses the memory pool if the requested region matches the pool's buffer configuration.
        
        Args:
            region: Region to capture (left, top, right, bottom). 
                    If None, uses self.region.
            
        Returns:
            A PooledBuffer wrapper (if pool was used and buffer is valid),
            a NumPy/CuPy array (if pool was bypassed or buffer became invalid), 
            or None if no update or error.
        """
        # Checked here rather than inside _grab(): that method has a catch-all
        # handler which would swallow this into a None return, hiding a caller
        # bug that stalls capture.
        self._ensure_no_live_frame("grab()")
        self._refuse_while_capturing("grab()")

        current_region_tuple: Tuple[int, int, int, int]
        if region is None:
            current_region_tuple = self.region
        else:
            current_region_tuple = self._normalize_region(region)

        return self._grab(current_region_tuple)

    def _checkout_output_buffer(self, width: int, height: int):
        """A pooled buffer for the converted frame, or None if not in use.

        On by default since 2.0. The buffer must be released by whoever receives
        it; ``rapidshot.create(pool_output=False)`` restores the pre-2.0
        behaviour of allocating a fresh array per frame, which costs ~1.6 ms on
        a 1080p RGB frame but needs no release.
        """
        if (not self._pool_output or not self._processor.converts_output
                or not getattr(self._processor, "accepts_output_target", False)):
            # A backend that cannot write into a target (CuPy) allocates its
            # own result. Checking a buffer out anyway allocated a host pool of
            # pool_size_frames full frames and took and returned one per frame
            # for nothing.
            return None

        shape = (height, width, self._processor.output_channels)
        pool = self._output_pool
        if pool is None or tuple(pool.buffer_shape) != shape:
            if pool is not None:
                pool.destroy_pool()
            from rapidshot.memory_pool import NumpyMemoryPool
            pool = NumpyMemoryPool(shape, np.uint8, self._output_pool_size)
            self._output_pool = pool
        try:
            return pool.checkout()
        except PoolExhaustedError:
            # Every buffer is still out with a caller. Fall back to allocating,
            # which is slower but always correct -- far better than blocking
            # capture or recycling a buffer somebody is still reading.
            logger.debug("Output pool exhausted; allocating for this frame.")
            return None

    def _staging_pool_size(self, pool_size_frames: int) -> int:
        """How many BGRA staging buffers the pool actually needs.

        A **converting** mode releases its staging buffer inside the same
        ``grab()``: the caller receives the *output* buffer and the staging one
        is, in `_grab_locked`'s own words, "finished with either way".
        `_grab_locked` runs under the duplication lock, so exactly one staging
        buffer is ever in flight and the other ``pool_size_frames - 1`` are
        unreachable. At 2560x1600 that was **49.2 MB** of buffers nothing could
        hand out, allocated at ``create()`` for every RGB camera.

        **BGRA is the exception, and the reason the pool is sized this way at
        all.** It converts nothing, so the staging buffer *is* the frame the
        caller receives and holds until release -- and in video mode the
        capture thread checks out more of them to fill
        ``_pooled_frames_deque``. Those need the full count, and
        ``pool_size_frames`` keeps meaning exactly what it documents there.

        This is the argument already made for :meth:`_scratch_staging_buffer`,
        which is one buffer for one reason: only one grab runs at a time.
        """
        processor = getattr(self, "_processor", None)
        converts = getattr(processor, "converts_output", None)
        if converts is None:
            # Sized before a processor exists, or one that cannot say. Assume
            # the worst case rather than guess: guessing high wastes buffers,
            # guessing low exhausts the pool at runtime.
            return pool_size_frames
        if getattr(self, "_pool_output", True) and not converts:
            return pool_size_frames
        return 1

    def _fit_requested_region(self) -> Tuple[int, int, int, int]:
        """The region to capture after a rebuild, at the current resolution.

        The caller's explicit request -- ``create(region=...)`` or
        ``start(region=...)`` -- when it still fits; otherwise the full screen.
        The request itself is kept either way, so a resolution that drops and
        comes back restores it instead of forgetting it.

        Previously a rebuild reset the region to the constructor's, which lost
        any region given to ``start()``, and an ``_on_output_change`` whose
        user region no longer fit raised ``ValueError`` out of the rebuild.
        """
        full = (0, 0, self.width, self.height)
        requested = self.__dict__.get("_requested_region")
        if requested is None and getattr(self, "_region_set_by_user", False):
            # Flagged as the user's without a recorded request: the region
            # currently set is the request.
            requested = self.region
        self._region_set_by_user = requested is not None
        if requested is None:
            return full
        left, top, right, bottom = requested
        if right <= self.width and bottom <= self.height:
            return tuple(requested)
        logger.warning(
            f"Requested region {tuple(requested)} does not fit the new "
            f"{self.width}x{self.height} resolution; capturing the full screen "
            "until it does.")
        return full

    def _staging_shape(self, region: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
        """(rows, columns, 4) of the BGRA staging read for a desktop *region*.

        The staging surface holds the region in the **panel's** orientation --
        ``region_to_memory_region`` of it -- and the processor rotates to the
        desktop's afterwards. At 90 and 270 degrees those are transposed.

        Every staging buffer used to be sized from the desktop region instead.
        The processor refuses a buffer of the wrong shape, so on a rotated
        portrait display every grab() failed: silently, as a black BGRA frame,
        until processing errors were made to raise; after that, as None and a
        recovery that failed the same way until capture gave up.
        """
        left, top, right, bottom = self.region_to_memory_region(
            region, self.rotation_angle, self._output)
        return (bottom - top, right - left, 4)

    def _scratch_staging_buffer(self, height: int, width: int):
        """A BGRA staging buffer for a region the pool does not cover.

        ``grab(region=...)`` whose shape differs from the pool's allocated a
        fresh buffer on every call. The allocation itself is cheap; filling it
        is not, because every page faults on first touch -- measured here at
        0.15 ms for a 400x400 region and 1.9 ms at 2560x1600, which is 77-94%
        of the cost of writing the buffer at all. It is the same effect
        ``pool_output`` exists to avoid, on the path that was still paying it.

        Reused **only when the processor converts the frame**. Then this buffer
        is a pure intermediate and the caller receives a different array. BGRA
        does no conversion, so this buffer *is* the frame that goes back to the
        caller, and reusing it would hand successive callers the same memory --
        the aliasing the pooled path raises ``BufferReleasedError`` to prevent.
        So BGRA keeps allocating per frame, which for that mode is the price of
        a frame the caller owns.

        One buffer rather than a pool: ``_grab_locked`` runs under the
        duplication lock, so only one grab can be using it at a time. It costs
        one region-sized buffer, held until the shape changes or the camera is
        released, and nothing at all for a camera that never grabs off-pool.
        """
        shape = (height, width, 4)

        if not self._processor.converts_output:
            if self.nvidia_gpu:
                cp = _require_cupy()
                return cp.empty(shape, dtype=cp.uint8)
            return np.empty(shape, dtype=np.uint8)

        scratch = getattr(self, "_scratch_staging", None)
        if scratch is None or scratch.shape != shape:
            if self.nvidia_gpu:
                cp = _require_cupy()
                scratch = cp.empty(shape, dtype=cp.uint8)
            else:
                scratch = np.empty(shape, dtype=np.uint8)
            self._scratch_staging = scratch
        return scratch

    def _sync_accumulator(self, memory_region):
        """Drop the accumulated frame unless this frame directly follows it.

        Returns the identity of the current frame, which the caller records
        with :meth:`_record_accumulator_frame` once the processor has actually
        folded it in.

        The accumulator is a converted copy of the last frame ``grab()``
        processed, and this frame's dirty rects describe the change since the
        *previous frame the duplicator acquired*. The two are the same frame
        only if nothing else acquired one in between. ``shot()`` and
        ``grab_frame()`` both do, and patching onto the accumulator after them
        produced a frame that was 99.5% the older image: every pixel changed in
        the frame they consumed was missed, with nothing to show for it.

        Checking the sequence rather than invalidating at each of those call
        sites covers every consumer at once, including ones added later. A new
        duplicator (an output change or a recovery) has a new ``instance_id``,
        so it can never be taken for a continuation of the old one either.

        Shape alone is not identity for the region, too. Alternating between
        two same-sized regions would otherwise patch one region's dirty rects
        onto the other region's pixels.
        """
        duplicator = self._duplicator
        serial = getattr(duplicator, "frame_serial", None)
        frame = None if serial is None else (getattr(duplicator, "instance_id", None), serial)
        follows = (frame is not None
                   and getattr(self, "_accumulator_frame", None) == (frame[0], frame[1] - 1))
        # A duplicator that cannot say which frame this is gets no benefit of
        # the doubt: an unnecessary full conversion is slow, a wrong patch is
        # silently wrong.
        if getattr(self, "_accumulator_region", None) != memory_region or not follows:
            invalidate = getattr(self._processor, "invalidate_accumulator", None)
            if invalidate is not None:
                invalidate()
            self._accumulator_region = memory_region
        # Forgotten until the processor has absorbed this frame. If anything
        # between here and there fails, the accumulator still holds the older
        # frame, and it must not be recorded as holding this one.
        self._accumulator_frame = None
        return frame

    def _record_accumulator_frame(self, frame) -> None:
        """Note that the accumulator now reflects ``frame``, as returned by
        :meth:`_sync_accumulator`."""
        self._accumulator_frame = frame

    def _dirty_rects_for(self, memory_region) -> Optional[list]:
        """This frame's dirty rects, translated into the staging surface.

        DXGI reports them in desktop coordinates, but the staging surface holds
        only ``memory_region`` — ``CopySubresourceRegion`` already cropped it.
        So the rects have to be clipped to that region and rebased to its
        top-left, or they would address the wrong pixels whenever a region is in
        use, and outside the buffer entirely when it is off-origin.

        Returns None when the metadata is unavailable, which the processor
        reads as "convert everything".

        A frame whose move rects are **unknown** is treated the same way, for
        the same reason: `None` from the duplicator means the metadata could
        not be read, not that there was none of it -- that is `[]`. Since every
        `None` is a genuine error rather than an ordinary empty frame, the
        full convert it forces is rare.

        A frame carrying **move** rects is treated the same way. The compositor
        satisfied part of it by copying pixels already on screen, and DXGI does
        not repeat those regions in the dirty rects -- so patching by dirty rect
        alone would leave the moved region showing the previous frame. Redrawing
        everything is the only answer that is right without implementing the
        copy, and it costs nothing where move rects never appear.

        Measured on this machine (Windows 11, 2560x1600): across 3,768 frames of
        window dragging and page scrolling, DWM reported **zero** move rects,
        with the metadata readable on every frame. With everything composited
        into per-window surfaces there is nothing left for a screen-to-screen
        blit to optimise. This branch is correctness insurance for the
        configurations where that is not true, not a path this hardware takes.
        """
        # `[]` is "the frame carried no move metadata"; `None` is "it could
        # not be read". The duplicator distinguishes them deliberately, and a
        # truthiness check collapsed the two -- so an unreadable frame was
        # patched by dirty rect alone, leaving any moved region showing the
        # previous contents with nothing to say so. Unknown is treated as
        # "there may have been moves", which is how unreadable *dirty*
        # metadata is already treated three lines below.
        moves = getattr(self._duplicator, "move_rects", None)
        if moves is None or moves:
            return None

        rects = getattr(self._duplicator, "dirty_rects", None)
        if not rects:
            return rects if rects is None else []

        left, top, right, bottom = memory_region
        clipped = []
        for rl, rt, rr, rb in rects:
            nl, nt = max(rl, left), max(rt, top)
            nr, nb = min(rr, right), min(rb, bottom)
            if nl < nr and nt < nb:
                clipped.append((nl - left, nt - top, nr - left, nb - top))
        return clipped

    def _dup_lock(self) -> RLock:
        """The duplication lock, created on first use if __init__ did not run.

        Test doubles are built with object.__new__ and never see __init__.
        """
        lock = self.__dict__.get("_duplication_lock")
        if lock is None:
            lock = self._duplication_lock = RLock()
        return lock

    def _refuse_while_capturing(self, caller: str) -> None:
        """Single-shot capture is not available while start() is running.

        The capture thread owns the duplicator for as long as it runs. A
        concurrent grab() would compete with it for frames, and grab_frame()
        would hold a DXGI frame the thread then cannot acquire past.

        grab() used to be meant to redirect to get_latest_frame() here, but the
        flag it tested was never set, so it raced the capture thread instead.
        A clear error is better than a redirect: the latest frame is a plain
        array with no release(), which code written for grab() would call.
        """
        if (getattr(self, "is_capturing", False)
                and current_thread() is not getattr(self, "_capture_thread", None)):
            raise RuntimeError(
                f"{caller} called while continuous capture is running. The "
                "capture thread owns the duplicator until stop(); read frames "
                "with get_latest_frame(), or call stop() first."
            )

    def _ensure_no_live_frame(self, caller: str) -> None:
        """
        Refuse to start a capture while a Frame still holds the desktop texture.

        DXGI would fail the acquire with DXGI_ERROR_INVALID_CALL, which gives no
        hint about the actual cause. Failing here names the problem instead.
        """
        live = self._live_frame
        if live is not None and not live.released:
            raise RuntimeError(
                f"{caller} cannot start: a Frame from grab_frame() has not been "
                "released. DXGI cannot acquire the next frame while a reference "
                "to the previous desktop surface is outstanding. Use `with "
                "camera.grab_frame() as frame:` so release happens automatically."
            )
        self._live_frame = None

    def grab_frame(self, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Capture a frame and hand back its GPU texture, skipping the CPU copy.

        This is the GPU-resident path. Unlike :meth:`grab`, no staging read and
        no color conversion happen — the caller receives the ``ID3D11Texture2D``
        DXGI produced, ready to pass to a GPU consumer (inference runtime,
        hardware encoder).

        The returned :class:`~rapidshot.frame.Frame` owns that texture for a
        bounded window and **must be released**. DXGI cannot acquire the next
        frame while a reference to the previous surface is outstanding, so a
        frame that is never released stalls capture completely. Use it as a
        context manager::

            with camera.grab_frame() as frame:
                do_gpu_work(frame.d3d11_texture)

        Args:
            region: Region metadata for the frame (left, top, right, bottom).
                Note the texture is the full desktop surface; the region is
                recorded on the frame rather than applied to the texture, since
                cropping would require a GPU copy this path exists to avoid.

        Returns:
            A Frame, or None if no new content was available.

        Raises:
            RuntimeError: If a previous Frame has not been released yet.
        """
        self._ensure_no_live_frame("grab_frame()")
        self._refuse_while_capturing("grab_frame()")

        if region is None:
            region = self.region
        else:
            region = self._normalize_region(region)

        with self._dup_lock():
            return self._grab_frame_locked(region)

    def _grab_frame_locked(self, region):
        """grab_frame() once the duplication lock is held."""
        from rapidshot.frame import Frame

        if self._capture_permanently_failed:
            logger.error(f"Capture permanently failed: {self._last_capture_error_message}")
            return None

        if self._needs_reinit and not self._attempt_reinitialization():
            return None

        if not self._is_initialized or self._duplicator is None:
            logger.error("grab_frame() called but capture resources are not initialized.")
            self._note_recovery_needed("capture resources not initialized")
            return None

        try:
            self._duplicator.update_frame()
        except RapidShotProtectedContentError as e:
            logger.error(f"Protected content blocks capture: {e}")
            self._last_capture_error_message = str(e)
            return None
        except (RapidShotReinitError, RapidShotDeviceError) as e:
            logger.warning(f"grab_frame(): {e}. Flagging for re-initialization.")
            self._note_recovery_needed("device or re-init error during grab_frame")
            return None
        except RapidShotError as e:
            logger.error(f"grab_frame(): {e}")
            return None

        if not self._duplicator.updated:
            # No new content. Any frame that was acquired still has to go back.
            if self._duplicator._frame_acquired:
                self._duplicator.release_frame()
            return None

        duplicator = self._duplicator
        frame = Frame(
            texture=duplicator.texture,
            on_release=duplicator.release_frame,
            region=region,
            rotation_angle=self.rotation_angle,
            present_time_qpc=duplicator.last_present_time,
            accumulated_frames=duplicator.accumulated_frames,
            source_id=duplicator.instance_id,
            protected_content=duplicator.protected_content_detected,
            cursor_visible=duplicator.cursor_visible,
            dirty_rects=duplicator.dirty_rects,
            move_rects=duplicator.move_rects,
            rects_coalesced=duplicator.rects_coalesced,
            sequence=self._next_sequence(),
            generation=self._generation,
            cursor=self._cursor_info(),
        )
        self._live_frame = frame
        return frame

    def _next_sequence(self) -> int:
        """Monotonic frame index for this camera, from 1.

        Continues across recoveries: a sequence number identifies one frame for
        the camera's whole life, which is what makes it useful in a log next to
        a generation.
        """
        self._sequence += 1
        return self._sequence

    def _cursor_info(self):
        """Snapshot the duplicator's cursor state into a plain CursorInfo.

        Copied rather than referenced because the duplicator mutates its Cursor
        in place on the next acquire, so a frame holding the live object would
        silently start describing a later cursor.
        """
        from rapidshot.frame import CursorInfo

        duplicator = self._duplicator
        cursor = getattr(duplicator, "cursor", None)
        if cursor is None:
            return CursorInfo(visible=bool(getattr(duplicator, "cursor_visible", False)))
        position = getattr(cursor, "PointerPositionInfo", None)
        shape_info = getattr(cursor, "PointerShapeInfo", None)
        point = getattr(position, "Position", None) if position is not None else None
        return CursorInfo(
            visible=bool(getattr(duplicator, "cursor_visible", False)),
            position=((int(point.x), int(point.y)) if point is not None else None),
            hotspot=((int(shape_info.HotSpot.x), int(shape_info.HotSpot.y))
                     if shape_info is not None and hasattr(shape_info, "HotSpot") else None),
            shape=getattr(cursor, "Shape", None),
            shape_type=int(getattr(shape_info, "Type", 0) or 0) if shape_info is not None else 0,
            shape_size=((int(shape_info.Width), int(shape_info.Height))
                        if shape_info is not None and hasattr(shape_info, "Width") else None),
            shape_pitch=int(getattr(shape_info, "Pitch", 0) or 0) if shape_info is not None else 0,
        )

    @property
    def generation(self) -> int:
        """How many times this camera has rebuilt its capture resources.

        0 until the first recovery. Compare against :attr:`Frame.generation` to
        detect that a held frame predates a rebuild.
        """
        return self._generation

    @property
    def last_capture_error(self) -> str:
        """Why the most recent capture failed, or ``""`` if none has.

        ``grab()`` returns None and ``shot()`` returns False both when nothing
        changed on screen and when capture failed; this is what tells the two
        apart. Kept until the next failure replaces it.
        """
        return self._last_capture_error_message

    @property
    def last_present_time(self) -> int:
        """QPC ticks when the compositor presented the newest captured frame.

        0 before the first frame, and again after a rebuild until the new
        duplicator delivers one. ``dxcam_compat`` reads this for DXcam's
        ``latest_frame_time``; it looked for it here, found nothing -- the value
        only ever lived on the duplicator -- and so reported 0 on every real
        camera. Its test passed because the fake camera it used defined the
        attribute.
        """
        return int(getattr(self._duplicator, "last_present_time", 0) or 0)

    @property
    def recovery_count(self) -> int:
        """Successful recoveries so far.

        Equal to :attr:`generation`; kept as a separate name because one reads
        as an identity for frames and the other as a health counter for the
        camera. A steadily climbing value on an otherwise idle machine is worth
        investigating -- capture is being torn down and rebuilt repeatedly.
        """
        return self._recovery_count

    @property
    def last_recovery_reason(self) -> "Optional[str]":
        """Why the most recent rebuild happened, or None if there has been none.

        A short human-readable cause -- access lost, mode change, device
        removed. Recorded when the rebuild is *scheduled*, so it survives to be
        read after the rebuild succeeds.
        """
        return self._last_recovery_reason

    def _note_recovery_needed(self, reason: str) -> None:
        """Record why a rebuild was scheduled. Does not itself rebuild."""
        self._last_recovery_reason = reason
        self._needs_reinit = True

    def grab_cursor(self):
        """
        Get cursor information.
        
        Returns:
            Cursor information
        """
        return self._duplicator.cursor

    def shot(
        self,
        image_ptr: Any,
        region: Optional[Tuple[int, int, int, int]] = None,
        buffer_size: Optional[int] = None,
    ) -> bool:
        """
        Capture directly into a caller-provided memory buffer.

        The buffer receives pixels in this instance's ``output_color`` format,
        matching what :meth:`grab` returns. Size it as
        ``width * height * channels`` where ``channels`` is 4 for BGRA/RGBA,
        3 for RGB/BGR and 1 for GRAY -- :attr:`bytes_per_frame` computes this
        for you.

        Prefer passing a NumPy array (or any sized buffer object): its size is
        read directly and validated before anything is written. A bare pointer
        carries no size, so one is only accepted together with ``buffer_size``.

        Args:
            image_ptr: Destination buffer -- a NumPy array, ctypes array,
                bytearray/memoryview, or a raw pointer plus ``buffer_size``
            region: Region to capture (left, top, right, bottom)
            buffer_size: Destination size in bytes; required only for raw pointers

        Returns:
            True if a new frame was written, False if there was no new content
            or the capture failed

        Raises:
            ValueError: If the destination is too small or its size is unknowable
            NotImplementedError: On the GPU backend (``nvidia_gpu=True``)
        """
        if image_ptr is None:
            raise ValueError("image_ptr cannot be None")
        self._refuse_while_capturing("shot()")
        processor = getattr(self, "_processor", None)
        if processor is not None and not processor.supports_direct_output:
            # Refused before capturing. Found later, the backend's error was a
            # capture failure: False, and a rebuild scheduled that cannot help.
            raise NotImplementedError(
                "shot() writes into host memory, which the GPU backend "
                "(nvidia_gpu=True) does not support; use grab() instead"
            )

        if region is None:
            region = self.region
        else:
            # Validate without adopting it. _validate_region() also assigns
            # self.region, so a one-off shot(region=...) used to change the
            # region every later grab() and start() captured.
            region = self._normalize_region(region)

        # Validate the destination up front, before any capture work. Deferring
        # this to the processor would make it fire only on the calls that
        # actually receive new frame content -- so an undersized buffer would
        # quietly return False on a static desktop and only raise later, once
        # something on screen happened to change.
        self._validate_destination(image_ptr, region, buffer_size)

        return self._shot(image_ptr, region, buffer_size)

    def _validate_destination(self, image_ptr, region, buffer_size) -> int:
        """
        Check that a shot() destination is large enough, before capturing.

        Args:
            image_ptr: Destination buffer or pointer
            region: Region that will be captured
            buffer_size: Caller-declared size in bytes, if any

        Returns:
            The required size in bytes

        Raises:
            ValueError: If the destination is too small, or its size is unknown
        """
        required = self.bytes_per_frame(region)
        address, detected_size = describe_destination(image_ptr)
        # 0 is an address as far as pointer_to_address is concerned, so
        # checks for None let `shot(0, buffer_size=n)` through to a memmove
        # into address 0: an access violation that ends the process.
        if not address:
            raise ValueError("shot() destination pointer is null")

        known_sizes = [s for s in (detected_size, buffer_size) if s is not None]
        if not known_sizes:
            raise ValueError(
                f"Cannot verify destination size for shot(): {region[2] - region[0]}x"
                f"{region[3] - region[1]} in {self.output_color} needs {required} "
                "bytes. Pass a NumPy array (or any sized buffer), or supply "
                "buffer_size explicitly -- writing through an unsized pointer "
                "risks corrupting memory."
            )

        smallest = min(known_sizes)
        if smallest < required:
            raise ValueError(
                f"Destination buffer is too small for shot(): {smallest} bytes "
                f"provided, {required} needed for {region[2] - region[0]}x"
                f"{region[3] - region[1]} in {self.output_color} "
                f"({self.channels} channel(s))."
            )
        return required

    @property
    def channels(self) -> int:
        """Number of channels frames from this instance carry."""
        return self._processor.output_channels

    @property
    def timeout_ms(self) -> int:
        """
        How long each acquire waits for the compositor to present a new frame.

        This trades CPU against throughput, and the trade is steep in one
        direction only. Measured on a 100 Hz output against a source presenting
        at ~610 updates/s:

        =========  ======  ======  ======
        timeout      fps    hit%    CPU%
        =========  ======  ======  ======
        0           127.8    2.4%   68.6
        1           119.2   74.5%   19.5
        5           118.2   95.3%   18.9
        10          118.9  100.0%   15.7
        =========  ======  ======  ======

        Frame rate barely moves across that range; CPU moves by more than 4x.
        Polling at ``0`` is what DXcam does -- it reached 134 fps at 66% CPU in
        the same comparison, so the extra frames are real, they are just
        expensive. Set ``0`` if capture is the only thing the machine is doing
        and you want every frame the compositor produces; leave the default if
        capture is one stage of a pipeline that needs its cores.

        Note that frames beyond the display's refresh rate were never shown as
        distinct images: useful as extra temporal samples for a model, redundant
        for a recorder.

        Reproduce with ``benchmarks/compare_libraries.py``.
        """
        return self._timeout_ms

    @timeout_ms.setter
    def timeout_ms(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"timeout_ms must be a non-negative int, got {value!r}"
            )
        self._timeout_ms = value
        # Apply to the live duplicator too, so the change takes effect on the
        # next acquire rather than at the next rebuild.
        if self._duplicator is not None:
            self._duplicator.timeout_ms = value

    def bytes_per_frame(self, region: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        Size in bytes that :meth:`shot` writes for *region*.

        Use this to allocate a destination buffer that is guaranteed to fit.

        Args:
            region: Region to measure; defaults to this instance's region

        Returns:
            Required buffer size in bytes
        """
        if region is None:
            region = self.region
        else:
            region = self._normalize_region(region)
        width = region[2] - region[0]
        height = region[3] - region[1]
        return width * height * self.channels

    def _shot(self, image_ptr, region, buffer_size=None) -> bool:
        """shot() under the duplication lock."""
        with self._dup_lock():
            return self._shot_locked(image_ptr, region, buffer_size)

    def _shot_locked(
        self,
        image_ptr,
        region: Tuple[int, int, int, int],
        buffer_size: Optional[int] = None,
    ) -> bool:
        """
        Internal implementation of shot.

        Args:
            image_ptr: Destination buffer or pointer to one
            region: Region to capture (left, top, right, bottom)
            buffer_size: Destination size in bytes, if known

        Returns:
            True if successful, False otherwise
        """
        self._ensure_no_live_frame("shot()")

        # The same contract as grab(): a capture that cannot happen returns
        # False and says why in last_capture_error; only a caller's mistake --
        # an unusable destination, rejected before this point -- raises. This
        # used to handle access loss and let everything else escape, so
        # protected content made shot() raise where grab() returned None, and a
        # camera that had given up or was mid-recovery was used as if healthy.
        if self._capture_permanently_failed:
            logger.error(f"shot(): capture permanently failed: {self._last_capture_error_message}")
            return False
        if self._needs_reinit and not self._attempt_reinitialization():
            return False
        if not self._is_initialized or self._duplicator is None:
            self._note_recovery_needed("capture resources not initialized")
            return False

        try:
            duplication_healthy = self._duplicator.update_frame()
        except (RapidShotReinitError, RapidShotDeviceError) as e:
            # Access lost / device reset: rebuild, then let the caller retry.
            logger.warning(f"shot(): {e}. Rebuilding capture resources.")
            self._last_capture_error_message = str(e)
            self._on_output_change()
            return False
        except RapidShotError as e:
            # Protected content included: no rebuild can fix that while the
            # content is on screen.
            logger.error(f"shot(): {e}")
            self._last_capture_error_message = str(e)
            return False

        if duplication_healthy:
            frame_needs_release = self._duplicator._frame_acquired
            mapped_rect = None
            try:
                if not self._duplicator.updated:
                    # No new content within the acquire timeout. That is the
                    # normal state of a static desktop, not an output change --
                    # rebuilding here used to stall every idle shot() call.
                    return False

                _region = self.region_to_memory_region(region, self.rotation_angle, self._output)
                _width = _region[2] - _region[0]
                _height = _region[3] - _region[1]

                if self._stagesurf.width != _width or self._stagesurf.height != _height:
                    self._stagesurf.release()
                    self._stagesurf.rebuild(output=self._output, device=self._device, dim=(_width, _height))

                source_region = D3D11_BOX(
                    left=_region[0],
                    top=_region[1],
                    right=_region[2],
                    bottom=_region[3],
                    front=0,
                    back=1,
                )

                self._device.im_context.CopySubresourceRegion(
                    self._stagesurf.texture,
                    0,
                    0,
                    0,
                    0,
                    self._duplicator.texture,
                    0,
                    ctypes.byref(source_region),
                )

                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                    frame_needs_release = False

                mapped_rect = self._stagesurf.map()
                try:
                    if self.rotation_angle == 0:
                        self._processor.process2(
                            image_ptr,
                            mapped_rect,
                            _width,
                            _height,
                            buffer_size,
                        )
                    else:
                        self._shot_rotated(image_ptr, mapped_rect, _width, _height)
                finally:
                    self._stagesurf.unmap()
                return True
            except Exception as e:
                # The destination was validated before capturing, so a failure
                # from here on is the capture's -- a device that stopped
                # answering, a mapping with no pointer -- and handled as grab()
                # handles it: logged, recorded, recovery scheduled.
                logger.error(f"shot(): capture failed: {e}")
                self._last_capture_error_message = f"shot() failed: {e}"
                self._note_recovery_needed("unhandled error during shot")
                return False
            finally:
                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
        else:
            self._on_output_change()
            return False

    def _shot_rotated(self, image_ptr, mapped_rect, width: int, height: int) -> None:
        """shot() on a rotated display: turn the frame, then copy it across.

        The direct path converts straight into the caller's memory, but it
        writes the staging surface as-is -- the panel's orientation. On a
        rotated display that is not what grab() returns, which is the promise
        shot() makes: at 180 degrees every pixel was in the wrong place, and at
        90 and 270 the bytes were a transposed image the caller's
        ``(height, width, channels)`` view would shear.

        Rotating in place is not possible (90 and 270 change the shape), so
        this takes the processor's rotated frame and copies it. One frame-sized
        allocation, on a path that was producing wrong pixels without it.

        The size is re-checked rather than assumed. ``_validate_destination``
        sizes the buffer before capturing and the rotated frame should have
        exactly that many bytes -- but "should" was the whole argument, and
        ``describe_destination`` was already returning the size this then threw
        away. Getting it wrong writes past the end of the caller's memory,
        which is the failure `shot()` itself refuses two checks earlier.
        """
        frame, _ = self._processor.process(
            mapped_rect, width, height, (0, 0, width, height),
            self.rotation_angle, None)
        frame = np.ascontiguousarray(frame)
        address, detected_size = describe_destination(image_ptr)
        if not address:
            raise ValueError("Invalid destination pointer for shot copy")
        if detected_size is not None and detected_size < frame.nbytes:
            raise ValueError(
                f"Destination buffer is too small for shot() on a rotated "
                f"display: {detected_size} bytes provided, {frame.nbytes} "
                f"needed for the rotated {frame.shape} frame."
            )
        ctypes.memmove(address, frame.ctypes.data, frame.nbytes)

    def _grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """_grab_locked() under the duplication lock.

        The capture thread and grab() both land here, so this is the one place
        that has to serialise them.
        """
        with self._dup_lock():
            return self._grab_locked(region)

    def _grab_locked(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Grab a frame with a specific region with improved error handling.

        Args:
            region: Region to capture (left, top, right, bottom)

        Returns:
            ndarray: Captured frame
        """
        try:
            if self._capture_permanently_failed:
                logger.error(f"Capture is permanently failed: {self._last_capture_error_message}")
                return None

            if self._needs_reinit:
                if not self._attempt_reinitialization():
                    logger.error(
                        "Re-initialization failed, current grab cannot proceed. "
                        f"Permanent failure: {self._capture_permanently_failed}"
                    )
                    return None

            if not self._is_initialized or self._duplicator is None:
                logger.error("Attempted to grab frame but capture resources are not initialized.")
                self._note_recovery_needed("capture resources not initialized")
                return None

            pooled_buffer_wrapper = None
            output_wrapper = None
            output_array_for_region = None
            can_use_pool = False

            staging_shape = self._staging_shape(region)
            if self.memory_pool:
                if tuple(self.memory_pool.buffer_shape) == staging_shape:
                    can_use_pool = True

            if can_use_pool:
                pooled_buffer_wrapper = self.memory_pool.checkout()
                output_array_for_region = pooled_buffer_wrapper.array
            else:
                logger.debug(
                    f"Region {region} not matching pool config. Using temporary buffer for this grab."
                )
                output_array_for_region = self._scratch_staging_buffer(
                    staging_shape[0], staging_shape[1]
                )

            try:
                self._duplicator.update_frame()
            except RapidShotReinitError as e:
                logger.warning(f"DXGI Re-init error during update_frame: {e}. Flagging for re-initialization.")
                self._note_recovery_needed("DXGI re-init error during update_frame")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotDeviceError as e:
                logger.error(f"DXGI Device error during update_frame: {e}. Flagging for re-initialization.")
                self._note_recovery_needed("DXGI device error during update_frame")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotProtectedContentError as e:
                # Not recoverable by retrying: the OS is refusing while the
                # protected surface is on screen. Do not enter the re-init loop.
                logger.error(f"Protected content blocks capture: {e}")
                self._last_capture_error_message = str(e)
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotDXGIError as e:
                logger.error(f"DXGI error during update_frame: {e}")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotError as e:
                logger.error(f"RapidShot error during update_frame: {e}")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            now = time.perf_counter()
            if self._duplicator.updated:
                self._last_frame_time = now
            else:
                # Warn about a screen that has actually been still for a while,
                # and say so in seconds. A run of empty acquires is normal at any
                # timeout and says nothing on its own.
                if self._last_frame_time is None:
                    self._last_frame_time = now
                quiet_for = now - self._last_frame_time
                if (quiet_for >= self._quiet_warning_after_s
                        and now - self._last_quiet_warning >= self._quiet_warning_after_s):
                    logger.warning(
                        f"No screen updates for {quiet_for:.1f}s. Desktop "
                        "Duplication only reports changed content, so a still "
                        "screen produces no frames by design."
                    )
                    self._last_quiet_warning = now

                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()

                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            frame_needs_release = self._duplicator._frame_acquired
            mapped_rect = None
            try:
                memory_region = self.region_to_memory_region(region, self.rotation_angle, self._output)
                region_width = memory_region[2] - memory_region[0]
                region_height = memory_region[3] - memory_region[1]

                if (
                    self._stagesurf.width != region_width
                    or self._stagesurf.height != region_height
                ):
                    self._stagesurf.release()
                    self._stagesurf.rebuild(
                        output=self._output,
                        device=self._device,
                        dim=(region_width, region_height),
                    )

                source_region = D3D11_BOX(
                    left=memory_region[0],
                    top=memory_region[1],
                    right=memory_region[2],
                    bottom=memory_region[3],
                    front=0,
                    back=1,
                )

                self._device.im_context.CopySubresourceRegion(
                    self._stagesurf.texture,
                    0,
                    0,
                    0,
                    0,
                    self._duplicator.texture,
                    0,
                    ctypes.byref(source_region),
                )

                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                    frame_needs_release = False

                accumulator_frame = self._sync_accumulator(memory_region)
                output_wrapper = self._checkout_output_buffer(region_width, region_height)
                mapped_rect = self._stagesurf.map()
                final_array, is_pooled_buffer_still_valid = self._processor.process(
                    mapped_rect,
                    region_width,
                    region_height,
                    (0, 0, region_width, region_height),
                    self.rotation_angle,
                    output_array_for_region,
                    dirty_rects=self._dirty_rects_for(memory_region),
                    output_target=None if output_wrapper is None else output_wrapper.array,
                )
                self._record_accumulator_frame(accumulator_frame)
            finally:
                if mapped_rect is not None:
                    self._stagesurf.unmap()
                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()

            if output_wrapper is not None:
                # The converted frame went into a pooled output buffer. The BGRA
                # staging buffer is finished with either way; the output buffer
                # is the caller's until they release it.
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                if is_pooled_buffer_still_valid and final_array is output_wrapper.array:
                    return output_wrapper
                # The processor declined the target (rotation, or a shape it
                # would not write). Hand the buffer straight back rather than
                # leaking it, and return whatever was actually produced.
                output_wrapper.release()
                return final_array

            if can_use_pool and pooled_buffer_wrapper:
                if is_pooled_buffer_still_valid:
                    if not self._pool_output:
                        # `pool_output=False` promises a plain ndarray the
                        # caller owns and never releases. Every converting mode
                        # gets one for free, because conversion allocates. BGRA
                        # converts nothing, so the frame *is* the staging
                        # buffer -- and this used to hand that buffer over
                        # anyway, still pooled.
                        #
                        # Two failures from one line: the caller got a
                        # PooledBuffer where the documentation says ndarray,
                        # and, having been told no release was needed, never
                        # released it. BGRA has no allocating fallback, so
                        # capture stopped after exactly pool_size_frames frames
                        # and returned None from then on, silently.
                        try:
                            return self._frame_array(pooled_buffer_wrapper).copy()
                        finally:
                            pooled_buffer_wrapper.release()
                    return pooled_buffer_wrapper
                pooled_buffer_wrapper.release()
                return final_array

            return final_array

        except PoolExhaustedError:
            logger.warning("Memory pool exhausted during grab. Consider increasing pool_size_frames.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in _grab: {e}")
            import traceback

            logger.error(traceback.format_exc())
            # Both buffers: the converted-output one is checked out before the
            # processor runs, so a processor that raises would otherwise leak
            # one output buffer per failed frame.
            for wrapper in (locals().get('pooled_buffer_wrapper'),
                            locals().get('output_wrapper')):
                if wrapper and wrapper.state == 'IN_USE':
                    try:
                        wrapper.release()
                    except Exception as rel_e:
                        logger.error(f"Error releasing buffer during exception handling in _grab: {rel_e}")
            self._note_recovery_needed("unhandled error during grab")
            self._last_capture_error_message = f"Unexpected error in _grab: {str(e)}"
            return None

    def _on_output_change(self) -> bool:
        """_on_output_change_locked() under the duplication lock.

        A rebuild replaces the duplicator and staging surface, so it must not
        overlap a grab that is copying out of them.
        """
        with self._dup_lock():
            return self._on_output_change_locked()

    def _on_output_change_locked(self) -> bool:
        """
        Rebuild duplication after a display mode change or access loss.

        This is the exclusive-fullscreen path: when a game takes or releases
        exclusive fullscreen, DXGI invalidates the duplication object and
        refuses to hand out a new one until the mode switch settles. Retrying
        immediately in a tight loop (the previous behaviour) either spins the
        CPU or hangs the caller forever, and the surviving stale stage surface
        is what surfaced as the "black screen in fullscreen" symptom.

        Returns:
            True if duplication was rebuilt, False if it could not be within
            the retry budget (caller should treat capture as degraded).
        """
        time.sleep(0.1)  # Wait for Display mode change (Access Lost)

        if self._duplicator is not None:
            self._duplicator.release()
            self._duplicator = None
        if self._stagesurf is not None:
            # Must be released, not just rebuilt: after a mode switch the old
            # staging texture is still sized for the previous resolution, and
            # StageSurface.rebuild() keeps an existing texture as-is.
            self._stagesurf.release()

        self._output.update_desc()
        self.width, self.height = self._output.resolution
        self.region = self._fit_requested_region()
        self._validate_region(self.region)
        self.rotation_angle = self._output.rotation_angle
        if self.is_capturing:
            self._rebuild_frame_buffer(self.region)

        for attempt in range(self._max_output_change_retries):
            try:
                # Duplicator first: _build_duplicator may settle on a
                # different adapter, and the stage surface must be created on
                # whichever device ended up owning the duplicated texture.
                # The caller's timeout is carried across the rebuild -- dropping
                # it would silently reset the setting on the first resolution
                # change or display reconnect, a regression that only shows up
                # as "it got slower after I unplugged a monitor".
                candidate = self._build_duplicator()
                try:
                    self._stagesurf.rebuild(
                        output=self._output, device=self._device)
                except Exception:
                    # The duplication interface is live as soon as
                    # _build_duplicator returns. Do not publish it until its
                    # matching stage surface also exists, and never carry a
                    # partial pair into the next retry: DXGI permits only one
                    # active duplication interface per output/process.
                    try:
                        candidate.release()
                    except Exception as cleanup_error:
                        logger.warning(
                            "Failed to release partial duplicator after stage "
                            f"surface rebuild failure: {cleanup_error}"
                        )
                    try:
                        self._stagesurf.release()
                    except Exception as cleanup_error:
                        logger.warning(
                            "Failed to release partial stage surface after "
                            f"rebuild failure: {cleanup_error}"
                        )
                    raise
                self._duplicator = candidate
                logger.info(
                    f"Duplication rebuilt after output change "
                    f"(attempt {attempt + 1}, resolution {self.width}x{self.height})."
                )
                return True
            except RapidShotProtectedContentError as e:
                # Retrying cannot help while the protected surface is on screen.
                logger.error(f"Cannot rebuild duplication: {e}")
                self._last_capture_error_message = str(e)
                return False
            except (comtypes.COMError, RapidShotError) as e:
                # DXGI commonly reports UNSUPPORTED/ACCESS_DENIED for a short
                # window while the mode switch is in flight. Back off instead of
                # busy-waiting, and give up rather than hang if it persists.
                wait = min(0.05 * (2 ** attempt), 1.0)
                logger.debug(
                    f"Duplication rebuild attempt {attempt + 1} failed ({e}); "
                    f"retrying in {wait:.2f}s."
                )
                time.sleep(wait)

        self._last_capture_error_message = (
            f"Failed to rebuild duplication after {self._max_output_change_retries} "
            "attempts following an output change."
        )
        logger.error(self._last_capture_error_message)
        self._note_recovery_needed("output change exhausted retries")
        return False

    def start(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 60,
        video_mode: bool = False,
        delay: float = 0,
    ):
        """
        Start capturing frames.

        Args:
            region: Region to capture (left, top, right, bottom)
            target_fps: Target frame rate
            video_mode: Whether to operate in video mode
            delay: Seconds to wait before starting, as in DXcam; fractions
                are fine. Capture resources are rebuilt after the wait, since
                the reason to delay is usually a display change that has to
                settle first.

        Raises:
            ValueError: If delay is negative or not a number.
            RuntimeError: If a previous capture thread that ``stop()`` gave up
                on is still running.
        """
        # This said milliseconds while time.sleep() takes seconds, so
        # delay=500 meant to be half a second waited over eight minutes.
        # Seconds is what the code always did and what DXcam means, and the
        # DXcam shim passes the value straight through, so the docstring was
        # the part that was wrong.
        if isinstance(delay, bool) or not isinstance(delay, (int, float)) or delay < 0:
            raise ValueError(f"delay must be a non-negative number of seconds, got {delay!r}")

        if self.is_capturing:
            logger.debug("start() called while capture is already active; ignoring request.")
            return

        # A thread stop() gave up on is still holding the duplicator and still
        # writing into the pool. Starting a second one would put two threads on
        # one duplication, which is the race _refuse_while_capturing() exists to
        # prevent -- and this path reaches it with is_capturing already False.
        previous = getattr(self, '_capture_thread', None)
        if previous is not None and previous.is_alive():
            raise RuntimeError(
                "The previous capture thread has not exited yet, so a second "
                "one would compete with it for the duplicator. This follows a "
                f"stop() that gave up after {self._stop_join_timeout_s}s; the "
                "camera cannot capture again until that thread ends."
            )

        if delay != 0:
            time.sleep(delay)
            self._on_output_change()
        # Checked again here as well as at construction: it is a public
        # attribute, and 0 used to build a zero-length queue whose first
        # eviction raised IndexError inside the thread, failing capture for
        # good with "deque index out of range".
        _require_positive_int("max_buffer_len", self.max_buffer_len)
        if region is None:
            region = self.region
        else:
            # Explicitly requested, so a later rebuild restores it; see
            # _fit_requested_region.
            self._requested_region = self._normalize_region(region)
            self._region_set_by_user = True
        self._validate_region(region)
        self.is_capturing = True
        
        # Phase 4: Initialize deque for continuous mode
        self._pooled_frames_deque = collections.deque(maxlen=self._queue_limit())
        self._last_dup_source = None
        self._frame_available_event.clear() # Clear before starting
        self._stop_capture_event.clear()

        # region is already validated and is self.region if None was passed
        # The capture thread will use self.region by default if grab is called with None
        
        self._capture_thread = Thread( # Renamed from self.__thread
            target=self._capture_thread_func, # Renamed from __capture
            name="ScreenCaptureThread", # More descriptive name
            args=(region, target_fps, video_mode),
        )
        self._capture_thread.daemon = True
        self._capture_thread.start()

    def _queue_limit(self) -> int:
        """How many frames the continuous-mode queue may hold.

        Normally ``max_buffer_len``. But when the queue holds buffers from the
        *staging* pool -- which is BGRA, the one mode that does no conversion
        and so hands its staging buffer straight to the caller -- the queue
        cannot be allowed to hold more than the pool can spare, or the producer
        runs out of buffers to check out and capture stops dead.

        That is what used to happen: the queue was bounded at 64 while the pool
        held 4, nothing was returned until the queue reached 64, and it never
        could. Continuous BGRA capture produced exactly `pool_size_frames`
        frames and then froze, logging a pool-exhaustion warning per attempt.
        Every other colour mode was unaffected, because its queued frames come
        from the output pool, which falls back to allocating when it runs dry.

        One buffer is always left free for the next grab, which is why this is
        ``pool_size - 1``.
        """
        limit = self.max_buffer_len
        processor = getattr(self, "_processor", None)
        if processor is None or getattr(processor, "converts_output", True):
            # No processor means a half-built camera in a test; a converting
            # one queues output-pool buffers, which fall back to allocating.
            return limit

        pool = getattr(self, "memory_pool", None)
        pool_size = (pool.num_buffers if pool is not None
                     else self._init_args.get("pool_size_frames", 2))
        if pool_size < 2:
            logger.warning(
                f"pool_size_frames={pool_size} leaves no buffer free while a "
                "frame is queued, so continuous BGRA capture cannot keep "
                "running. Use pool_size_frames=2 or more.")
            return 1

        spare = pool_size - 1
        if spare < limit:
            logger.debug(
                f"Continuous-mode queue limited to {spare} frames by "
                f"pool_size_frames={pool_size}, not max_buffer_len={limit}: "
                "BGRA queues staging-pool buffers, and one must stay free for "
                "the next capture.")
        return min(limit, spare)

    def stop(self):
        """
        Stop capturing frames.

        Returns:
            True once the capture thread has finished. False if it was still
            running when the wait ran out, or if ``stop()`` was called from the
            capture thread itself. In both of those cases the thread is still
            on its way out and owns its own cleanup, so this leaves the timer
            handle and the frame queue alone.
        """
        thread_finished = True

        if getattr(self, 'is_capturing', False):
            self._stop_capture_event.set() # Use renamed event
            thread = getattr(self, '_capture_thread', None)
            if thread is not None:
                if current_thread() is thread:
                    # stop() from inside the capture thread. It is about to
                    # unwind into the cleanup at the end of
                    # _capture_thread_func, so its resources are not ours.
                    thread_finished = False
                else:
                    thread.join(timeout=self._stop_join_timeout_s)
                    if thread.is_alive():
                        thread_finished = False
                        logger.error(
                            "Capture thread did not stop within "
                            f"{self._stop_join_timeout_s}s and is still running. "
                            "Its timer handle and queued frames are left to it; "
                            "start() will refuse until it exits."
                        )
                    else:
                        self._capture_thread = None

        self.is_capturing = False
        self._frame_count = 0
        self._frame_available_event.clear()
        # self._stop_capture_event is already set, clear if restartable, but usually not needed

        if not thread_finished:
            # Everything below belongs to a thread that is still running.
            #
            # The timer handle is created and closed inside the capture thread;
            # closing it here as well is a second CloseHandle on a handle
            # Windows may already have reissued to something unrelated -- and
            # this used to happen on both the abandoned path and the
            # stop-from-the-capture-thread path.
            #
            # Emptying the queue is no safer: the thread appends to it, and
            # setting it to None underneath faults the thread into the
            # catch-all that marks the camera permanently failed.
            return False

        if self._timer_handle:
            try:
                cancel_timer(self._timer_handle)
            except Exception as timer_error:
                logger.warning(f"Failed to cancel timer during stop(): {timer_error}")
            finally:
                try:
                    close_timer(self._timer_handle)
                except Exception as close_error:
                    logger.warning(f"Failed to close timer handle during stop(): {close_error}")
                self._timer_handle = None

        # Phase 4/5: Release any remaining buffers in the deque
        self._drain_frame_queue()
        self._pooled_frames_deque = None
        return True

    def get_latest_frame(self, as_numpy: bool = True):
        """
        Get the latest captured frame, as an array that is the caller's to keep.

        Args:
            as_numpy: If True, always return a NumPy array even when using GPU
                acceleration. If False and using GPU acceleration, return a
                CuPy array for better performance.

        Returns:
            The latest captured frame, or None if none arrived within a second.

        The frame is copied out of the capture queue. That queue holds pooled
        buffers, and the producer hands an evicted one straight back to the pool
        for the next capture to write into -- so returning the pooled array
        itself, which this used to do, meant the caller's frame could change
        under it at any moment with nothing raising. ``stop()`` did the same, by
        returning every queued buffer to the pool.

        Use :meth:`get_latest_frame_buffer` to skip the copy; it hands over the
        buffer itself, and with it the duty to release it.
        """
        frame = self.get_latest_frame_buffer()
        if frame is None:
            return None

        try:
            frame_array = self._frame_array(frame)

            # Guarded by the flag first: probing CuPy on a CPU camera would
            # import it, which is the cost this module defers.
            cp = _require_cupy() if self.nvidia_gpu else None
            if cp is not None and isinstance(frame_array, cp.ndarray):
                # asnumpy() already copies to the host; only the stay-on-device
                # path still needs one.
                return cp.asnumpy(frame_array) if as_numpy else frame_array.copy()

            if isinstance(frame_array, np.ndarray):
                return frame_array.copy()

            logger.error(f"Unexpected array type in deque: {type(frame_array)}")
            return None
        finally:
            self._discard_frame(frame)

    def get_latest_frame_buffer(self):
        """
        The latest captured frame, handed over without a copy.

        Ownership transfers to the caller. The frame leaves the capture queue,
        so the producer will not recycle it, and the caller must ``release()``
        it -- until then that buffer is unavailable to capture. Colour modes
        other than BGRA may yield a plain array with no ``release()``, so use
        ``getattr(frame, "release", None)`` rather than assuming one.

        Returns:
            A :class:`~rapidshot.memory_pool.PooledBuffer` or a plain array, or
            None if no frame arrived within a second.

        Prefer :meth:`get_latest_frame` unless the copy shows up in a profile.
        """
        if not self._frame_available_event.wait(timeout=1.0): # Wait for a short duration
            logger.debug("get_latest_frame timed out waiting for frame_available_event.")
            return None # No frame available or timeout

        with self._capture_lock: # Protect access to deque
            if not self._pooled_frames_deque:
                self._frame_available_event.clear() # Clear if deque is empty after wait
                return None

            # Take the newest frame out of the queue rather than peeking at it.
            # While it sits there the producer may evict and release it, and a
            # caller reading the array it wrapped would then be looking at a
            # buffer the next capture is writing into.
            frame = self._pooled_frames_deque.pop()

            # video_mode duplicates the newest frame when the screen is idle.
            # That frame is now the caller's and may be released at any moment,
            # so the producer must not copy out of it; it picks up a fresh
            # source on the next real frame.
            if self._last_dup_source is frame:
                self._last_dup_source = None

            if not self._pooled_frames_deque:
                # Nothing left to hand out: the next call waits for a new frame,
                # which is what "blocks until a new frame arrives" promises.
                self._frame_available_event.clear()

        return frame

    def _capture_thread_func( # Renamed from __capture
        self, region: Tuple[int, int, int, int], target_fps: int = 60, video_mode: bool = False
    ):
        """
        Internal capture thread implementation for continuous mode.
        
        Args:
            region: Region to capture (left, top, right, bottom). This is the default region.
            target_fps: Target frame rate.
            video_mode: Whether to operate in video mode (duplicates last frame if no new one).
        """
        if target_fps > 0: # Allow target_fps = 0 for max speed
            period_ms = 1000 // target_fps
            self._timer_handle = create_high_resolution_timer()
            set_periodic_timer(self._timer_handle, period_ms)
        else: # Running at max speed
            self._timer_handle = None


        self._capture_start_time = time.perf_counter()
        capture_error = None

        while not self._stop_capture_event.is_set():
            if self._timer_handle:
                res = wait_for_timer(self._timer_handle, INFINITE)
                if res == WAIT_FAILED: # Timer error
                    self._stop_capture_event.set()
                    capture_error = ctypes.WinError()
                    logger.error(f"High-resolution timer wait failed: {capture_error}")
                    continue
            
            grab_result = None
            try:
                if self._capture_permanently_failed: # Check before each grab attempt in loop
                    logger.error(f"Capture permanently failed. Stopping capture thread. Last error: {self._last_capture_error_message}")
                    self._stop_capture_event.set() # Signal thread to stop
                    break # Exit while loop

                # Use self.region for continuous capture, which was set during start()
                # _grab will handle _needs_reinit flag internally.
                grab_result = self._grab(self.region) 

                if grab_result is not None:
                    self._frame_count += 1
                    # grab_result is a PooledBuffer only when the processor could
                    # write the result in place, i.e. BGRA output. Every other
                    # color mode changes the channel count and yields a freshly
                    # allocated array -- which is equally valid to queue, and
                    # rejecting it (as this used to) left the deque permanently
                    # empty for RGB/BGR/RGBA/GRAY consumers.
                    evicted_buffer = None
                    with self._capture_lock:
                        if len(self._pooled_frames_deque) == self._pooled_frames_deque.maxlen:
                            evicted_buffer = self._pooled_frames_deque[0]
                        self._pooled_frames_deque.append(grab_result)
                        self._last_dup_source = grab_result
                    # Check the evicted buffer back in outside the capture
                    # lock: release() takes the pool's own lock, and holding
                    # both here stalls every get_latest_frame() consumer for
                    # the duration of the pool round-trip.
                    self._discard_frame(evicted_buffer)
                    self._frame_available_event.set()
                
                elif self._needs_reinit: # _grab returned None and might have set _needs_reinit
                    logger.info("Continuous mode: Grab failed, re-initialization pending or in progress.")
                    # Optional: Short sleep before next attempt if re-init is happening via _grab
                    time.sleep(0.1) # Avoid tight loop if _grab keeps failing due to re-init
                    continue # Try again, _grab will attempt re-init

                elif video_mode:
                    # No new content this tick: re-queue a copy of the last frame
                    # so the output stream keeps a constant frame rate.
                    with self._capture_lock:
                        dup_source = self._last_dup_source
                    if dup_source is None:
                        # Nothing captured yet, or the last frame was handed to
                        # a caller by get_latest_frame_buffer(). Either way
                        # there is nothing safe to copy from until the next one.
                        continue
                    duplicate_frame = None
                    try:
                        source_array = self._frame_array(dup_source)
                        if isinstance(dup_source, PooledBuffer):
                            if self.memory_pool is None:
                                logger.warning("Video_mode: Memory pool not available for duplicating frame.")
                                raise PoolExhaustedError("no pool")
                            duplicate_frame = self.memory_pool.checkout()
                            if self.nvidia_gpu: # cp array
                                duplicate_frame.array[:] = source_array
                            else: # np array
                                np.copyto(duplicate_frame.array, source_array)
                        else:
                            # Plain array (non-BGRA output): copy directly, the
                            # pool's BGRA buffers are the wrong shape for it.
                            duplicate_frame = source_array.copy()
                        if self.nvidia_gpu:
                            # A device copy is only queued by the assignment
                            # above. The check below is about when the copy
                            # *read* the source, so it has to have run first.
                            self._wait_for_device_copy()

                        evicted_buffer = None
                        taken_during_copy = False
                        with self._capture_lock:
                            # The copy ran outside the lock, and in that window
                            # get_latest_frame_buffer() may have handed the
                            # source to a consumer, whose buffer it then is --
                            # theirs to draw on or release. Anything copied out
                            # of it after that is not a frame this producer
                            # owned. Handing it out clears _last_dup_source, so
                            # checked here, in the same lock that publishes the
                            # duplicate, it says exactly whether that happened.
                            if self._last_dup_source is not dup_source:
                                taken_during_copy = True
                            else:
                                if len(self._pooled_frames_deque) == self._pooled_frames_deque.maxlen:
                                    evicted_buffer = self._pooled_frames_deque[0]
                                self._pooled_frames_deque.append(duplicate_frame)
                                self._last_dup_source = duplicate_frame
                        if taken_during_copy:
                            # The consumer has the frame; the next real one
                            # starts duplication again.
                            self._discard_frame(duplicate_frame)
                            continue
                        self._discard_frame(evicted_buffer)  # Outside the lock, see above
                        self._frame_available_event.set()
                        self._frame_count += 1
                    except PoolExhaustedError:
                        logger.warning("Video_mode: Pool exhausted, cannot duplicate frame.")
                        self._discard_frame(duplicate_frame)
                    except Exception as dup_e:
                        logger.error(f"Video_mode: Error duplicating frame: {dup_e}")
                        self._discard_frame(duplicate_frame)
            
            # _grab_locked turns every exception into a None return and a
            # scheduled rebuild, so what reaches here is a fault in this loop
            # itself. The RapidShotReinitError/DeviceError clauses that sat
            # above this could not be reached.
            except Exception as e: 
                import traceback
                logger.error(f"Error in capture thread: {e}\n{traceback.format_exc()}")
                self._last_capture_error_message = f"Runtime error in capture thread: {str(e)}"
                self._capture_permanently_failed = True # Assume critical error
                self._stop_capture_event.set() 
                capture_error = e
                
        # Clean up timer
        if self._timer_handle:
            try:
                cancel_timer(self._timer_handle)
            except Exception as timer_error:
                logger.warning(f"Failed to cancel capture timer: {timer_error}")
            finally:
                try:
                    close_timer(self._timer_handle)
                except Exception as close_error:
                    logger.warning(f"Failed to close capture timer handle: {close_error}")
                self._timer_handle = None
        
        if capture_error is not None or self._capture_permanently_failed:
            logger.error(f"Capture thread terminated. Error: {capture_error}. Permanent failure: {self._capture_permanently_failed}. Last message: {self._last_capture_error_message}")
            
        capture_duration = time.perf_counter() - self._capture_start_time
        if capture_duration > 0 and self._frame_count > 0: 
            actual_fps = self._frame_count / capture_duration
            logger.info(f"ScreenCapture continuous mode stopped. Captured {self._frame_count} frames in {capture_duration:.2f}s (FPS: {actual_fps:.2f}).")
        else:
            logger.info(f"ScreenCapture continuous mode stopped. No frames captured or capture time was zero.")

    @staticmethod
    def _wait_for_device_copy() -> None:
        """Block until work queued on the current CUDA stream has run."""
        cp = _require_cupy()
        get_stream = getattr(getattr(cp, "cuda", None), "get_current_stream", None)
        if get_stream is not None:
            get_stream().synchronize()

    def _drain_frame_queue(self) -> None:
        """Return every queued frame to its pool and empty the queue.

        Call this before destroying the pool those buffers came from: a
        wrapper outliving its pool releases into a dead one, which refuses the
        check-in, so the buffer is dropped rather than recycled.

        Safe to call when there is no queue -- outside continuous mode there is
        nothing to drain.
        """
        queue = getattr(self, "_pooled_frames_deque", None)
        if queue is None:
            self._last_dup_source = None
            return

        with self._capture_lock:
            # Copy then clear under the lock; the pool check-ins happen outside
            # it so a slow pool cannot block the producer.
            stale_frames = list(queue)
            queue.clear()
            self._last_dup_source = None
        for frame in stale_frames:
            self._discard_frame(frame)
        self._frame_available_event.clear()

    @staticmethod
    def _discard_frame(frame) -> None:
        """
        Drop a queued frame, returning it to the pool if it came from one.

        The continuous-mode deque holds PooledBuffer wrappers for BGRA output
        and plain arrays for every other color mode, so callers must not assume
        a ``release()`` method exists.
        """
        if frame is None:
            return
        release = getattr(frame, "release", None)
        if release is None:
            return  # Plain array: ordinary garbage collection owns it
        try:
            release()
        except Exception as e:
            logger.debug(f"Ignoring error releasing pooled frame: {e}")

    @staticmethod
    def _frame_array(frame):
        """Return the underlying array for a queued frame (pooled or plain)."""
        return getattr(frame, "array", frame)

    def _rebuild_frame_buffer(self, region: Tuple[int, int, int, int]):
        """
        Rebuild the continuous-mode frame buffer after a resolution change.

        Drops every buffer still queued (they are sized for the old resolution)
        and rebuilds the memory pool to the new region shape.

        Args:
            region: Region to capture (left, top, right, bottom)
        """
        if region is None:
            region = self.region

        frame_shape = self._staging_shape(region)  # BGRA, panel orientation

        # Return queued buffers to the pool before it is torn down, otherwise
        # the wrappers outlive their pool and their release() targets a dead one.
        self._drain_frame_queue()

        # 2, matching the constructor. This read 10 -- the default two
        # releases ago -- so a rebuilt pool was five times the size of the one
        # it replaced, for any camera that did not pass the argument.
        pool_size = self._staging_pool_size(
            self._init_args.get("pool_size_frames", 2))
        if self.memory_pool is not None:
            if tuple(self.memory_pool.buffer_shape) == frame_shape:
                return  # Shape unchanged, existing pool is still correct
            self.memory_pool.destroy_pool()
            self.memory_pool = None

        logger.debug(f"Rebuilding memory pool for new frame shape {frame_shape}.")
        if self.nvidia_gpu and cupy_available():
            self.memory_pool = CupyMemoryPool(frame_shape, np.uint8, pool_size)
        else:
            self.memory_pool = NumpyMemoryPool(frame_shape, np.uint8, pool_size)

    def _normalize_region(self, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Validate *region* without mutating capture state."""
        if region is None:
            raise ValueError("Region cannot be None")

        if not hasattr(self, 'width') or not hasattr(self, 'height'):
            raise ValueError("Capture dimensions are not initialized")

        try:
            l, t, r, b = map(int, region)
        except (TypeError, ValueError) as conversion_error:
            raise ValueError(f"Region must be a tuple of four integers: {conversion_error}") from conversion_error

        if l < 0 or t < 0:
            raise ValueError(f"Region start ({l}, {t}) is outside the capture bounds (0, 0)")

        if r > self.width or b > self.height:
            raise ValueError(
                f"Region end ({r}, {b}) exceeds capture bounds ({self.width}, {self.height})"
            )

        if l >= r or t >= b:
            raise ValueError(f"Region coordinates must form a positive area, got {region}")

        return (l, t, r, b)

    def _validate_region(self, region: Tuple[int, int, int, int]):
        """
        Validate region coordinates.

        Args:
            region: Region to validate (left, top, right, bottom)

        Raises:
            ValueError: If region is invalid
        """
        validated_region = self._normalize_region(region)
        l, t, r, b = validated_region
        self.region = validated_region

        if hasattr(self, '_sourceRegion') and self._sourceRegion is not None:
            self._sourceRegion.left = l
            self._sourceRegion.top = t
            self._sourceRegion.right = r
            self._sourceRegion.bottom = b

        self.shot_w, self.shot_h = r - l, b - t

    @property
    def released(self) -> bool:
        """True once :meth:`release` has run. A released camera never captures
        again; :func:`rapidshot.create` builds a new one rather than return it."""
        return getattr(self, "_released", False)

    def release(self):
        """
        Release all resources.
        """
        # Set first, so a teardown that raises part-way still marks the camera
        # as unusable rather than leaving the factory to hand it out again.
        self._released = True
        locked = False
        try:
            if hasattr(self, 'is_capturing') and self.is_capturing: # Check is_capturing before calling stop
                if not self.stop():
                    # Say so once, here: the teardown below frees the
                    # duplicator, stage surface and pool that the surviving
                    # thread is still using. Releasing anyway remains the right
                    # call -- the alternative is a release() that never returns
                    # -- but it is not a clean one.
                    logger.warning(
                        "release(): the capture thread is still running. "
                        "Tearing down resources it is still using.")

            # Tear down only once no grab is mid-copy on another thread. Bounded,
            # because stop() gives up on a capture thread after 10 s and that
            # thread may still hold the lock; waiting forever would turn a hung
            # thread into a hung release().
            locked = self._dup_lock().acquire(timeout=5)
            if not locked:
                logger.warning("release(): a capture call still holds the duplicator "
                               "after 5 s; releasing resources anyway.")

            # A Frame still holding the desktop texture would keep DXGI's
            # surface pinned past the duplicator's own teardown.
            live = getattr(self, '_live_frame', None)
            if live is not None and not live.released:
                live.release()
            self._live_frame = None

            if hasattr(self, '_duplicator') and self._duplicator:
                self._duplicator.release()
                
            if hasattr(self, '_stagesurf') and self._stagesurf:
                self._stagesurf.release()

            # Phase 5: Destroy memory pool
            if hasattr(self, 'memory_pool') and self.memory_pool:
                logger.info("Destroying memory pool.")
                self.memory_pool.destroy_pool()
                self.memory_pool = None
            # Held for as long as the camera is; on the CuPy path it is device
            # memory, so dropping it here rather than at collection matters.
            self._scratch_staging = None

        except Exception as e:
            logger.warning(f"Error during release: {e}")
        finally:
            if locked:
                self._dup_lock().release()

    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        try:
            self.release()
        except Exception as e:
            logger.warning(f"Error during destruction: {e}")

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation of the ScreenCapture instance
        """
        try:
            return "<{}:\n\t{},\n\t{},\n\t{},\n\t{}\n>".format(
                "ScreenCapture",
                self._device if hasattr(self, '_device') else "No device",
                self._output if hasattr(self, '_output') else "No output",
                self._stagesurf if hasattr(self, '_stagesurf') else "No stage surface",
                self._duplicator if hasattr(self, '_duplicator') else "No duplicator",
            )
        except Exception:
            return "<ScreenCapture: initialization incomplete>"


def __getattr__(name):
    """Keep `capture.CUPY_AVAILABLE` and `capture.cp` working.

    Both were module-level names until CuPy's import was deferred. Reading
    either still answers correctly; it just pays for the import at that
    point rather than at `import rapidshot`.
    """
    if name == "CUPY_AVAILABLE":
        return cupy_available()
    if name == "cp":
        return _require_cupy()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
