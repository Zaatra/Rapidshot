import collections
import logging
import threading
import numpy as np

logger = logging.getLogger(__name__)


# Custom Exception
class PoolExhaustedError(RuntimeError):
    """Raised when no buffers are available in the pool."""
    pass

class BufferReleasedError(RuntimeError):
    """Raised when a buffer is used after being returned to its pool."""
    pass


class PooledBuffer:
    """
    A wrapper around a NumPy or CuPy array managed by a memory pool.

    Behaves like the array it wraps for the operations frames are actually put
    through — indexing, slicing, ``np.asarray``, ``len``, ``shape``, ``dtype``
    — so a caller can treat it as the frame. This matters because ``grab()``
    returns one of these: without it, every consumer would need to know about
    pooling just to read a pixel.

    What it deliberately does *not* do is pretend to be an ``ndarray`` subclass.
    The buffer goes back to the pool on :meth:`release` and is then handed to
    the next capture, so anything still holding it would silently see the wrong
    frame. Use after release raises instead — the same bargain
    :class:`rapidshot.frame.Frame` makes for GPU textures.

    ``np.asarray(buffer)`` is a **view**, not a copy: zero-cost, and invalid
    once released. Call ``.copy()`` to outlive the release.
    """
    def __init__(self, array, pool_ref):
        self.array = array
        self.state = 'AVAILABLE'  # Initial state
        self._pool = pool_ref

    def release(self):
        """Releases the buffer back to its pool.

        A buffer whose pool was destroyed while this was held -- a capture
        rebuild after a resolution change or device loss does that -- has no
        pool to go back to. Releasing it just ends the caller's use of it; this
        used to raise RuntimeError from the dead pool.
        """
        if self.state == 'DETACHED':
            self.state = 'RELEASED'
            return
        if self.state in ('AVAILABLE', 'RELEASED'):
            # Already released: a second call ends nothing and returns nothing.
            # It used to raise from checkin, or from the dead pool. This cannot
            # catch a release through a stale reference after the buffer was
            # checked out again -- the wrapper is the same object -- so that
            # stays the caller's to avoid.
            return
        self._pool.checkin(self)

    def _live(self):
        """The wrapped array, or an error naming the mistake."""
        if self.state not in ('IN_USE', 'DETACHED'):
            raise BufferReleasedError(
                "This frame's buffer has been returned to the pool and may "
                "already hold a different frame. Copy the data before calling "
                "release() if it needs to outlive the frame."
            )
        return self.array

    def __repr__(self):
        # NumPy exposes the address through .ctypes, CuPy through .data.ptr;
        # reading .ctypes alone made repr() raise for every GPU buffer.
        address = getattr(getattr(self.array, "ctypes", None), "data", None)
        if address is None:
            address = getattr(getattr(self.array, "data", None), "ptr", None)
        where = f" data_ptr=0x{address:X}" if isinstance(address, int) else ""
        return f"<PooledBuffer state='{self.state}'{where} pool='{self._pool.__class__.__name__}'>"

    # -- array-like surface -------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """Hand the underlying array to NumPy, OpenCV, PIL and friends.

        A view, so `np.asarray(frame)` costs nothing — which is the entire
        point of pooling — and dies with the release.
        """
        array = self._live()
        if dtype is not None and array.dtype != dtype:
            return array.astype(dtype)          # astype already copies
        if copy:
            # NumPy 2 passes `copy` through and trusts the answer: accepting the
            # argument and ignoring it makes `np.array(frame, copy=True)` return
            # a *view* of a pooled buffer. The caller then holds what looks like
            # its own array, the next capture overwrites it, and the data is
            # wrong with nothing raising -- which is exactly what pooling is
            # documented to protect against.
            return array.copy()
        return array

    def __getitem__(self, key):
        return self._live()[key]

    def __setitem__(self, key, value):
        self._live()[key] = value

    def __len__(self):
        return len(self._live())

    def copy(self):
        """An independent array that survives release()."""
        return self._live().copy()

    # For convenience, allow direct access to the array's shape and dtype
    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def size(self):
        return self.array.size

    @property
    def nbytes(self):
        return self.array.nbytes

class BaseMemoryPool:
    """
    Abstract-like base class for memory pools.
    """
    def __init__(self, buffer_shape, dtype, num_buffers):
        self.buffer_shape = buffer_shape
        self.dtype = dtype
        if num_buffers <= 0:
            raise ValueError("Number of buffers must be positive.")
        self.num_buffers = num_buffers
        
        self._buffers = []  # List of all PooledBuffer objects
        self._available_buffers = collections.deque()
        self._lock = threading.Lock()
        self._initialized = False

    def _create_buffer(self):
        """
        Abstract method to be implemented by subclasses.
        Should return a single allocated array (e.g., np.empty, cp.empty).
        """
        raise NotImplementedError("Subclasses must implement _create_buffer.")

    def initialize_pool(self):
        """
        Allocates and initializes all buffers in the pool.
        This method should be called by the subclass's __init__ after super().__init__.
        """
        if self._initialized:
            # Or raise an error, or allow re-initialization with cleanup
            logger.debug("Pool is already initialized.")
            return

        with self._lock: # Ensure thread safety during initialization
            if self._initialized: # Double check after acquiring lock
                return
            
            for _ in range(self.num_buffers):
                try:
                    actual_array = self._create_buffer()
                    buffer_wrapper = PooledBuffer(array=actual_array, pool_ref=self)
                    self._buffers.append(buffer_wrapper)
                    self._available_buffers.append(buffer_wrapper)
                except Exception as e:
                    # Handle partial initialization failure?
                    # For now, let it propagate, or log and stop.
                    logger.error(f"Error creating a buffer during pool initialization: {e}")
                    # Potentially clean up already created buffers if needed.
                    self._buffers.clear()
                    self._available_buffers.clear()
                    raise # Re-raise the exception
            
            self._initialized = True

    def checkout(self, timeout=None): # timeout is not used yet
        """
        Checks out a buffer from the pool.
        
        Args:
            timeout: Not currently implemented.
            
        Returns:
            A PooledBuffer object.
            
        Raises:
            PoolExhaustedError: If no buffers are available.
            RuntimeError: If the pool is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Memory pool is not initialized. Call initialize_pool() first.")

        with self._lock:
            if self._available_buffers:
                buffer_wrapper = self._available_buffers.popleft()
                buffer_wrapper.state = 'IN_USE'
                return buffer_wrapper
            else:
                # Lock is released automatically when exiting 'with' block
                raise PoolExhaustedError("No buffers available in the pool.")

    def checkin(self, buffer_wrapper: PooledBuffer):
        """
        Returns a buffer to the pool.
        
        Args:
            buffer_wrapper: The PooledBuffer object to return.
            
        Raises:
            ValueError: If the buffer does not belong to this pool or is not in use.
            RuntimeError: If the pool is not initialized.
        """
        if not self._initialized:
            # This case might be less critical if buffers are only checked in post-init
            raise RuntimeError("Memory pool is not initialized.")

        with self._lock:
            if buffer_wrapper._pool is not self:
                raise ValueError("Buffer does not belong to this pool.")
            if buffer_wrapper.state != 'IN_USE':
                # Or handle idempotently if already available
                raise ValueError(f"Buffer is not 'IN_USE', current state: {buffer_wrapper.state}.")
            if not any(b is buffer_wrapper for b in self._buffers): # Check if it's one of our managed buffers
                 raise ValueError("Buffer was not created by this pool (identity check failed).")


            buffer_wrapper.state = 'AVAILABLE'
            self._available_buffers.append(buffer_wrapper)

    def get_stats(self):
        """
        Returns statistics about the pool's buffer usage.
        """
        if not self._initialized:
            return {'total': self.num_buffers, 'available': 0, 'in_use': 0, 'initialized': False}
        
        with self._lock: # Ensure consistent read of available_buffers length
            available_count = len(self._available_buffers)
        
        # Calculate in_use based on total and available, as buffers list might not reflect current state directly
        # without iterating and checking state, which is less efficient.
        # The number of buffers in _buffers that are not in _available_buffers.
        # A more accurate in_use could be calculated by:
        # in_use_count = sum(1 for buf in self._buffers if buf.state == 'IN_USE')
        # However, using num_buffers - available_count is simpler if checkout/checkin logic is sound.
        in_use_count = self.num_buffers - available_count
        
        return {
            'total': self.num_buffers,
            'available': available_count,
            'in_use': in_use_count,
            'initialized': self._initialized
        }

    def release_all_buffers(self):
        """Make every buffer available again, without taking one from a caller.

        A buffer still checked out is *detached* -- as :meth:`destroy_pool`
        detaches it -- and replaced with a freshly allocated one. Its holder
        keeps the array, which stays readable, and a ``release()`` that ends
        their use of it without returning anything to the pool.

        This used to mark every buffer AVAILABLE, including ones a caller still
        held, so the next ``checkout()`` handed the same memory to a second
        owner: the next capture wrote into a frame the first owner was still
        reading, with nothing raising on either side.

        Atomic: the replacements are allocated before anything changes, so a
        failed allocation leaves the pool exactly as it was.
        """
        if not self._initialized:
            # Cannot release buffers if pool wasn't even initialized with them
            logger.warning("Pool not initialized, cannot release buffers.")
            return

        with self._lock:
            held = [b for b in self._buffers if b.state == 'IN_USE']
            replacements = [PooledBuffer(array=self._create_buffer(), pool_ref=self)
                            for _ in held]
            for buffer_wrapper in held:
                buffer_wrapper.state = 'DETACHED'
            self._buffers = [b for b in self._buffers if b.state != 'DETACHED'] + replacements
            for buffer_wrapper in self._buffers:
                buffer_wrapper.state = 'AVAILABLE'
            self._available_buffers = collections.deque(self._buffers)

    def destroy_pool(self):
        """
        Clears buffer lists and marks the pool as uninitialized.
        Actual memory deallocation depends on the subclass and Python's GC.
        For CuPy, its internal memory pool handles GPU memory.
        """
        with self._lock:
            # A buffer a caller still holds is detached, not destroyed: it keeps
            # its array, stays readable, and its release() becomes a no-op. This
            # used to `del buf.array` on every buffer, so a frame from grab()
            # held across a capture rebuild raised AttributeError when read and
            # RuntimeError when released. Idle buffers are only referenced from
            # the pool, so clearing the lists below is what frees them.
            for buf in self._buffers:
                buf.state = 'DETACHED' if buf.state == 'IN_USE' else 'DESTROYED'

            self._buffers.clear()
            self._available_buffers.clear()
            self._initialized = False
            # print(f"Pool {self.__class__.__name__} destroyed. Buffers cleared.")


class NumpyMemoryPool(BaseMemoryPool):
    """
    A memory pool for NumPy arrays.
    """
    def __init__(self, buffer_shape, dtype, num_buffers):
        super().__init__(buffer_shape, dtype, num_buffers)
        self.initialize_pool() # Automatically initialize upon creation

    def _create_buffer(self):
        # numpy is already imported as np at the module level
        return np.empty(self.buffer_shape, dtype=self.dtype)


class CupyMemoryPool(BaseMemoryPool):
    """
    A memory pool for CuPy arrays.
    Requires CuPy to be installed.
    """
    def __init__(self, buffer_shape, dtype, num_buffers):
        super().__init__(buffer_shape, dtype, num_buffers)
        # Ensure CuPy is available before trying to initialize the pool with it
        try:
            import cupy # Check if cupy can be imported
        except ImportError:
            raise ImportError("CupyMemoryPool requires CuPy to be installed.")
        
        self.initialize_pool() # Automatically initialize upon creation

    def _create_buffer(self):
        import cupy as cp # Import locally to ensure it's available here
        return cp.empty(self.buffer_shape, dtype=self.dtype)

    # destroy_pool is the base class's. CuPy returns device memory to its own
    # pool once the last reference to an array goes, which clearing the pool's
    # lists does for idle buffers; a held buffer keeps its memory until the
    # caller lets go, exactly as on the NumPy path.
