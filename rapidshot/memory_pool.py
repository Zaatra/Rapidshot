import collections
import threading
import numpy as np

# Custom Exception
class PoolExhaustedError(RuntimeError):
    """Raised when no buffers are available in the pool."""
    pass

class PooledBuffer:
    """
    A wrapper around a NumPy or CuPy array managed by a memory pool.
    """
    def __init__(self, array, pool_ref):
        self.array = array
        self.state = 'AVAILABLE'  # Initial state
        self._pool = pool_ref

    def release(self):
        """Releases the buffer back to its pool."""
        self._pool.checkin(self)

    def __repr__(self):
        return f"<PooledBuffer state='{self.state}' data_ptr=0x{self.array.ctypes.data:X} pool='{self._pool.__class__.__name__}'>"

    # For convenience, allow direct access to the array's shape and dtype
    @property
    def shape(self):
        return self.array.shape

    @property
    def dtype(self):
        return self.array.dtype

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
            print("Pool is already initialized.") 
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
                    print(f"Error creating a buffer during pool initialization: {e}")
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
        """
        Marks all buffers as available, effectively resetting the pool's available queue.
        This is primarily for resetting state, not for deallocation.
        """
        if not self._initialized:
            # Cannot release buffers if pool wasn't even initialized with them
            print("Pool not initialized, cannot release buffers.")
            return

        with self._lock:
            self._available_buffers.clear()
            for buffer_wrapper in self._buffers:
                buffer_wrapper.state = 'AVAILABLE'
                self._available_buffers.append(buffer_wrapper)
            
            # Sanity check
            if len(self._available_buffers) != self.num_buffers:
                # This might indicate an issue if some buffers were lost or duplicated
                print(f"Warning: After release_all_buffers, available count ({len(self._available_buffers)}) "
                      f"does not match total buffers ({self.num_buffers}).")


    def destroy_pool(self):
        """
        Clears buffer lists and marks the pool as uninitialized.
        Actual memory deallocation depends on the subclass and Python's GC.
        For CuPy, its internal memory pool handles GPU memory.
        """
        with self._lock:
            # For Numpy arrays, Python's GC will handle memory when references are cleared.
            # For CuPy arrays, CuPy's memory pool will manage the GPU memory.
            # Explicitly deleting arrays might be needed if they hold external resources
            # not managed by Python's GC or CuPy's pool (e.g., registered interop resources).
            # For simple np.empty/cp.empty, clearing lists should be sufficient.
            
            for buf in self._buffers:
                # If buffers need explicit cleanup beyond GC, do it here.
                # e.g., if PooledBuffer held a custom resource.
                del buf.array # Remove reference to the array; let GC handle it
            
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

    def destroy_pool(self):
        """
        Destroys the pool, clearing CuPy buffers.
        CuPy's memory pool should handle deallocation, but explicit del can help.
        """
        with self._lock:
            if not self._initialized:
                return

            # It's good practice to ensure CuPy arrays are explicitly deleted
            # if there's any doubt about GC behavior with GPU resources,
            # though CuPy's memory pool is generally effective.
            for buf_wrapper in self._buffers:
                # This breaks the PooledBuffer's reference to the array.
                # CuPy's memory pool will reclaim the GPU memory when its
                # internal reference count for that block drops to zero.
                del buf_wrapper.array 
            
            self._buffers.clear()
            self._available_buffers.clear()
            self._initialized = False
            # print(f"CupyMemoryPool destroyed. Buffers cleared.")
            # Optionally, can try to clear CuPy's memory pool if aggressive cleanup is needed,
            # but this affects all CuPy allocations:
            # import cupy as cp
            # cp.get_default_memory_pool().free_all_blocks()
```
