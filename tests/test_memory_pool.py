import pytest
import numpy as np
from rapidshot.memory_pool import (
    NumpyMemoryPool,
    CupyMemoryPool,
    PoolExhaustedError,
    PooledBuffer
)

# Conditional import for CuPy
try:
    import cupy
    # Basic check to see if a GPU device is available for CuPy.
    # This doesn't guarantee tests will run if CUDA runtime is misconfigured,
    # but it's a step beyond just import cupy.
    if cupy.cuda.runtime.getDeviceCount() > 0:
        cupy.cuda.runtime.setDevice(0) # Set current device to 0
        cupy_available = True
    else:
        cupy_available = False
except ImportError:
    cupy_available = False
except cupy.cuda.runtime.CUDARuntimeError: # Handles cases where CUDA driver is not found or init fails
    cupy_available = False


# Default pool configurations
BUFFER_SHAPE = (100, 100, 4) # height, width, channels (BGRA)
BUFFER_DTYPE = np.uint8

# --- NumPy Memory Pool Tests ---

def test_numpy_pool_initialization():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=3)
    stats = pool.get_stats()
    assert stats['total'] == 3
    assert stats['available'] == 3
    assert stats['in_use'] == 0
    assert stats['initialized'] is True
    pool.destroy_pool()

def test_numpy_pool_checkout_and_checkin():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    
    buf_wrapper = pool.checkout()
    assert isinstance(buf_wrapper, PooledBuffer)
    assert buf_wrapper.state == 'IN_USE'
    assert buf_wrapper.shape == BUFFER_SHAPE
    assert buf_wrapper.dtype == BUFFER_DTYPE
    assert isinstance(buf_wrapper.array, np.ndarray)
    
    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 1
    
    pool.checkin(buf_wrapper)
    assert buf_wrapper.state == 'AVAILABLE'
    
    stats = pool.get_stats()
    assert stats['available'] == 1
    assert stats['in_use'] == 0
    pool.destroy_pool()

def test_numpy_pool_buffer_release_method():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    
    buf_wrapper = pool.checkout()
    assert isinstance(buf_wrapper, PooledBuffer)
    assert buf_wrapper.state == 'IN_USE'
    
    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 1
    
    buf_wrapper.release() # Use the buffer's own release method
    assert buf_wrapper.state == 'AVAILABLE'
    
    stats = pool.get_stats()
    assert stats['available'] == 1
    assert stats['in_use'] == 0
    pool.destroy_pool()

def test_numpy_pool_exhaustion():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf1 = pool.checkout()
    assert buf1 is not None
    
    with pytest.raises(PoolExhaustedError):
        pool.checkout()
        
    buf1.release() # Release the buffer to allow pool cleanup
    pool.destroy_pool()

def test_numpy_pool_release_all_buffers():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=2)
    buf1 = pool.checkout()
    buf2 = pool.checkout()
    
    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 2
    
    pool.release_all_buffers()
    
    stats = pool.get_stats()
    assert stats['available'] == 2
    assert stats['in_use'] == 0
    assert buf1.state == 'AVAILABLE'
    assert buf2.state == 'AVAILABLE'
    pool.destroy_pool()

# --- CuPy Memory Pool Tests (Conditional) ---

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_initialization():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=3)
    stats = pool.get_stats()
    assert stats['total'] == 3
    assert stats['available'] == 3
    assert stats['in_use'] == 0
    assert stats['initialized'] is True
    pool.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_checkout_and_checkin():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    
    buf_wrapper = pool.checkout()
    assert isinstance(buf_wrapper, PooledBuffer)
    assert buf_wrapper.state == 'IN_USE'
    assert buf_wrapper.shape == BUFFER_SHAPE
    assert buf_wrapper.dtype == BUFFER_DTYPE # CuPy arrays still have np.dtype compatible objects
    assert isinstance(buf_wrapper.array, cupy.ndarray)

    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 1
    
    pool.checkin(buf_wrapper)
    assert buf_wrapper.state == 'AVAILABLE'
    
    stats = pool.get_stats()
    assert stats['available'] == 1
    assert stats['in_use'] == 0
    pool.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_buffer_release_method():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    
    buf_wrapper = pool.checkout()
    assert isinstance(buf_wrapper, PooledBuffer)
    assert buf_wrapper.state == 'IN_USE'
    
    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 1
    
    buf_wrapper.release() # Use the buffer's own release method
    assert buf_wrapper.state == 'AVAILABLE'
    
    stats = pool.get_stats()
    assert stats['available'] == 1
    assert stats['in_use'] == 0
    pool.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_exhaustion():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf1 = pool.checkout()
    assert buf1 is not None
    
    with pytest.raises(PoolExhaustedError):
        pool.checkout()
        
    buf1.release() # Release the buffer to allow pool cleanup
    pool.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_release_all_buffers():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=2)
    buf1 = pool.checkout()
    buf2 = pool.checkout()
    
    stats = pool.get_stats()
    assert stats['available'] == 0
    assert stats['in_use'] == 2
    
    pool.release_all_buffers()
    
    stats = pool.get_stats()
    assert stats['available'] == 2
    assert stats['in_use'] == 0
    assert buf1.state == 'AVAILABLE'
    assert buf2.state == 'AVAILABLE'
    pool.destroy_pool()

def test_numpy_pool_destroy():
    """Test that pool can be destroyed and stats reflect it."""
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    assert pool.get_stats()['initialized'] is True
    pool.destroy_pool()
    assert pool.get_stats()['initialized'] is False
    assert pool.get_stats()['available'] == 0
    with pytest.raises(RuntimeError): # Should raise if trying to checkout from destroyed pool
        pool.checkout()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_destroy():
    """Test that cupy pool can be destroyed and stats reflect it."""
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    assert pool.get_stats()['initialized'] is True
    pool.destroy_pool()
    assert pool.get_stats()['initialized'] is False
    assert pool.get_stats()['available'] == 0
    with pytest.raises(RuntimeError): # Should raise if trying to checkout from destroyed pool
        pool.checkout()

def test_numpy_pool_checkin_invalid_buffer():
    pool1 = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    pool2 = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf_from_pool1 = pool1.checkout()

    with pytest.raises(ValueError, match="Buffer does not belong to this pool."):
        pool2.checkin(buf_from_pool1)
    
    buf_from_pool1.release() # Release it back to its original pool
    pool1.destroy_pool()
    pool2.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_checkin_invalid_buffer():
    pool1 = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    pool2 = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf_from_pool1 = pool1.checkout()

    with pytest.raises(ValueError, match="Buffer does not belong to this pool."):
        pool2.checkin(buf_from_pool1)

    buf_from_pool1.release()
    pool1.destroy_pool()
    pool2.destroy_pool()

def test_numpy_pool_double_checkin():
    pool = NumpyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf = pool.checkout()
    buf.release()
    with pytest.raises(ValueError, match="Buffer is not 'IN_USE'"):
        pool.checkin(buf) # Or buf.release() again, depending on implementation strictness
    pool.destroy_pool()

@pytest.mark.skipif(not cupy_available, reason="CuPy not available or no GPU")
def test_cupy_pool_double_checkin():
    pool = CupyMemoryPool(buffer_shape=BUFFER_SHAPE, dtype=BUFFER_DTYPE, num_buffers=1)
    buf = pool.checkout()
    buf.release()
    with pytest.raises(ValueError, match="Buffer is not 'IN_USE'"):
        pool.checkin(buf)
    pool.destroy_pool()
