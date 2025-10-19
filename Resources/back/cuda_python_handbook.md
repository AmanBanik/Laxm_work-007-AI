# CUDA Programming in Python: Complete Handbook

## Table of Contents
1. [Introduction to CUDA and GPU Computing](#chapter-1)
2. [Setting Up CUDA Environment](#chapter-2)
3. [CUDA Basics: Kernels and Thread Hierarchy](#chapter-3)
4. [Memory Management](#chapter-4)
5. [Thread Synchronization and Communication](#chapter-5)
6. [Optimization Techniques](#chapter-6)
7. [Advanced Memory Patterns](#chapter-7)
8. [Streams and Concurrency](#chapter-8)
9. [Debugging and Profiling](#chapter-9)
10. [Real-World Applications](#chapter-10)

---

## Chapter 1: Introduction to CUDA and GPU Computing {#chapter-1}

### 1.1 What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model. It enables dramatic increases in computing performance by harnessing the power of GPUs.

**Key Concepts:**
- **Host**: The CPU and its memory
- **Device**: The GPU and its memory
- **Kernel**: A function that runs on the GPU
- **Thread**: Basic execution unit on GPU

### 1.2 GPU Architecture

GPUs contain thousands of smaller cores designed for parallel processing:
- **Streaming Multiprocessors (SMs)**: Groups of CUDA cores
- **CUDA Cores**: Individual processing units
- **Memory Hierarchy**: Registers, shared memory, global memory

### 1.3 When to Use CUDA

Use CUDA when:
- Processing large datasets with independent operations
- Performing matrix operations
- Running scientific simulations
- Processing images/videos
- Training neural networks

---

## Chapter 2: Setting Up CUDA Environment {#chapter-2}

### 2.1 Installation

```bash
# Install CUDA Toolkit (varies by OS)
# For Ubuntu:
sudo apt-get install nvidia-cuda-toolkit

# Install Python CUDA libraries
pip install numba
pip install cupy-cuda11x  # Replace with your CUDA version
pip install pycuda
```

### 2.2 Verifying Installation

```python
# Using Numba
from numba import cuda
import numpy as np

# Check if CUDA is available
print("CUDA Available:", cuda.is_available())

# Get GPU information
if cuda.is_available():
    gpu = cuda.get_current_device()
    print(f"GPU Name: {gpu.name}")
    print(f"Compute Capability: {gpu.compute_capability}")
    print(f"Total Memory: {gpu.total_memory / 1e9:.2f} GB")
```

### 2.3 Choosing a Python CUDA Library

**Numba**: JIT compiler, easiest to learn, Pythonic syntax
**CuPy**: NumPy-like interface, great for array operations
**PyCUDA**: Low-level control, requires CUDA C knowledge

---

## Chapter 3: CUDA Basics: Kernels and Thread Hierarchy {#chapter-3}

### 3.1 Your First CUDA Kernel

```python
from numba import cuda
import numpy as np

@cuda.jit
def add_kernel(x, y, out):
    """Simple element-wise addition kernel"""
    idx = cuda.grid(1)  # Get global thread ID
    if idx < out.size:
        out[idx] = x[idx] + y[idx]

# Host code
n = 100000
x = np.ones(n, dtype=np.float32)
y = np.ones(n, dtype=np.float32)
out = np.zeros(n, dtype=np.float32)

# Copy data to device
x_device = cuda.to_device(x)
y_device = cuda.to_device(y)
out_device = cuda.device_array(n, dtype=np.float32)

# Configure and launch kernel
threads_per_block = 256
blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
add_kernel[blocks_per_grid, threads_per_block](x_device, y_device, out_device)

# Copy result back to host
out = out_device.copy_to_host()
print(f"Result: {out[:10]}")  # Should be all 2.0s
```

### 3.2 Thread Hierarchy

CUDA organizes threads into a three-level hierarchy:

```python
@cuda.jit
def thread_hierarchy_demo(output):
    """Demonstrates thread indexing"""
    # Thread indices within block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    # Block indices within grid
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z
    
    # Block dimensions
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y
    bd = cuda.blockDim.z
    
    # Calculate global thread index
    x = tx + bx * bw
    y = ty + by * bh
    z = tz + bz * bd
    
    # Example: 2D grid calculation
    idx = x + y * cuda.gridDim.x * cuda.blockDim.x

# Launch with 2D grid and blocks
threads_per_block = (16, 16)
blocks_per_grid = (4, 4)
output = cuda.device_array((64, 64))
thread_hierarchy_demo[blocks_per_grid, threads_per_block](output)
```

### 3.3 Grid Stride Loops

For processing arrays larger than the grid:

```python
@cuda.jit
def grid_stride_kernel(x, y, out):
    """Handles arbitrary array sizes"""
    start = cuda.grid(1)
    stride = cuda.gridsize(1)
    
    for i in range(start, out.size, stride):
        out[i] = x[i] + y[i]

# Works with any array size
n = 1000000
x_device = cuda.to_device(np.ones(n))
y_device = cuda.to_device(np.ones(n))
out_device = cuda.device_array(n)

grid_stride_kernel[1024, 256](x_device, y_device, out_device)
```

---

## Chapter 4: Memory Management {#chapter-4}

### 4.1 Memory Types

**Global Memory**: Large, slow, accessible by all threads
**Shared Memory**: Fast, limited, shared within a block
**Local Memory**: Private to each thread
**Constant Memory**: Read-only, cached
**Registers**: Fastest, very limited

### 4.2 Memory Allocation and Transfer

```python
import numpy as np
from numba import cuda

# Host arrays
h_array = np.random.rand(1000).astype(np.float32)

# Device allocation methods
d_array1 = cuda.to_device(h_array)  # Copy from host
d_array2 = cuda.device_array(1000, dtype=np.float32)  # Allocate empty
d_array3 = cuda.device_array_like(h_array)  # Same shape/dtype

# Copy device to host
result = d_array1.copy_to_host()

# In-place copy
cuda.to_device(h_array, to=d_array2)

# Free device memory explicitly
del d_array1, d_array2, d_array3
```

### 4.3 Shared Memory

```python
from numba import cuda
import numpy as np

@cuda.jit
def shared_memory_example(data, result):
    """Using shared memory for faster access"""
    # Allocate shared memory
    shared = cuda.shared.array(shape=(256,), dtype=float32)
    
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = tid + bid * cuda.blockDim.x
    
    # Load data into shared memory
    if idx < data.size:
        shared[tid] = data[idx]
    
    # Synchronize threads in block
    cuda.syncthreads()
    
    # Process using shared memory (faster!)
    if idx < data.size:
        # Example: compute average of neighbors
        left = shared[tid - 1] if tid > 0 else 0
        right = shared[tid + 1] if tid < 255 else 0
        result[idx] = (left + shared[tid] + right) / 3.0
    
    cuda.syncthreads()

# Usage
n = 10000
data = cuda.to_device(np.random.rand(n).astype(np.float32))
result = cuda.device_array(n, dtype=np.float32)
shared_memory_example[n // 256 + 1, 256](data, result)
```

### 4.4 Memory Coalescing

```python
@cuda.jit
def coalesced_access(matrix, result):
    """Good: Coalesced memory access pattern"""
    row = cuda.blockIdx.x
    col = cuda.threadIdx.x
    
    # Threads in a warp access consecutive memory
    if row < matrix.shape[0] and col < matrix.shape[1]:
        result[row, col] = matrix[row, col] * 2

@cuda.jit
def uncoalesced_access(matrix, result):
    """Bad: Uncoalesced memory access pattern"""
    row = cuda.threadIdx.x
    col = cuda.blockIdx.x
    
    # Threads in a warp access strided memory (slower!)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        result[row, col] = matrix[row, col] * 2
```

---

## Chapter 5: Thread Synchronization and Communication {#chapter-5}

### 5.1 Barrier Synchronization

```python
from numba import cuda
import numpy as np

@cuda.jit
def parallel_prefix_sum(data, output):
    """Compute prefix sum using synchronization"""
    shared = cuda.shared.array(256, dtype=float32)
    tid = cuda.threadIdx.x
    
    # Load data into shared memory
    shared[tid] = data[tid] if tid < data.size else 0
    cuda.syncthreads()
    
    # Up-sweep phase
    offset = 1
    d = 128
    while d > 0:
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            shared[bi] += shared[ai]
        offset *= 2
        d //= 2
        cuda.syncthreads()
    
    # Clear last element
    if tid == 0:
        shared[255] = 0
    cuda.syncthreads()
    
    # Down-sweep phase
    d = 1
    offset = 128
    while d < 256:
        if tid < d:
            ai = offset * (2 * tid + 1) - 1
            bi = offset * (2 * tid + 2) - 1
            temp = shared[ai]
            shared[ai] = shared[bi]
            shared[bi] += temp
        d *= 2
        offset //= 2
        cuda.syncthreads()
    
    # Write result
    if tid < data.size:
        output[tid] = shared[tid]
```

### 5.2 Atomic Operations

```python
from numba import cuda
import numpy as np

@cuda.jit
def atomic_histogram(data, bins, hist):
    """Compute histogram using atomic operations"""
    idx = cuda.grid(1)
    
    if idx < data.size:
        bin_idx = int(data[idx] * bins)
        if bin_idx >= bins:
            bin_idx = bins - 1
        
        # Atomic add to prevent race conditions
        cuda.atomic.add(hist, bin_idx, 1)

# Usage
n = 1000000
data = cuda.to_device(np.random.rand(n).astype(np.float32))
hist = cuda.to_device(np.zeros(10, dtype=np.int32))

atomic_histogram[n // 256 + 1, 256](data, 10, hist)
result = hist.copy_to_host()
print(f"Histogram: {result}")
```

### 5.3 Warp-Level Primitives

```python
from numba import cuda

@cuda.jit
def warp_reduce_sum(data, result):
    """Efficient reduction using warp shuffles"""
    tid = cuda.threadIdx.x
    lane = tid % 32  # Lane within warp
    warp_id = tid // 32
    
    # Shared memory for partial sums
    shared = cuda.shared.array(32, dtype=float32)
    
    # Load data
    val = data[tid] if tid < data.size else 0.0
    
    # Warp-level reduction using shuffle
    offset = 16
    while offset > 0:
        val += cuda.shfl_down_sync(0xffffffff, val, offset)
        offset //= 2
    
    # First thread in warp writes to shared memory
    if lane == 0:
        shared[warp_id] = val
    
    cuda.syncthreads()
    
    # First warp reduces partial sums
    if warp_id == 0:
        val = shared[lane] if lane < 32 else 0.0
        offset = 16
        while offset > 0:
            val += cuda.shfl_down_sync(0xffffffff, val, offset)
            offset //= 2
        
        if lane == 0:
            result[0] = val
```

---

## Chapter 6: Optimization Techniques {#chapter-6}

### 6.1 Occupancy Optimization

```python
from numba import cuda

# Check occupancy for different configurations
def analyze_occupancy(threads_per_block):
    """Analyze kernel occupancy"""
    @cuda.jit
    def sample_kernel(data):
        idx = cuda.grid(1)
        if idx < data.size:
            data[idx] *= 2
    
    # Get occupancy information
    device = cuda.get_current_device()
    max_threads = device.MAX_THREADS_PER_BLOCK
    
    print(f"Threads per block: {threads_per_block}")
    print(f"Max threads per block: {max_threads}")
    print(f"Occupancy: {threads_per_block / max_threads * 100:.1f}%")

# Test different configurations
for tpb in [64, 128, 256, 512, 1024]:
    analyze_occupancy(tpb)
```

### 6.2 Bank Conflict Avoidance

```python
from numba import cuda
import numpy as np

@cuda.jit
def no_bank_conflict(data, result):
    """Padding to avoid bank conflicts"""
    # Shared memory with padding (33 instead of 32)
    shared = cuda.shared.array((32, 33), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Load with padding
    shared[tx, ty] = data[tx, ty]
    cuda.syncthreads()
    
    # Transpose (now conflict-free)
    result[ty, tx] = shared[tx, ty]
```

### 6.3 Loop Unrolling

```python
from numba import cuda

@cuda.jit
def unrolled_dot_product(a, b, result):
    """Manually unrolled loop for better performance"""
    idx = cuda.grid(1)
    
    if idx < a.size // 4:
        base = idx * 4
        # Process 4 elements at once
        sum_val = (a[base] * b[base] +
                   a[base + 1] * b[base + 1] +
                   a[base + 2] * b[base + 2] +
                   a[base + 3] * b[base + 3])
        cuda.atomic.add(result, 0, sum_val)
```

### 6.4 Register Pressure Management

```python
from numba import cuda
import numpy as np

@cuda.jit
def high_register_pressure(data):
    """Too many local variables = register spilling"""
    idx = cuda.grid(1)
    if idx < data.size:
        # Many temporary variables increase register usage
        t1 = data[idx] * 2.0
        t2 = data[idx] * 3.0
        t3 = data[idx] * 4.0
        t4 = t1 + t2
        t5 = t3 + t4
        data[idx] = t5

@cuda.jit
def optimized_registers(data):
    """Reuse variables to reduce register pressure"""
    idx = cuda.grid(1)
    if idx < data.size:
        # Reuse variable names
        temp = data[idx]
        temp = temp * 2.0 + temp * 3.0 + temp * 4.0
        data[idx] = temp
```

---

## Chapter 7: Advanced Memory Patterns {#chapter-7}

### 7.1 Tiled Matrix Multiplication

```python
from numba import cuda
import numpy as np

TILE_SIZE = 16

@cuda.jit
def matmul_tiled(A, B, C):
    """Optimized matrix multiplication using tiling"""
    # Shared memory tiles
    sA = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    sB = cuda.shared.array((TILE_SIZE, TILE_SIZE), dtype=float32)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    
    row = by * TILE_SIZE + ty
    col = bx * TILE_SIZE + tx
    
    tmp = 0.0
    
    # Loop over tiles
    for m in range((A.shape[1] + TILE_SIZE - 1) // TILE_SIZE):
        # Load tile into shared memory
        if row < A.shape[0] and m * TILE_SIZE + tx < A.shape[1]:
            sA[ty, tx] = A[row, m * TILE_SIZE + tx]
        else:
            sA[ty, tx] = 0.0
        
        if col < B.shape[1] and m * TILE_SIZE + ty < B.shape[0]:
            sB[ty, tx] = B[m * TILE_SIZE + ty, col]
        else:
            sB[ty, tx] = 0.0
        
        cuda.syncthreads()
        
        # Compute partial product
        for k in range(TILE_SIZE):
            tmp += sA[ty, k] * sB[k, tx]
        
        cuda.syncthreads()
    
    # Write result
    if row < C.shape[0] and col < C.shape[1]:
        C[row, col] = tmp

# Usage
N = 1024
A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)

A_device = cuda.to_device(A)
B_device = cuda.to_device(B)
C_device = cuda.to_device(C)

threads_per_block = (TILE_SIZE, TILE_SIZE)
blocks_per_grid = ((N + TILE_SIZE - 1) // TILE_SIZE,
                   (N + TILE_SIZE - 1) // TILE_SIZE)

matmul_tiled[blocks_per_grid, threads_per_block](A_device, B_device, C_device)
```

### 7.2 Reduction Patterns

```python
from numba import cuda
import numpy as np

@cuda.jit
def parallel_reduction(data, partial_sums):
    """Efficient parallel reduction"""
    shared = cuda.shared.array(256, dtype=float32)
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.x
    idx = bid * cuda.blockDim.x + tid
    
    # Load data into shared memory
    if idx < data.size:
        shared[tid] = data[idx]
    else:
        shared[tid] = 0.0
    cuda.syncthreads()
    
    # Reduction in shared memory
    s = cuda.blockDim.x // 2
    while s > 0:
        if tid < s:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2
    
    # Write block result
    if tid == 0:
        partial_sums[bid] = shared[0]

def sum_reduce(data):
    """Complete reduction using multiple kernel launches"""
    n = data.size
    threads_per_block = 256
    blocks = (n + threads_per_block - 1) // threads_per_block
    
    d_data = cuda.to_device(data)
    d_partial = cuda.device_array(blocks, dtype=np.float32)
    
    # First reduction
    parallel_reduction[blocks, threads_per_block](d_data, d_partial)
    
    # Reduce partial sums on CPU (small array)
    partial = d_partial.copy_to_host()
    return np.sum(partial)

# Test
data = np.random.rand(1000000).astype(np.float32)
gpu_sum = sum_reduce(data)
cpu_sum = np.sum(data)
print(f"GPU sum: {gpu_sum:.6f}, CPU sum: {cpu_sum:.6f}")
```

### 7.3 Texture Memory (CuPy)

```python
import cupy as cp

# Texture memory provides caching and interpolation
def use_texture_memory():
    """Example using CuPy for texture-like operations"""
    # Create array
    data = cp.random.rand(1000, 1000, dtype=cp.float32)
    
    # Texture memory is good for spatial locality
    # CuPy automatically optimizes memory access patterns
    result = cp.roll(data, shift=(1, 1), axis=(0, 1))
    
    return result
```

---

## Chapter 8: Streams and Concurrency {#chapter-8}

### 8.1 CUDA Streams

```python
from numba import cuda
import numpy as np

@cuda.jit
def vector_add(a, b, c):
    idx = cuda.grid(1)
    if idx < c.size:
        c[idx] = a[idx] + b[idx]

def concurrent_execution():
    """Execute multiple kernels concurrently using streams"""
    n = 1000000
    n_streams = 4
    chunk_size = n // n_streams
    
    # Create streams
    streams = [cuda.stream() for _ in range(n_streams)]
    
    # Host arrays
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    # Device arrays for each stream
    d_arrays = []
    for i in range(n_streams):
        start = i * chunk_size
        end = start + chunk_size
        
        d_a = cuda.to_device(a[start:end], stream=streams[i])
        d_b = cuda.to_device(b[start:end], stream=streams[i])
        d_c = cuda.device_array(chunk_size, dtype=np.float32, stream=streams[i])
        
        d_arrays.append((d_a, d_b, d_c))
        
        # Launch kernel on stream
        threads = 256
        blocks = (chunk_size + threads - 1) // threads
        vector_add[blocks, threads, streams[i]](d_a, d_b, d_c)
    
    # Copy results back
    for i, (_, _, d_c) in enumerate(d_arrays):
        start = i * chunk_size
        end = start + chunk_size
        d_c.copy_to_host(c[start:end], stream=streams[i])
    
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()
    
    return c

result = concurrent_execution()
print(f"First 10 results: {result[:10]}")
```

### 8.2 Overlapping Computation and Transfer

```python
from numba import cuda
import numpy as np

@cuda.jit
def compute_kernel(data, result):
    idx = cuda.grid(1)
    if idx < data.size:
        # Simulate computation
        temp = data[idx]
        for _ in range(100):
            temp = temp * 1.01
        result[idx] = temp

def overlap_compute_transfer():
    """Overlap data transfer with computation"""
    n_chunks = 4
    chunk_size = 250000
    
    # Pinned memory for faster transfers
    a_pinned = cuda.pinned_array(chunk_size, dtype=np.float32)
    result_pinned = cuda.pinned_array(chunk_size, dtype=np.float32)
    
    streams = [cuda.stream() for _ in range(n_chunks)]
    
    d_data = [cuda.device_array(chunk_size, dtype=np.float32) 
              for _ in range(n_chunks)]
    d_result = [cuda.device_array(chunk_size, dtype=np.float32) 
                for _ in range(n_chunks)]
    
    results = []
    
    for i in range(n_chunks):
        # Generate data
        a_pinned[:] = np.random.rand(chunk_size).astype(np.float32)
        
        # Async copy to device
        d_data[i].copy_to_device(a_pinned, stream=streams[i])
        
        # Launch kernel
        threads = 256
        blocks = (chunk_size + threads - 1) // threads
        compute_kernel[blocks, threads, streams[i]](d_data[i], d_result[i])
        
        # Async copy back
        d_result[i].copy_to_host(result_pinned, stream=streams[i])
        
        results.append(result_pinned.copy())
    
    # Synchronize
    cuda.synchronize()
    
    return np.concatenate(results)
```

### 8.3 Multi-GPU Programming

```python
from numba import cuda
import numpy as np

@cuda.jit
def simple_kernel(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] *= 2

def multi_gpu_computation():
    """Distribute work across multiple GPUs"""
    n_gpus = cuda.gpus.count
    print(f"Found {n_gpus} GPUs")
    
    if n_gpus < 2:
        print("Need at least 2 GPUs for this example")
        return
    
    n = 1000000
    chunk_size = n // n_gpus
    
    results = []
    
    for gpu_id in range(n_gpus):
        # Select GPU
        cuda.select_device(gpu_id)
        
        # Create data for this GPU
        data = np.random.rand(chunk_size).astype(np.float32)
        d_data = cuda.to_device(data)
        
        # Launch kernel
        threads = 256
        blocks = (chunk_size + threads - 1) // threads
        simple_kernel[blocks, threads](d_data)
        
        # Copy result back
        result = d_data.copy_to_host()
        results.append(result)
    
    return np.concatenate(results)
```

---

## Chapter 9: Debugging and Profiling {#chapter-9}

### 9.1 Error Checking

```python
from numba import cuda
import numpy as np

@cuda.jit
def kernel_with_error_check(data):
    idx = cuda.grid(1)
    if idx < data.size:
        data[idx] = 1.0 / data[idx]  # Division by zero possible!

def safe_kernel_launch():
    """Proper error handling"""
    try:
        n = 1000
        data = np.random.rand(n).astype(np.float32)
        data[500] = 0.0  # Introduce error
        
        d_data = cuda.to_device(data)
        
        threads = 256
        blocks = (n + threads - 1) // threads
        kernel_with_error_check[blocks, threads](d_data)
        
        # Synchronize to catch errors
        cuda.synchronize()
        
        result = d_data.copy_to_host()
        print("Kernel executed successfully")
        
    except cuda.CudaAPIError as e:
        print(f"CUDA Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
```

### 9.2 Performance Timing

```python
from numba import cuda
import numpy as np
import time

@cuda.jit
def benchmark_kernel(data, iterations):
    idx = cuda.grid(1)
    if idx < data.size:
        val = data[idx]
        for _ in range(iterations):
            val = val * 1.01 + 0.5
        data[idx] = val

def benchmark_performance():
    """Accurate GPU timing"""
    n = 1000000
    iterations = 100
    
    data = np.random.rand(n).astype(np.float32)
    d_data = cuda.to_device(data)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    
    # Warmup
    benchmark_kernel[blocks, threads](d_data, iterations)
    cuda.synchronize()
    
    # Timing using CUDA events
    start_event = cuda.event()
    end_event = cuda.event()
    
    start_event.record()
    benchmark_kernel[blocks, threads](d_data, iterations)
    end_event.record()
    end_event.synchronize()
    
    elapsed_time = cuda.event_elapsed_time(start_event, end_event)
    print(f"Kernel execution time: {elapsed_time:.2f} ms")
    
    # Calculate throughput
    operations = n * iterations
    throughput = operations / (elapsed_time / 1000) / 1e9
    print(f"Throughput: {throughput:.2f} GFLOPS")

benchmark_performance()
```

### 9.3 Memory Profiling

```python
from numba import cuda
import numpy as np

def profile_memory_usage():
    """Monitor GPU memory usage"""
    def print_memory_info():
        meminfo = cuda.current_context().get_memory_info()
        free = meminfo[0] / 1e9
        total = meminfo[1] / 1e9
        used = total - free
        print(f"GPU Memory - Used: {used:.2f} GB, Free: {free:.2f} GB, Total: {total:.2f} GB")
    
    print("Initial memory:")
    print_memory_info()
    
    # Allocate arrays
    arrays = []
    for i in range(5):
        arr = cuda.device_array(10000000, dtype=np.float32)
        arrays.append(arr)
        print(f"\nAfter allocating array {i+1}:")
        print_memory_info()
    
    # Free memory
    del arrays
    cuda.synchronize()
    
    print("\nAfter freeing arrays:")
    print_memory_info()

profile_memory_usage()
```

### 9.4 Debugging with CUDA-MEMCHECK

```python
from numba import cuda
import numpy as np
import os

@cuda.jit
def buggy_kernel(data, output):
    """Kernel with potential memory errors"""
    idx = cuda.grid(1)
    # Bug: no bounds checking
    output[idx] = data[idx] * 2  # May access out of bounds!

def run_with_memcheck():
    """
    Run with cuda-memcheck from command line:
    cuda-memcheck python your_script.py
    """
    n = 1000
    data = np.random.rand(n).astype(np.float32)
    output = np.zeros(n - 10, dtype=np.float32)  # Intentionally smaller
    
    d_data = cuda.to_device(data)
    d_output = cuda.to_device(output)
    
    threads = 256
    blocks = (n + threads - 1) // threads
    buggy_kernel[blocks, threads](d_data, d_output)
    
    try:
        cuda.synchronize()
    except Exception as e:
        print(f"Error detected: {e}")
```

### 9.5 Performance Analysis

```python
from numba import cuda
import numpy as np
import time

def compare_implementations():
    """Compare CPU vs GPU performance"""
    n = 10000000
    
    # CPU implementation
    a_cpu = np.random.rand(n).astype(np.float32)
    b_cpu = np.random.rand(n).astype(np.float32)
    
    start = time.time()
    c_cpu = a_cpu + b_cpu
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time*1000:.2f} ms")
    
    # GPU implementation
    @cuda.jit
    def add_gpu(a, b, c):
        idx = cuda.grid(1)
        if idx < c.size:
            c[idx] = a[idx] + b[idx]
    
    d_a = cuda.to_device(a_cpu)
    d_b = cuda.to_device(b_cpu)
    d_c = cuda.device_array(n, dtype=np.float32)
    
    # Warmup
    add_gpu[(n + 255) // 256, 256](d_a, d_b, d_c)
    cuda.synchronize()
    
    start = time.time()
    add_gpu[(n + 255) // 256, 256](d_a, d_b, d_c)
    cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time*1000:.2f} ms")
    
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")

compare_implementations()
```

---

## Chapter 10: Real-World Applications {#chapter-10}

### 10.1 Image Processing: Gaussian Blur

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def gaussian_blur(image, output, kernel, kernel_size):
    """Apply Gaussian blur to an image"""
    x, y = cuda.grid(2)
    height, width = image.shape
    
    if x < width and y < height:
        sum_val = 0.0
        kernel_sum = 0.0
        half_size = kernel_size // 2
        
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                # Calculate image coordinates
                ix = x + kx - half_size
                iy = y + ky - half_size
                
                # Check bounds
                if 0 <= ix < width and 0 <= iy < height:
                    k_val = kernel[ky, kx]
                    sum_val += image[iy, ix] * k_val
                    kernel_sum += k_val
        
        output[y, x] = sum_val / kernel_sum if kernel_sum > 0 else 0

def apply_gaussian_blur(image, sigma=1.0, kernel_size=5):
    """Complete Gaussian blur implementation"""
    # Create Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = math.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel /= kernel.sum()
    
    # Transfer to GPU
    d_image = cuda.to_device(image.astype(np.float32))
    d_output = cuda.device_array_like(d_image)
    d_kernel = cuda.to_device(kernel)
    
    # Configure grid
    threads_per_block = (16, 16)
    blocks_per_grid_x = (image.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (image.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    
    # Launch kernel
    gaussian_blur[blocks_per_grid, threads_per_block](
        d_image, d_output, d_kernel, kernel_size
    )
    
    return d_output.copy_to_host()

# Example usage
image = np.random.rand(1920, 1080).astype(np.float32)
blurred = apply_gaussian_blur(image, sigma=2.0, kernel_size=7)
```

### 10.2 Monte Carlo Simulation

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def monte_carlo_pi(rng_states, counts):
    """Estimate Pi using Monte Carlo method"""
    idx = cuda.grid(1)
    
    if idx < rng_states.size:
        # Simple LCG random number generator
        seed = rng_states[idx]
        hits = 0
        
        for _ in range(10000):
            # Generate random point
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            x = (seed / 2147483647.0) * 2 - 1
            
            seed = (1103515245 * seed + 12345) & 0x7fffffff
            y = (seed / 2147483647.0) * 2 - 1
            
            # Check if inside circle
            if x*x + y*y <= 1.0:
                hits += 1
        
        counts[idx] = hits
        rng_states[idx] = seed

def estimate_pi(n_threads=10000):
    """Parallel Pi estimation"""
    # Initialize random states
    rng_states = np.random.randint(0, 2**31, n_threads, dtype=np.int32)
    counts = np.zeros(n_threads, dtype=np.int32)
    
    d_rng = cuda.to_device(rng_states)
    d_counts = cuda.to_device(counts)
    
    # Launch kernel
    threads = 256
    blocks = (n_threads + threads - 1) // threads
    monte_carlo_pi[blocks, threads](d_rng, d_counts)
    
    # Calculate Pi
    counts = d_counts.copy_to_host()
    total_hits = counts.sum()
    total_samples = n_threads * 10000
    pi_estimate = 4.0 * total_hits / total_samples
    
    print(f"Pi estimate: {pi_estimate:.6f}")
    print(f"Error: {abs(pi_estimate - math.pi):.6f}")
    
    return pi_estimate

estimate_pi()
```

### 10.3 Signal Processing: FFT Convolution

```python
from numba import cuda
import numpy as np
import cupy as cp

def fft_convolution_gpu(signal, kernel):
    """Fast convolution using FFT on GPU"""
    # Convert to CuPy arrays
    signal_gpu = cp.asarray(signal)
    kernel_gpu = cp.asarray(kernel)
    
    # Pad to next power of 2
    n = len(signal) + len(kernel) - 1
    fft_size = 2**int(np.ceil(np.log2(n)))
    
    # Pad arrays
    signal_padded = cp.pad(signal_gpu, (0, fft_size - len(signal)))
    kernel_padded = cp.pad(kernel_gpu, (0, fft_size - len(kernel)))
    
    # FFT
    signal_fft = cp.fft.fft(signal_padded)
    kernel_fft = cp.fft.fft(kernel_padded)
    
    # Multiply in frequency domain
    result_fft = signal_fft * kernel_fft
    
    # Inverse FFT
    result = cp.fft.ifft(result_fft).real[:n]
    
    return cp.asnumpy(result)

# Example
signal = np.random.randn(100000)
kernel = np.exp(-np.linspace(-3, 3, 100)**2)
result = fft_convolution_gpu(signal, kernel)
```

### 10.4 Machine Learning: Matrix Operations

```python
from numba import cuda
import numpy as np

@cuda.jit
def relu_activation(x, out):
    """ReLU activation function"""
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = max(0.0, x[idx])

@cuda.jit
def relu_gradient(x, grad_out, grad_in):
    """ReLU gradient for backpropagation"""
    idx = cuda.grid(1)
    if idx < x.size:
        grad_in[idx] = grad_out[idx] if x[idx] > 0 else 0.0

@cuda.jit
def sigmoid_activation(x, out):
    """Sigmoid activation function"""
    idx = cuda.grid(1)
    if idx < x.size:
        out[idx] = 1.0 / (1.0 + math.exp(-x[idx]))

@cuda.jit
def batch_normalization(x, mean, var, gamma, beta, out, epsilon=1e-5):
    """Batch normalization"""
    idx = cuda.grid(1)
    if idx < x.size:
        normalized = (x[idx] - mean[0]) / math.sqrt(var[0] + epsilon)
        out[idx] = gamma[0] * normalized + beta[0]

class GPUNeuralNetLayer:
    """Simple neural network layer on GPU"""
    
    def __init__(self, input_size, output_size):
        self.weights = cuda.to_device(
            np.random.randn(input_size, output_size).astype(np.float32) * 0.01
        )
        self.bias = cuda.to_device(np.zeros(output_size, dtype=np.float32))
    
    def forward(self, x):
        """Forward pass using matrix multiplication"""
        # Use CuPy for efficient matrix operations
        import cupy as cp
        
        x_cp = cp.asarray(x)
        w_cp = cp.asarray(self.weights.copy_to_host())
        b_cp = cp.asarray(self.bias.copy_to_host())
        
        # Matrix multiplication
        output = cp.dot(x_cp, w_cp) + b_cp
        
        return output

# Example usage
layer = GPUNeuralNetLayer(784, 128)
x = np.random.randn(32, 784).astype(np.float32)  # Batch of 32
output = layer.forward(x)
```

### 10.5 N-Body Simulation

```python
from numba import cuda
import numpy as np
import math

@cuda.jit
def compute_forces(positions, velocities, masses, dt, G=1.0):
    """N-body gravitational simulation"""
    i = cuda.grid(1)
    n = positions.shape[0]
    
    if i < n:
        fx = 0.0
        fy = 0.0
        fz = 0.0
        
        for j in range(n):
            if i != j:
                # Calculate distance
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                
                dist_sq = dx*dx + dy*dy + dz*dz + 1e-10  # Softening
                dist = math.sqrt(dist_sq)
                
                # Force magnitude
                force = G * masses[i] * masses[j] / dist_sq
                
                # Force components
                fx += force * dx / dist
                fy += force * dy / dist
                fz += force * dz / dist
        
        # Update velocity
        velocities[i, 0] += fx / masses[i] * dt
        velocities[i, 1] += fy / masses[i] * dt
        velocities[i, 2] += fz / masses[i] * dt
        
        # Update position
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt
        positions[i, 2] += velocities[i, 2] * dt

def run_nbody_simulation(n_bodies=1000, n_steps=100):
    """Run N-body simulation"""
    # Initialize random positions and velocities
    positions = np.random.randn(n_bodies, 3).astype(np.float32)
    velocities = np.random.randn(n_bodies, 3).astype(np.float32) * 0.1
    masses = np.ones(n_bodies, dtype=np.float32)
    
    d_positions = cuda.to_device(positions)
    d_velocities = cuda.to_device(velocities)
    d_masses = cuda.to_device(masses)
    
    dt = 0.01
    threads = 256
    blocks = (n_bodies + threads - 1) // threads
    
    # Simulation loop
    for step in range(n_steps):
        compute_forces[blocks, threads](
            d_positions, d_velocities, d_masses, dt
        )
        
        if step % 10 == 0:
            print(f"Step {step}/{n_steps}")
    
    # Copy final positions back
    final_positions = d_positions.copy_to_host()
    return final_positions

# Run simulation
final_pos = run_nbody_simulation(n_bodies=1000, n_steps=100)
print(f"Simulation complete. Final positions shape: {final_pos.shape}")
```

### 10.6 Data Analytics: Parallel Sorting

```python
from numba import cuda
import numpy as np

@cuda.jit
def bitonic_sort_step(data, j, k):
    """Single step of bitonic sort"""
    i = cuda.grid(1)
    ixj = i ^ j
    
    if ixj > i:
        if (i & k) == 0:
            if data[i] > data[ixj]:
                # Swap
                data[i], data[ixj] = data[ixj], data[i]
        else:
            if data[i] < data[ixj]:
                # Swap
                data[i], data[ixj] = data[ixj], data[i]

def gpu_bitonic_sort(data):
    """Parallel bitonic sort on GPU"""
    n = len(data)
    # Pad to power of 2
    next_power = 2**int(np.ceil(np.log2(n)))
    padded = np.pad(data, (0, next_power - n), constant_values=np.inf)
    
    d_data = cuda.to_device(padded.astype(np.float32))
    
    threads = 256
    blocks = (next_power + threads - 1) // threads
    
    k = 2
    while k <= next_power:
        j = k // 2
        while j > 0:
            bitonic_sort_step[blocks, threads](d_data, j, k)
            cuda.synchronize()
            j //= 2
        k *= 2
    
    result = d_data.copy_to_host()
    return result[:n]

# Test
data = np.random.rand(1000).astype(np.float32)
sorted_data = gpu_bitonic_sort(data)
print(f"Sorted correctly: {np.all(sorted_data[:-1] <= sorted_data[1:])}")
```

### 10.7 Cryptography: Parallel Hashing

```python
from numba import cuda
import numpy as np

@cuda.jit
def parallel_hash(data, hashes, prime=31, mod=1000000007):
    """Simple parallel polynomial rolling hash"""
    idx = cuda.grid(1)
    
    if idx < data.shape[0]:
        hash_val = 0
        power = 1
        
        for i in range(data.shape[1]):
            hash_val = (hash_val + data[idx, i] * power) % mod
            power = (power * prime) % mod
        
        hashes[idx] = hash_val

def compute_hashes(strings):
    """Compute hashes for multiple strings in parallel"""
    # Convert strings to integer arrays
    max_len = max(len(s) for s in strings)
    data = np.zeros((len(strings), max_len), dtype=np.int32)
    
    for i, s in enumerate(strings):
        for j, c in enumerate(s):
            data[i, j] = ord(c)
    
    hashes = np.zeros(len(strings), dtype=np.int64)
    
    d_data = cuda.to_device(data)
    d_hashes = cuda.to_device(hashes)
    
    threads = 256
    blocks = (len(strings) + threads - 1) // threads
    parallel_hash[blocks, threads](d_data, d_hashes)
    
    return d_hashes.copy_to_host()

# Example
strings = [f"string_{i}" for i in range(10000)]
hashes = compute_hashes(strings)
print(f"Computed {len(hashes)} hashes")
print(f"Sample hashes: {hashes[:5]}")
```

---

## Appendix A: Best Practices Summary

### Performance Best Practices
1. **Maximize Occupancy**: Use 128-256 threads per block
2. **Minimize Divergence**: Avoid branching within warps
3. **Coalesce Memory Access**: Align memory access patterns
4. **Use Shared Memory**: Cache frequently accessed data
5. **Minimize Host-Device Transfer**: Keep data on GPU
6. **Async Operations**: Use streams for concurrency
7. **Profile First**: Measure before optimizing

### Memory Best Practices
1. **Prefer Shared Memory**: For repeated access within blocks
2. **Avoid Bank Conflicts**: Pad shared memory arrays
3. **Use Pinned Memory**: For faster host-device transfers
4. **Minimize Registers**: Reduce local variable usage
5. **Check Alignment**: Ensure proper memory alignment

### Code Organization
1. **Small Kernels**: Keep kernels focused and simple
2. **Error Checking**: Always check for CUDA errors
3. **Documentation**: Comment complex memory patterns
4. **Testing**: Verify against CPU implementations
5. **Profiling**: Use nvprof or Nsight tools

---

## Appendix B: Common Pitfalls

### Pitfall 1: Race Conditions
```python
# Wrong: Race condition
@cuda.jit
def bad_reduction(data, result):
    idx = cuda.grid(1)
    if idx < data.size:
        result[0] += data[idx]  # Race condition!

# Correct: Use atomic operations
@cuda.jit
def good_reduction(data, result):
    idx = cuda.grid(1)
    if idx < data.size:
        cuda.atomic.add(result, 0, data[idx])
```

### Pitfall 2: Forgetting Synchronization
```python
# Wrong: No synchronization
@cuda.jit
def no_sync(data):
    shared = cuda.shared.array(256, float32)
    tid = cuda.threadIdx.x
    shared[tid] = data[tid]
    # Missing cuda.syncthreads()!
    data[tid] = shared[tid + 1]  # May read uninitialized data

# Correct: Add synchronization
@cuda.jit
def with_sync(data):
    shared = cuda.shared.array(256, float32)
    tid = cuda.threadIdx.x
    shared[tid] = data[tid]
    cuda.syncthreads()  # Wait for all threads
    if tid < 255:
        data[tid] = shared[tid + 1]
```

### Pitfall 3: Exceeding Shared Memory
```python
# Wrong: Too much shared memory
@cuda.jit
def too_much_shared(data):
    # May exceed available shared memory!
    shared = cuda.shared.array(100000, float32)
    # ...

# Correct: Check device limits
device = cuda.get_current_device()
max_shared = device.MAX_SHARED_MEMORY_PER_BLOCK
print(f"Max shared memory: {max_shared} bytes")
```

---

## Appendix C: Quick Reference

### Common Patterns
```python
# Get global thread ID (1D)
idx = cuda.grid(1)

# Get global thread ID (2D)
x, y = cuda.grid(2)

# Grid stride loop
start = cuda.grid(1)
stride = cuda.gridsize(1)
for i in range(start, n, stride):
    # process data[i]

# Shared memory allocation
shared = cuda.shared.array(shape, dtype)

# Synchronization
cuda.syncthreads()

# Atomic operations
cuda.atomic.add(array, index, value)
cuda.atomic.max(array, index, value)
cuda.atomic.min(array, index, value)
```

### Useful Functions
```python
# Memory operations
cuda.to_device(array)
cuda.device_array(shape, dtype)
cuda.pinned_array(shape, dtype)
array.copy_to_host()
array.copy_to_device(host_array)

# Device info
cuda.is_available()
cuda.get_current_device()
cuda.gpus.count

# Synchronization
cuda.synchronize()
stream.synchronize()

# Events
event = cuda.event()
event.record()
event.synchronize()
cuda.event_elapsed_time(start, end)
```

---

## Conclusion

This handbook covered the essential concepts of CUDA programming in Python, from basic kernels to advanced optimization techniques. Key takeaways:

1. **Understanding the GPU architecture** is crucial for writing efficient code
2. **Memory management** often determines performance more than computation
3. **Thread organization** and synchronization prevent bugs
4. **Profiling and optimization** should be data-driven
5. **Real-world applications** benefit from massive parallelism

Continue learning by:
- Experimenting with the code examples
- Profiling your kernels with nvprof
- Reading NVIDIA's CUDA documentation
- Joining GPU programming communities
- Building real projects

Happy GPU programming!