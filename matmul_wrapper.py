# matmul_wrapper.py
import numpy as np
import ctypes
import os
from time import time

# Load the compiled library
lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'matmul.so')
_lib = ctypes.CDLL(lib_path)

# Define function prototypes
_lib.matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

_lib.matmul_transposed.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

_lib.matmul_tiled.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

def matmul(A, B, implementation='avx'):
    """
    Perform optimized matrix multiplication C = A * B
    
    Parameters:
    -----------
    A : numpy.ndarray
        First matrix with shape (m, k)
    B : numpy.ndarray
        Second matrix with shape (k, n)
    implementation : str
        Which implementation to use: 'avx', 'transposed', or 'tiled'
        
    Returns:
    --------
    C : numpy.ndarray
        Result matrix with shape (m, n)
    """
    # Ensure matrices are float32 and C-contiguous
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    
    # Get matrix dimensions
    m, k = A.shape
    k2, n = B.shape
    
    # Verify dimensions
    if k != k2:
        raise ValueError(f"Incompatible matrix dimensions: {A.shape} and {B.shape}")
    
    # Create output matrix
    C = np.zeros((m, n), dtype=np.float32)
    
    # Call the appropriate C function based on implementation choice
    if implementation == 'avx':
        _lib.matmul(A, B, C, m, k, n)
    elif implementation == 'transposed':
        _lib.matmul_transposed(A, B, C, m, k, n)
    elif implementation == 'tiled':
        _lib.matmul_tiled(A, B, C, m, k, n)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")
    
    return C

def benchmark(sizes=[(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)], 
              implementations=['avx', 'transposed', 'tiled', 'numpy']):
    """
    Benchmark different matrix multiplication implementations
    
    Parameters:
    -----------
    sizes : list of tuples
        List of (m, k, n) matrix sizes to benchmark
    implementations : list of str
        List of implementations to benchmark
        
    Returns:
    --------
    results : dict
        Dictionary of benchmark results
    """
    results = {}
    
    for size in sizes:
        m, k, n = size
        print(f"Benchmarking size {m}x{k} * {k}x{n}")
        
        # Create random matrices
        A = np.random.random((m, k)).astype(np.float32)
        B = np.random.random((k, n)).astype(np.float32)
        
        # Compute reference result using numpy
        np_C = np.matmul(A, B)
        
        for impl in implementations:
            if impl == 'numpy':
                # Benchmark numpy implementation
                start = time()
                C = np.matmul(A, B)
                elapsed = time() - start
            else:
                # Benchmark our custom implementation
                start = time()
                C = matmul(A, B, implementation=impl)
                elapsed = time() - start
                
                # Verify correctness
                if not np.allclose(C, np_C, rtol=1e-5, atol=1e-5):
                    print(f"Warning: {impl} implementation may be incorrect!")
                    print(f"Max difference: {np.max(np.abs(C - np_C))}")
            
            print(f"  {impl}: {elapsed:.4f} seconds")
            results[(m, k, n, impl)] = elapsed
    
    return results

if __name__ == "__main__":
    # Example usage
    A = np.random.random((1000, 1000)).astype(np.float32)
    B = np.random.random((1000, 1000)).astype(np.float32)
    
    print("Running benchmark...")
    results = benchmark()
    
    # Find the best implementation for each size
    for size in [(100, 100, 100), (500, 500, 500), (1000, 1000, 1000)]:
        best_impl = min([(impl, results[(size[0], size[1], size[2], impl)]) 
                         for impl in ['avx', 'transposed', 'tiled', 'numpy']], 
                        key=lambda x: x[1])
        print(f"Best implementation for size {size}: {best_impl[0]} ({best_impl[1]:.4f} seconds)")
