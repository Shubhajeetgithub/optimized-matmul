# Optimized Matrix Multiplication for Intel i3-1215U

This repository contains optimized matrix multiplication code specifically tuned for 12th Gen Intel Core i3-1215U processors. It implements AVX2 vectorization and OpenMP parallelization for maximum performance.

## Features

- Multiple optimized implementations:
  - AVX2 vectorized with OpenMP parallelization
  - Transposed matrix optimization for better cache behavior
  - Tiled matrix multiplication for improved cache utilization
- Python wrapper with NumPy integration
- Benchmarking utilities to find the optimal implementation for your use case

## Requirements

- C++ compiler with AVX2 and OpenMP support (gcc/g++ recommended)
- Python 3.6+
- NumPy

## Installation

1. Clone this repository:
git clone https://github.com/Shubhajeetgithub/optimized-matmul.git
cd optimized-matmul

2. Compile the C++ code:
g++ -O3 -march=alderlake -mavx2 -ffast-math -fopenmp -shared -fPIC matmul.cpp -o matmul.so

## Usage

```python
import numpy as np
from matmul_wrapper import matmul, benchmark

# Create example matrices
A = np.random.random((1000, 800)).astype(np.float32)
B = np.random.random((800, 1200)).astype(np.float32)

# Multiply using our optimized function (default implementation is 'avx')
C = matmul(A, B)

# Try different implementations
C_transposed = matmul(A, B, implementation='transposed')
C_tiled = matmul(A, B, implementation='tiled')

# Run benchmarks to find the fastest implementation
benchmark()


