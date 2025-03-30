// matmul.cpp
// Optimized matrix multiplication for 12th Gen Intel Core i3-1215U
// Compile with: g++ -O3 -march=alderlake -mavx2 -ffast-math -fopenmp -shared -fPIC matmul.cpp -o matmul.so

#include <immintrin.h> // For AVX2 intrinsics
#include <omp.h>       // For OpenMP parallelization
#include <cstring>     // For memset
#include <algorithm>   //for std::min

// Function to perform optimized matrix multiplication C = A * B
// A: m x k matrix
// B: k x n matrix
// C: m x n matrix (result)
extern "C" void matmul(const float* A, const float* B, float* C, int m, int k, int n) {
    // Clear the output matrix first
    memset(C, 0, m * n * sizeof(float));
    
    // Set the number of threads to match your CPU (8 threads for i3-1215U)
    omp_set_num_threads(8);
    
    // Use OpenMP to parallelize the outer loop
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        // For each row of A
        for (int j = 0; j < n; j += 8) {
            // Process 8 columns of B at once when possible (using AVX2)
            if (j + 8 <= n) {
                for (int l = 0; l < k; l++) {
                    // Broadcast single element from A
                    __m256 a_val = _mm256_set1_ps(A[i * k + l]);
                    
                    // Load 8 elements from B
                    __m256 b_vals = _mm256_loadu_ps(&B[l * n + j]);
                    
                    // Load current result
                    __m256 c_vals = _mm256_loadu_ps(&C[i * n + j]);
                    
                    // Multiply and add (C += A * B)
                    c_vals = _mm256_fmadd_ps(a_val, b_vals, c_vals);
                    
                    // Store back the result
                    _mm256_storeu_ps(&C[i * n + j], c_vals);
                }
            } else {
                // Handle remaining columns (less than 8) with scalar operations
                for (int jj = j; jj < n; jj++) {
                    for (int l = 0; l < k; l++) {
                        C[i * n + jj] += A[i * k + l] * B[l * n + jj];
                    }
                }
                break; // Exit the j loop since we've handled the remainder
            }
        }
    }
}

// Function to transpose matrix B for better memory access patterns
// This can improve performance for certain matrix sizes
extern "C" void transpose_matrix(const float* src, float* dst, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// Alternative implementation using matrix B transposition for better cache utilization
extern "C" void matmul_transposed(const float* A, const float* B, float* C, int m, int k, int n) {
    // Create a transposed copy of B for better cache locality
    float* B_transposed = new float[k * n];
    transpose_matrix(B, B_transposed, k, n);
    
    // Clear the output matrix
    memset(C, 0, m * n * sizeof(float));
    
    // Set the number of threads to match your CPU
    omp_set_num_threads(8);
    
    // Use OpenMP to parallelize the computation
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            __m256 sum = _mm256_setzero_ps();
            
            // Process 8 elements at a time
            for (int l = 0; l < k - 7; l += 8) {
                __m256 a_vals = _mm256_loadu_ps(&A[i * k + l]);
                __m256 b_vals = _mm256_loadu_ps(&B_transposed[j * k + l]);
                
                // Use FMA instruction for better performance
                sum = _mm256_fmadd_ps(a_vals, b_vals, sum);
            }
            
            // Horizontal sum of the 8 partial results
            float temp[8];
            _mm256_storeu_ps(temp, sum);
            float result = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
            
            // Handle remaining elements
            for (int l = (k / 8) * 8; l < k; l++) {
                result += A[i * k + l] * B_transposed[j * k + l];
            }
            
            C[i * n + j] = result;
        }
    }
    
    delete[] B_transposed;
}

// Tiled matrix multiplication for better cache utilization
extern "C" void matmul_tiled(const float* A, const float* B, float* C, int m, int k, int n) {
    // Clear the output matrix
    memset(C, 0, m * n * sizeof(float));
    
    // Define tile sizes based on cache size
    // L1 data cache on i3-1215U is 48KB per core
    const int TILE_SIZE_M = 32;
    const int TILE_SIZE_N = 32;
    const int TILE_SIZE_K = 32;
    
    // Set the number of threads
    omp_set_num_threads(8);
    
    // Use OpenMP to parallelize the tiled computation
    #pragma omp parallel for collapse(2)
    for (int i0 = 0; i0 < m; i0 += TILE_SIZE_M) {
        for (int j0 = 0; j0 < n; j0 += TILE_SIZE_N) {
            for (int k0 = 0; k0 < k; k0 += TILE_SIZE_K) {
                // Determine actual tile sizes (handle boundaries)
                int i_end = std::min(i0 + TILE_SIZE_M, m);
                int j_end = std::min(j0 + TILE_SIZE_N, n);
                int k_end = std::min(k0 + TILE_SIZE_K, k);
                
                // Process current tile
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j++) {
                        float sum = 0.0f;
                        for (int l = k0; l < k_end; l++) {
                            sum += A[i * k + l] * B[l * n + j];
                        }
                        C[i * n + j] += sum;
                    }
                }
            }
        }
    }
}
