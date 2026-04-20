/*******************************************************************************
 * Full Coverage Benchmark: Small-M + Irregular N
 * 
 * Comprehensive coverage of oneDNN weakness domain:
 * - M = 1, 2, 3, 4, 5, 6, 7 (all small batch sizes)
 * - N = all prime numbers 11-127 + non-power-of-2 6-250
 * - K = various sizes
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cmath>

namespace {

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

double bench_sgemm(int M, int N, int K, int warmup=5, int runs=30) {
    size_t a_sz = (size_t)M * K;
    size_t b_sz = (size_t)K * N;
    size_t c_sz = (size_t)M * N;
    
    if (a_sz == 0 || b_sz == 0 || c_sz == 0) return 0;
    
    float* A = (float*)aligned_alloc(64, a_sz * sizeof(float));
    float* B = (float*)aligned_alloc(64, b_sz * sizeof(float));
    float* C = (float*)aligned_alloc(64, c_sz * sizeof(float));
    
    fill_random(A, a_sz);
    fill_random(B, b_sz);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int w = 0; w < warmup; w++)
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < runs; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    
    free(A);
    free(B);
    free(C);
    
    return times[runs/2];  // median
}

bool is_prime(int n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int i = 3; i <= sqrt(n); i += 2)
        if (n % i == 0) return false;
    return true;
}

bool is_power_of_two(int n) {
    return n > 0 && (n & (n-1)) == 0;
}

}  // namespace

int main() {
    srand(42);
    
    printf("================================================================================\n");
    printf("  DNN-Opt Full Coverage Benchmark: Small-M + Irregular N\n");
    printf("================================================================================\n\n");
    
    // ================================================================
    // Part 1: M=1-7, all N combinations
    // ================================================================
    printf("=== Part 1: Small-M Full Coverage (M=1-7) ===\n\n");
    
    // N values to test: primes + non-powers + powers
    std::vector<int> n_values;
    
    // Prime numbers 11-127
    for (int n = 11; n <= 127; n++) {
        if (is_prime(n)) n_values.push_back(n);
    }
    
    // Non-power-of-2 numbers (skip primes already added)
    for (int n = 6; n <= 250; n++) {
        if (!is_power_of_two(n) && !is_prime(n)) {
            if (n <= 127 || n % 16 == 0 || n % 32 == 1)  // limit large values
                n_values.push_back(n);
        }
    }
    
    // Power-of-2 (baseline comparison)
    for (int n = 8; n <= 256; n *= 2) {
        n_values.push_back(n);
    }
    
    // Sort unique
    std::sort(n_values.begin(), n_values.end());
    n_values.erase(std::unique(n_values.begin(), n_values.end()), n_values.end());
    
    // M values
    std::vector<int> m_values = {1, 2, 3, 4, 5, 6, 7, 8};
    
    // K values (fixed for now)
    std::vector<int> k_values = {64, 128};
    
    printf("Testing %d M values x %d N values x %d K values = %d shapes\n",
           (int)m_values.size(), (int)n_values.size(), (int)k_values.size(),
           (int)m_values.size() * (int)n_values.size() * (int)k_values.size());
    
    printf("\nM=1 (GEMV) Results:\n");
    printf("%-8s %-8s %-8s %-12s %-10s %-12s\n", "N", "K", "Type", "Time(us)", "GFLOPS", "Prime?");
    printf("----------------------------------------------------------------\n");
    
    for (int K : k_values) {
        for (int N : n_values) {
            if (N > 4 * K) continue;  // skip extreme ratios
            
            double us = bench_sgemm(1, N, K, 3, 20);
            double gflops = 2.0 * 1 * N * K / (us * 1e3);
            
            const char* type = is_power_of_two(N) ? "pow2" : 
                               is_prime(N) ? "prime" : "irreg";
            const char* prime_mark = is_prime(N) ? "YES" : "";
            
            printf("%-8d %-8d %-8s %-12.1f %-10.2f %-12s\n", N, K, type, us, gflops, prime_mark);
        }
        printf("\n");
    }
    
    // ================================================================
    // Part 2: M=2-7 detailed coverage
    // ================================================================
    printf("\n=== Part 2: M=2-7 Full Coverage ===\n\n");
    
    for (int M = 2; M <= 7; M++) {
        printf("M=%d Results:\n", M);
        printf("%-8s %-8s %-12s %-10s\n", "N", "K", "Time(us)", "GFLOPS");
        printf("----------------------------------------\n");
        
        // Test all N up to 256
        for (int N = 8; N <= 256; N += 8) {
            for (int K : {64, 128}) {
                if (N > 4 * K) continue;
                
                double us = bench_sgemm(M, N, K, 3, 20);
                double gflops = 2.0 * M * N * K / (us * 1e3);
                printf("%-8d %-8d %-12.1f %-10.2f\n", N, K, us, gflops);
            }
        }
        
        // Prime N values for this M
        printf("\nPrime N values for M=%d:\n", M);
        printf("%-8s %-8s %-12s %-10s\n", "N", "K", "Time(us)", "GFLOPS");
        printf("----------------------------------------\n");
        
        for (int N : {13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127}) {
            for (int K : {64, 128}) {
                double us = bench_sgemm(M, N, K, 3, 20);
                double gflops = 2.0 * M * N * K / (us * 1e3);
                printf("%-8d %-8d %-12.1f %-10.2f\n", N, K, us, gflops);
            }
        }
        printf("\n");
    }
    
    // ================================================================
    // Part 3: M=8 (boundary case)
    // ================================================================
    printf("\n=== Part 3: M=8 (Boundary) ===\n\n");
    printf("%-8s %-8s %-12s %-10s %-12s\n", "N", "K", "Time(us)", "GFLOPS", "Type");
    printf("------------------------------------------------\n");
    
    for (int N : {8, 12, 16, 17, 24, 32, 37, 48, 64, 73, 128}) {
        for (int K : {64, 128}) {
            double us = bench_sgemm(8, N, K, 3, 20);
            double gflops = 2.0 * 8 * N * K / (us * 1e3);
            const char* type = is_prime(N) ? "prime" : 
                               is_power_of_two(N) ? "pow2" : "irreg";
            printf("%-8d %-8d %-12.1f %-10.2f %-12s\n", N, K, us, gflops, type);
        }
    }
    
    // ================================================================
    // Part 4: Irregular K (prime K values)
    // ================================================================
    printf("\n=== Part 4: Irregular K (Prime K) ===\n\n");
    printf("%-8s %-8s %-8s %-12s %-10s\n", "M", "N", "K", "Time(us)", "GFLOPS");
    printf("--------------------------------------------\n");
    
    for (int M : {4, 6, 8}) {
        for (int N : {32, 64, 128}) {
            for (int K : {13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71}) {
                if (K > N * 2) continue;  // reasonable ratio
                
                double us = bench_sgemm(M, N, K, 3, 20);
                double gflops = 2.0 * M * N * K / (us * 1e3);
                printf("%-8d %-8d %-8d %-12.1f %-10.2f\n", M, N, K, us, gflops);
            }
        }
    }
    
    // ================================================================
    // Part 5: Tiny shapes (N<=32, M<=8)
    // ================================================================
    printf("\n=== Part 5: Tiny Shapes (N<=32) ===\n\n");
    printf("%-8s %-8s %-8s %-12s %-10s\n", "M", "N", "K", "Time(us)", "GFLOPS");
    printf("--------------------------------------------\n");
    
    for (int M = 1; M <= 8; M++) {
        for (int N = 4; N <= 32; N += 4) {
            for (int K = N; K <= 64 && K <= N * 2; K += N) {
                double us = bench_sgemm(M, N, K, 3, 20);
                double gflops = 2.0 * M * N * K / (us * 1e3);
                printf("%-8d %-8d %-8d %-12.1f %-10.2f\n", M, N, K, us, gflops);
            }
        }
    }
    
    // ================================================================
    // Part 6: Tall-skinny (N small, M large)
    // ================================================================
    printf("\n=== Part 6: Tall-Skinny (N small, M large) ===\n\n");
    printf("%-8s %-8s %-8s %-12s %-10s\n", "M", "N", "K", "Time(us)", "GFLOPS");
    printf("--------------------------------------------\n");
    
    for (int M : {32, 64, 128, 256}) {
        for (int N : {1, 2, 4, 8, 12, 16, 24, 32}) {
            int K = M;  // K = M for typical tall-skinny
            double us = bench_sgemm(M, N, K, 3, 20);
            double gflops = 2.0 * M * N * K / (us * 1e3);
            printf("%-8d %-8d %-8d %-12.1f %-10.2f\n", M, N, K, us, gflops);
        }
    }
    
    // ================================================================
    // Part 7: Non-power-of-2 boundaries
    // ================================================================
    printf("\n=== Part 7: Power-of-2 Boundaries ===\n\n");
    printf("%-8s %-8s %-8s %-12s %-10s %-12s\n", "M", "N", "K", "Time(us)", "GFLOPS", "N_type");
    printf("------------------------------------------------\n");
    
    for (int M : {4, 6, 8}) {
        // Around key boundaries
        for (int base : {32, 64, 128}) {
            for (int offset : {-1, 0, 1}) {
                int N = base + offset;
                int K = 128;
                double us = bench_sgemm(M, N, K, 3, 20);
                double gflops = 2.0 * M * N * K / (us * 1e3);
                const char* n_type = (offset == 0) ? "pow2" : 
                                    (offset == -1) ? "pow2-1" : "pow2+1";
                printf("%-8d %-8d %-8d %-12.1f %-10.2f %-12s\n", M, N, K, us, gflops, n_type);
            }
        }
    }
    
    printf("\n================================================================================\n");
    printf("  Benchmark Complete\n");
    printf("================================================================================\n");
    
    return 0;
}
