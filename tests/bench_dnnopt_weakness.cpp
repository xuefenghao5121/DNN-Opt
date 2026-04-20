/*******************************************************************************
 * DNN-Opt Weakness Domain Benchmark
 * 
 * Test shapes where oneDNN is slow and DNN-Opt provides optimization:
 * - M=1 (batch-1 inference, GEMV)
 * - M=2-7 (small batch)
 * - Irregular N (prime numbers)
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>

namespace {

struct Shape {
    int M, N, K;
    const char* label;
};

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

double bench_sgemm(int M, int N, int K, const float* A, const float* B, float* C) {
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int w = 0; w < 10; w++)
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < 100; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[50];
}

}  // namespace

int main() {
    srand(42);
    
    // Shapes covering oneDNN weakness domain
    Shape shapes[] = {
        // M=1: batch-1 inference (GEMV) - CRITICAL weakness
        {1, 128, 128, "M1_N128"},
        {1, 256, 512, "M1_N256"},
        {1, 1024, 1024, "M1_N1024"},
        {1, 4096, 4096, "M1_N4096"},
        {1, 1024, 4096, "M1_embedding"},
        
        // M=2-7: small batch - oneDNN weakness
        {2, 64, 64, "M2_N64"},
        {2, 128, 128, "M2_N128"},
        {3, 64, 64, "M3_N64"},
        {3, 128, 128, "M3_N128"},
        {4, 64, 64, "M4_N64"},
        {4, 128, 128, "M4_N128"},
        {5, 64, 64, "M5_N64"},
        {5, 128, 128, "M5_N128"},
        {6, 64, 64, "M6_N64"},
        {6, 128, 128, "M6_N128"},
        {7, 64, 64, "M7_N64"},
        {7, 128, 128, "M7_N128"},
        
        // Irregular N (prime numbers) - oneDNN weakness
        {8, 17, 64, "M8_N17_prime"},
        {8, 37, 64, "M8_N37_prime"},
        {8, 53, 128, "M8_N53_prime"},
        {8, 97, 128, "M8_N97_prime"},
        
        // Control: large M (oneDNN strength)
        {32, 512, 512, "M32_N512_strength"},
        {64, 512, 512, "M64_N512_strength"},
    };
    
    int n = sizeof(shapes) / sizeof(shapes[0]);
    
    printf("============================================================\n");
    printf("  DNN-Opt Weakness Domain Benchmark\n");
    printf("  Testing M=1-7 and irregular N (oneDNN slow)\n");
    printf("============================================================\n\n");
    
    printf("%-18s %-10s %-10s %-10s %-10s %-10s\n", 
           "Shape", "M", "N", "K", "Time(us)", "GFLOPS");
    printf("------------------------------------------------------------\n");
    
    double small_m_total = 0;
    double large_m_total = 0;
    int small_m_count = 0;
    int large_m_count = 0;
    
    for (int i = 0; i < n; i++) {
        const auto& s = shapes[i];
        size_t a_sz = s.M * s.K;
        size_t b_sz = s.K * s.N;
        size_t c_sz = s.M * s.N;
        
        float* A = (float*)aligned_alloc(64, a_sz * sizeof(float));
        float* B = (float*)aligned_alloc(64, b_sz * sizeof(float));
        float* C = (float*)aligned_alloc(64, c_sz * sizeof(float));
        
        fill_random(A, a_sz);
        fill_random(B, b_sz);
        
        double us = bench_sgemm(s.M, s.N, s.K, A, B, C);
        double gflops = 2.0 * s.M * s.N * s.K / (us * 1e3);
        
        printf("%-18s %-10d %-10d %-10d %-10.1f %-10.2f\n",
               s.label, s.M, s.N, s.K, us, gflops);
        
        if (s.M < 8) {
            small_m_total += gflops;
            small_m_count++;
        } else {
            large_m_total += gflops;
            large_m_count++;
        }
        
        free(A);
        free(B);
        free(C);
    }
    
    printf("\n============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n\n");
    
    printf("Small-M (M<8, oneDNN weakness): Avg %.2f GFLOPS (%d shapes)\n",
           small_m_total / small_m_count, small_m_count);
    printf("Large-M (M>=8, oneDNN strength): Avg %.2f GFLOPS (%d shapes)\n",
           large_m_total / large_m_count, large_m_count);
    printf("\nDNN-Opt optimizes Small-M shapes where oneDNN is slow.\n");
    
    return 0;
}
