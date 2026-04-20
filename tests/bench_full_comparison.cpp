/*******************************************************************************
 * Full Performance Comparison: Small Shapes + Irregular Shapes
 * 
 * Three configurations tested:
 * 1. TensorFlow Eigen (via tf.matmul, TF_DISABLE_ONEDNN=1)
 * 2. TensorFlow oneDNN (via tf.matmul, default)
 * 3. oneDNN + DNN-Opt (via dnnl_sgemm)
 *
 * Shapes tested:
 * - M=1 (GEMV, batch-1 inference)
 * - M=2-7 (small batch)
 * - Irregular N (prime: 13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97)
 * - Irregular K (prime)
 * - Non-power-of-2 (N=15,33,49,65,97)
 * - Tiny shapes (N<=32)
 * - Tall-skinny (N small)
 * - Short-wide (M small, N large)
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>

namespace {

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

double bench_sgemm(int M, int N, int K) {
    size_t a_sz = M * K;
    size_t b_sz = K * N;
    size_t c_sz = M * N;
    
    float* A = (float*)aligned_alloc(64, a_sz * sizeof(float));
    float* B = (float*)aligned_alloc(64, b_sz * sizeof(float));
    float* C = (float*)aligned_alloc(64, c_sz * sizeof(float));
    
    fill_random(A, a_sz);
    fill_random(B, b_sz);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int w = 0; w < 10; w++)
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // Benchmark
    std::vector<double> times;
    for (int i = 0; i < 50; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    
    free(A);
    free(B);
    free(C);
    
    return times[25];  // median
}

void print_section(const char* title) {
    printf("\n=== %s ===\n", title);
    printf("%-15s %-6s %-6s %-6s %-12s %-10s\n", "Shape", "M", "N", "K", "Time(us)", "GFLOPS");
    printf("--------------------------------------------------------\n");
}

void test_shape(int M, int N, int K, const char* label = nullptr) {
    double us = bench_sgemm(M, N, K);
    double gflops = 2.0 * M * N * K / (us * 1e3);
    
    char shape_str[32];
    if (label) {
        snprintf(shape_str, sizeof(shape_str), "%s", label);
    } else {
        snprintf(shape_str, sizeof(shape_str), "[%d,%d,%d]", M, N, K);
    }
    
    printf("%-15s %-6d %-6d %-6d %-12.1f %-10.2f\n", shape_str, M, N, K, us, gflops);
}

}  // namespace

int main() {
    srand(42);
    
    printf("================================================================\n");
    printf("  oneDNN + DNN-Opt: Full Shape Performance Benchmark\n");
    printf("  Testing small-M and irregular shapes (oneDNN weakness domain)\n");
    printf("================================================================\n");
    
    // ================================================================
    // M=1: GEMV (batch-1 inference)
    // ================================================================
    print_section("M=1: GEMV (batch-1 inference)");
    
    // Small N
    test_shape(1, 8, 8);
    test_shape(1, 16, 16);
    test_shape(1, 32, 32);
    test_shape(1, 64, 64);
    test_shape(1, 128, 128);
    
    // Medium N
    test_shape(1, 256, 256);
    test_shape(1, 512, 512);
    test_shape(1, 768, 768);
    
    // Large N (LLM-like)
    test_shape(1, 1024, 1024);
    test_shape(1, 2048, 2048);
    test_shape(1, 4096, 4096);
    test_shape(1, 4096, 4096, "M1_N4096");
    
    // Embedding shapes
    test_shape(1, 256, 1024);
    test_shape(1, 512, 1024);
    test_shape(1, 1024, 4096);
    
    // ================================================================
    // M=2-7: Small batch inference
    // ================================================================
    print_section("M=2-7: Small batch inference");
    
    // M=2
    test_shape(2, 32, 32);
    test_shape(2, 64, 64);
    test_shape(2, 128, 128);
    test_shape(2, 256, 256);
    
    // M=3
    test_shape(3, 32, 32);
    test_shape(3, 64, 64);
    test_shape(3, 128, 128);
    test_shape(3, 256, 256);
    
    // M=4
    test_shape(4, 32, 32);
    test_shape(4, 64, 64);
    test_shape(4, 128, 128);
    test_shape(4, 256, 256);
    test_shape(4, 512, 512);
    test_shape(4, 768, 768);
    test_shape(4, 1024, 1024);
    
    // M=5
    test_shape(5, 32, 32);
    test_shape(5, 64, 64);
    test_shape(5, 128, 128);
    test_shape(5, 256, 256);
    
    // M=6
    test_shape(6, 32, 32);
    test_shape(6, 64, 64);
    test_shape(6, 128, 128);
    test_shape(6, 256, 256);
    test_shape(6, 512, 512);
    test_shape(6, 768, 768);
    
    // M=7
    test_shape(7, 32, 32);
    test_shape(7, 64, 64);
    test_shape(7, 128, 128);
    test_shape(7, 256, 256);
    
    // ================================================================
    // Irregular N (prime numbers)
    // ================================================================
    print_section("Irregular N (prime numbers)");
    
    test_shape(4, 13, 64, "prime_N13");
    test_shape(4, 17, 64, "prime_N17");
    test_shape(4, 19, 64, "prime_N19");
    test_shape(4, 23, 64, "prime_N23");
    test_shape(4, 29, 64, "prime_N29");
    test_shape(4, 31, 64, "prime_N31");
    test_shape(4, 37, 64, "prime_N37");
    test_shape(4, 41, 64, "prime_N41");
    test_shape(4, 43, 64, "prime_N43");
    test_shape(4, 47, 64, "prime_N47");
    test_shape(4, 53, 128, "prime_N53");
    test_shape(4, 59, 128, "prime_N59");
    test_shape(4, 61, 128, "prime_N61");
    test_shape(4, 67, 128, "prime_N67");
    test_shape(4, 71, 128, "prime_N71");
    test_shape(4, 73, 128, "prime_N73");
    test_shape(4, 79, 128, "prime_N79");
    test_shape(4, 83, 128, "prime_N83");
    test_shape(4, 89, 128, "prime_N89");
    test_shape(4, 97, 128, "prime_N97");
    test_shape(4, 101, 128, "prime_N101");
    test_shape(4, 103, 128, "prime_N103");
    test_shape(4, 107, 128, "prime_N107");
    
    // ================================================================
    // Irregular K (prime numbers)
    // ================================================================
    print_section("Irregular K (prime numbers)");
    
    test_shape(4, 64, 13, "prime_K13");
    test_shape(4, 64, 17, "prime_K17");
    test_shape(4, 64, 19, "prime_K19");
    test_shape(4, 64, 23, "prime_K23");
    test_shape(4, 64, 29, "prime_K29");
    test_shape(4, 64, 31, "prime_K31");
    test_shape(4, 64, 37, "prime_K37");
    test_shape(4, 64, 41, "prime_K41");
    test_shape(4, 64, 43, "prime_K43");
    test_shape(4, 64, 47, "prime_K47");
    test_shape(4, 128, 53, "prime_K53");
    test_shape(4, 128, 59, "prime_K59");
    test_shape(4, 128, 61, "prime_K61");
    test_shape(4, 128, 67, "prime_K67");
    test_shape(4, 128, 71, "prime_K71");
    
    // ================================================================
    // Non-power-of-2 boundaries
    // ================================================================
    print_section("Non-power-of-2 boundaries");
    
    // Around N=32
    test_shape(4, 31, 64);
    test_shape(4, 32, 64);
    test_shape(4, 33, 64);
    
    // Around N=64
    test_shape(4, 63, 64);
    test_shape(4, 64, 64);
    test_shape(4, 65, 64);
    
    // Around N=128
    test_shape(4, 127, 128);
    test_shape(4, 128, 128);
    test_shape(4, 129, 128);
    
    // Around N=256
    test_shape(4, 255, 256);
    test_shape(4, 256, 256);
    test_shape(4, 257, 256);
    
    // ================================================================
    // Tiny shapes (N<=32)
    // ================================================================
    print_section("Tiny shapes (N<=32)");
    
    test_shape(1, 4, 4);
    test_shape(1, 8, 8);
    test_shape(1, 12, 12);
    test_shape(1, 16, 16);
    test_shape(1, 24, 24);
    test_shape(1, 32, 32);
    
    test_shape(4, 4, 4);
    test_shape(4, 8, 8);
    test_shape(4, 12, 12);
    test_shape(4, 16, 16);
    test_shape(4, 24, 24);
    test_shape(4, 32, 32);
    
    test_shape(8, 4, 4);
    test_shape(8, 8, 8);
    test_shape(8, 12, 12);
    test_shape(8, 16, 16);
    test_shape(8, 24, 24);
    test_shape(8, 32, 32);
    
    // ================================================================
    // Tall-skinny (N small, M large)
    // ================================================================
    print_section("Tall-skinny (N small, M large)");
    
    test_shape(64, 1, 128);
    test_shape(64, 2, 128);
    test_shape(64, 4, 128);
    test_shape(64, 8, 128);
    test_shape(64, 12, 128);
    test_shape(64, 16, 128);
    test_shape(64, 32, 128);
    
    test_shape(128, 1, 128);
    test_shape(128, 2, 128);
    test_shape(128, 4, 128);
    test_shape(128, 8, 128);
    
    test_shape(256, 1, 256);
    test_shape(256, 2, 256);
    test_shape(256, 4, 256);
    
    // ================================================================
    // Short-wide (M small, N large)
    // ================================================================
    print_section("Short-wide (M small, N large)");
    
    test_shape(2, 128, 128);
    test_shape(2, 256, 128);
    test_shape(2, 512, 128);
    test_shape(2, 1024, 128);
    
    test_shape(4, 128, 128);
    test_shape(4, 256, 128);
    test_shape(4, 512, 128);
    test_shape(4, 1024, 128);
    
    test_shape(6, 128, 128);
    test_shape(6, 256, 128);
    test_shape(6, 512, 128);
    
    // ================================================================
    // BERT-like shapes
    // ================================================================
    print_section("BERT-like shapes");
    
    test_shape(1, 768, 768, "bert_M1");
    test_shape(4, 768, 768, "bert_M4");
    test_shape(6, 768, 768, "bert_M6");
    test_shape(8, 768, 768, "bert_M8");
    
    test_shape(4, 3072, 768, "bert_ffn1");
    test_shape(4, 768, 3072, "bert_ffn2");
    
    // ================================================================
    // LLM-like shapes
    // ================================================================
    print_section("LLM-like shapes");
    
    test_shape(1, 4096, 4096, "llm_M1");
    test_shape(4, 4096, 4096, "llm_M4");
    test_shape(8, 4096, 4096, "llm_M8");
    test_shape(8, 512, 512, "llm_qkv");
    test_shape(8, 1376, 512, "llm_ffn1");
    test_shape(8, 512, 1376, "llm_ffn2");
    
    // ================================================================
    // Control: Large M (oneDNN strength)
    // ================================================================
    print_section("Large M (oneDNN strength - control)");
    
    test_shape(16, 256, 256);
    test_shape(32, 256, 256);
    test_shape(32, 512, 512);
    test_shape(64, 512, 512);
    test_shape(128, 256, 256);
    test_shape(128, 512, 512);
    
    printf("\n================================================================\n");
    printf("  End of benchmark\n");
    printf("================================================================\n");
    
    return 0;
}
