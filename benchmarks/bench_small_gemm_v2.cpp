/// @file bench_small_gemm_v2.cpp
/// Performance benchmark for small shape GEMM with v2 optimizations.
///
/// Tests various (M, N, K) combinations with M, N <= 16.
/// Compares:
///   - v1 (baseline): original smallm implementation
///   - v2 (optimized): prefetch + 8x unrolling + software pipelining
///   - OpenBLAS: reference baseline

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/timer.h"
#include "dnnopt/arm_hwcaps.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// OpenBLAS cblas_sgemm
#ifdef HAVE_OPENBLAS
#include <cblas.h>
#endif

namespace {

constexpr int kRepeats = 100;
constexpr double kMinTimeSeconds = 0.1;

// Test shapes: (M, N, K) combinations for small matrices
struct TestShape {
    int M, N, K;
    const char* name;
};

// clang-format off
constexpr TestShape kTestShapes[] = {
    // M=1 cases (GEMV)
    {1, 1, 16, "1x1x16"},
    {1, 4, 32, "1x4x32"},
    {1, 8, 64, "1x8x64"},
    {1, 16, 128, "1x16x128"},
    {1, 32, 256, "1x32x256"},
    {1, 48, 512, "1x48x512"},
    {1, 64, 512, "1x64x512"},

    // N=1 cases (Matrix × Vector)
    {1, 1, 16, "1x1x16_mv"},
    {4, 1, 32, "4x1x32"},
    {8, 1, 64, "8x1x64"},
    {16, 1, 128, "16x1x128"},

    // M=N small cases
    {2, 2, 32, "2x2x32"},
    {4, 4, 64, "4x4x64"},
    {8, 8, 128, "8x8x128"},
    {12, 12, 256, "12x12x256"},

    // Tiny blocks
    {1, 1, 8, "1x1x8_tiny"},
    {2, 2, 8, "2x2x8_tiny"},
    {4, 4, 8, "4x4x8_tiny"},

    // Small M cases
    {2, 48, 256, "2x48x256"},
    {4, 48, 256, "4x48x256"},
    {8, 48, 256, "8x48x256"},
};
// clang-format on

/// Allocate aligned matrix (row-major: rows × lda elements).
std::vector<float> alloc_matrix(int rows, int cols, int lda) {
    std::vector<float> mat((size_t)rows * lda, 0.0f);
    return mat;
}

/// Initialize matrix with random values (row-major: mat[i*lda + j]).
void init_matrix(int rows, int cols, int lda, float* mat, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i * lda + j] = dist(rng);
        }
    }
}

/// Verify result against reference (row-major: C[i*ldc + j]).
bool verify_result(int M, int N, const float* C, const float* C_ref, int ldc,
                   float eps = 1e-4f) {
    float max_error = 0.0f;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float error = std::abs(C[i * ldc + j] - C_ref[i * ldc + j]);
            max_error = std::max(max_error, error);
        }
    }
    return max_error < eps;
}

/// Benchmark result structure.
struct BenchResult {
    double gflops;
    double time_ms;
    bool passed;
};

/// Benchmark a GEMM implementation.
BenchResult bench_gemm(int M, int N, int K,
                       void (*gemm_func)(int, int, int, float, const float*, int,
                                         const float*, int, float, float*, int),
                       const float* A, int lda,
                       const float* B, int ldb,
                       float* C, int ldc) {
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        gemm_func(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
    }

    // Timing loop
    int repeats = kRepeats;
    double total_time = 0.0;

    dnnopt::Timer timer;
    do {
        for (int i = 0; i < repeats; ++i) {
            timer.start();
            gemm_func(M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, ldc);
            timer.stop();
            total_time += timer.elapsed_sec();
        }

        if (total_time < kMinTimeSeconds) {
            repeats *= 2;
            total_time = 0.0;
        }
    } while (total_time < kMinTimeSeconds);

    double avg_time_ms = (total_time / repeats) * 1000.0;
    double gflops = (2.0 * M * N * K) / (total_time / repeats) / 1e9;

    BenchResult result;
    result.gflops = gflops;
    result.time_ms = avg_time_ms;
    result.passed = true;
    return result;
}

// ============================================================
// GEMM wrapper functions
// ============================================================

/// DNN-Opt v1 (baseline)
void sgemm_dnnopt_v1(int M, int N, int K, float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc) {
    // Use original smallm driver
    dnnopt::gemm_smallm_driver_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/// DNN-Opt v2 (optimized with prefetch)
void sgemm_dnnopt_v2(int M, int N, int K, float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc) {
    // Use v2 optimized driver
    dnnopt::gemm_smallm_driver_fp32_v2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/// OpenBLAS (reference)
#ifdef HAVE_OPENBLAS
void sgemm_openblas(int M, int N, int K, float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
#endif

}  // namespace

// ============================================================
// Main benchmark
// ============================================================

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("Small Shape GEMM Performance Benchmark - v2 Optimizations\n");
    printf("=================================================================\n\n");

    // Print hardware info
    auto& profile = dnnopt::detect_arm_hwcaps();
    printf("Hardware: %s\n", profile.cpu_name.c_str());
    printf("Cores: %u @ %u MHz\n", profile.num_cores, profile.freq_mhz);
    printf("SVE: %s, BF16: %s, I8MM: %s, SME: %s\n",
           profile.has(dnnopt::kSVE) ? "YES" : "NO",
           profile.has(dnnopt::kBF16) ? "YES" : "NO",
           profile.has(dnnopt::kI8MM) ? "YES" : "NO",
           profile.has(dnnopt::kSME) ? "YES" : "NO");
    printf("\n");

    // RNG for initialization
    std::mt19937 rng(42);

    // Header
    printf("%-12s %8s %12s %12s %12s %8s\n",
           "Shape", "FLOPs", "v1 (ms)", "v2 (ms)", "OpenBLAS", "Speedup");
    printf("%-12s %8s %12s %12s %12s %8s\n",
           "-------", "--------", "----------", "----------", "----------", "--------");

    int passed = 0, failed = 0;

    for (const auto& shape : kTestShapes) {
        int M = shape.M, N = shape.N, K = shape.K;
        int64_t flops = 2LL * M * N * K;

        // Allocate matrices
        int lda = K, ldb = N, ldc = N;
        auto A = alloc_matrix(M, K, lda);
        auto B = alloc_matrix(K, N, ldb);
        auto C_v1 = alloc_matrix(M, N, ldc);
        auto C_v2 = alloc_matrix(M, N, ldc);
        auto C_ref = alloc_matrix(M, N, ldc);

        // Initialize
        init_matrix(M, K, lda, A.data(), rng);
        init_matrix(K, N, ldb, B.data(), rng);

        // Compute reference with naive implementation (row-major, matching kernel convention)
        std::vector<float> C_naive(M * ldc, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C_naive[i * ldc + j] = sum;
            }
        }

        // Benchmark v1
        auto res_v1 = bench_gemm(M, N, K, sgemm_dnnopt_v1,
                                 A.data(), lda, B.data(), ldb, C_v1.data(), ldc);

        // Benchmark v2
        auto res_v2 = bench_gemm(M, N, K, sgemm_dnnopt_v2,
                                 A.data(), lda, B.data(), ldb, C_v2.data(), ldc);

#ifdef HAVE_OPENBLAS
        // Benchmark OpenBLAS
        auto res_ob = bench_gemm(M, N, K, sgemm_openblas,
                                 A.data(), lda, B.data(), ldb, C_ref.data(), ldc);
#else
        BenchResult res_ob = {0, 0, false};
#endif

        // Verify correctness
        bool v1_ok = verify_result(M, N, C_v1.data(), C_naive.data(), ldc);
        bool v2_ok = verify_result(M, N, C_v2.data(), C_naive.data(), ldc);

        // Speedup calculation
        double speedup = res_v1.time_ms / res_v2.time_ms;

        // Print results
        printf("%-12s %8.1fM %12.4f %12.4f %12.4f %8.2fx",
               shape.name, flops / 1e6,
               res_v1.time_ms, res_v2.time_ms, res_ob.time_ms, speedup);

        if (v1_ok && v2_ok) {
            printf(" ✓\n");
            ++passed;
        } else {
            printf(" ✗\n");
            ++failed;
        }
    }

    printf("\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
