/// @file bench_onednn_vs_autotune.cpp
/// Compare oneDNN native vs DNN-Opt autotune for irregular shapes.

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/timer.h"
#include "dnnopt/aligned_alloc.h"

// oneDNN header
#include <dnnl.h>

#include <cstdio>
#include <cstring>
#include <algorithm>

using namespace dnnopt;

struct TestShape {
    int M, N, K;
    const char* category;
};

static double bench_onednn(int M, int N, int K) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);

    // Warmup
    for (int w = 0; w < 3; ++w) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        dnnl_sgemm('N', 'N', M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    }

    // Timed runs
    double times[7];
    Timer timer;
    for (int t = 0; t < 7; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        dnnl_sgemm('N', 'N', M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    std::sort(times, times + 7);
    return times[3];
}

static double bench_dnnopt(int M, int N, int K, bool autotune) {
    auto A = aligned_array<float>(M * K);
    auto B = aligned_array<float>(K * N);
    auto C = aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);

    // Clear caches
    get_tile_cache().clear();
    get_blocking_cache().clear();
    get_gemm_shape_cache().clear();

    if (autotune) {
        setenv("DNNOPT_AUTOTUNE", "1", 1);
    } else {
        unsetenv("DNNOPT_AUTOTUNE");
    }

    // Warmup
    for (int w = 0; w < 3; ++w) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    }

    // Timed runs
    double times[7];
    Timer timer;
    for (int t = 0; t < 7; ++t) {
        std::memset(C.get(), 0, M * N * sizeof(float));
        timer.start();
        gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
        timer.stop();
        times[t] = timer.elapsed_us();
    }

    std::sort(times, times + 7);
    return times[3];
}

int main() {
    printf("=== oneDNN vs DNN-Opt Autotune: Irregular Shapes ===\n\n");

    // Irregular shapes
    TestShape shapes[] = {
        // Prime numbers
        {127, 127, 127, "Prime"},
        {257, 257, 257, "Prime"},
        {511, 511, 511, "Prime"},
        {1021, 1021, 1021, "Prime"},
        {257, 511, 1023, "Prime-mix"},
        {127, 127, 1024, "Prime-MNK"},

        // Odd M (small irregular)
        {9, 1024, 1024, "Odd-M"},
        {13, 1024, 1024, "Odd-M"},
        {15, 1024, 1024, "Odd-M"},
        {17, 1024, 1024, "Odd-M"},
        {25, 1024, 1024, "Odd-M"},
        {35, 1024, 1024, "Odd-M"},
        {49, 1024, 1024, "Odd-M"},

        // 3^n patterns
        {27, 1024, 1024, "3^n"},
        {81, 1024, 1024, "3^n"},
        {243, 1024, 1024, "3^n"},
        {729, 1024, 1024, "3^n"},
        {2187, 1024, 1024, "3^n"},

        // 5^n patterns
        {25, 1024, 1024, "5^n"},
        {125, 1024, 1024, "5^n"},
        {625, 1024, 1024, "5^n"},
        {3125, 1024, 1024, "5^n"},

        // 7^n patterns
        {49, 1024, 1024, "7^n"},
        {343, 1024, 1024, "7^n"},
        {2401, 1024, 1024, "7^n"},

        // Neural network sizes
        {768, 768, 1024, "NN-BERT"},
        {768, 3072, 768, "NN-BERT-FFN"},
        {3072, 768, 768, "NN-BERT"},
        {11008, 4096, 4096, "NN-LLaMA"},
        {14336, 5120, 5120, "NN-LLaMA2"},
        {12288, 12288, 12288, "NN-LLaMA-large"},

        // Boundary cases
        {3, 1024, 1024, "Boundary"},
        {5, 1024, 1024, "Boundary"},
        {7, 1024, 1024, "Boundary"},
        {10, 1024, 1024, "Boundary"},
        {14, 1024, 1024, "Boundary"},
        {16, 1024, 1024, "Boundary"},

        // Mixed irregular
        {123, 456, 789, "Mixed"},
        {234, 567, 890, "Mixed"},
        {345, 678, 901, "Mixed"},
        {456, 789, 1023, "Mixed"},
        {567, 890, 1234, "Mixed"},
        {789, 1023, 1678, "Mixed"},
        {901, 1234, 1890, "Mixed"},
        {1023, 1456, 2123, "Mixed"},

        // Regular (baseline)
        {64, 64, 64, "Regular"},
        {128, 128, 128, "Regular"},
        {256, 256, 256, "Regular"},
        {512, 512, 512, "Regular"},
        {1024, 1024, 1024, "Regular"},
        {2048, 2048, 2048, "Regular"},
        {4096, 4096, 4096, "Regular"},

        // Tall-skinny / Short-wide
        {4096, 64, 4096, "TallSkinny"},
        {64, 4096, 4096, "ShortWide"},
        {2048, 128, 2048, "TallSkinny"},
        {128, 2048, 2048, "ShortWide"},
        {4096, 128, 4096, "TallSkinny"},
        {128, 4096, 4096, "ShortWide"},
        {8192, 64, 4096, "TallSkinny"},
        {64, 8192, 4096, "ShortWide"},
    };

    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    printf("Shape            oneDNN     dnnopt(H)   dnnopt(A)   Ratio_H    Ratio_A    Category\n");
    printf("==========================================================================================\n");

    double total_onednn = 0, total_dnnopt_h = 0, total_dnnopt_a = 0;
    int n_tested = 0;

    for (int i = 0; i < n_shapes; ++i) {
        int M = shapes[i].M, N = shapes[i].N, K = shapes[i].K;
        int64_t vol = (int64_t)M * N * K;

        // Skip very large shapes
        if (vol > 10LL * 1024 * 1024 * 1024) continue;

        double us_onednn = bench_onednn(M, N, K);
        double us_dnnopt_h = bench_dnnopt(M, N, K, false);
        double us_dnnopt_a = bench_dnnopt(M, N, K, true);

        double gflops_onednn = 2.0 * M * N * K / (us_onednn * 1000.0);
        double gflops_dnnopt_h = 2.0 * M * N * K / (us_dnnopt_h * 1000.0);
        double gflops_dnnopt_a = 2.0 * M * N * K / (us_dnnopt_a * 1000.0);

        double ratio_h = gflops_dnnopt_h / gflops_onednn;
        double ratio_a = gflops_dnnopt_a / gflops_onednn;

        // Mark significant differences
        const char* marker = "";
        if (ratio_h > 1.05 || ratio_a > 1.05) marker = "★";
        else if (ratio_h < 0.95 || ratio_a < 0.95) marker = "⚠";

        printf("M=%4d N=%4d K=%4d  %7.2f    %7.2f    %7.2f    %.4f    %.4f %s %s\n",
               M, N, K, gflops_onednn, gflops_dnnopt_h, gflops_dnnopt_a,
               ratio_h, ratio_a, marker, shapes[i].category);

        total_onednn += gflops_onednn;
        total_dnnopt_h += gflops_dnnopt_h;
        total_dnnopt_a += gflops_dnnopt_a;
        n_tested++;
    }

    printf("==========================================================================================\n");
    printf("Average:         %7.2f    %7.2f    %7.2f    %.4f    %.4f\n",
           total_onednn/n_tested, total_dnnopt_h/n_tested, total_dnnopt_a/n_tested,
           total_dnnopt_h/total_onednn, total_dnnopt_a/total_onednn);

    printf("\n=== Summary ===\n");
    printf("oneDNN → dnnopt heuristic: %.2fx (%.1f%%)\n",
           total_dnnopt_h/total_onednn,
           (total_dnnopt_h/total_onednn - 1) * 100);
    printf("oneDNN → dnnopt autotune:  %.2fx (%.1f%%)\n",
           total_dnnopt_a/total_onednn,
           (total_dnnopt_a/total_onednn - 1) * 100);
    printf("heuristic → autotune:     %.1f%%\n",
           (total_dnnopt_a/total_dnnopt_h - 1) * 100);

    return 0;
}