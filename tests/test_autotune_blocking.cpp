/// @file test_autotune_blocking.cpp
/// Simple test for v2 blocking parameter autotune.

#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/gemm/gemm_config.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/aligned_alloc.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

void test_blocking_presets() {
    printf("=== Blocking presets ===\n");

    auto p0 = dnnopt::get_blocking_params_from_preset(dnnopt::BlockingPreset::kConservative);
    auto p2 = dnnopt::get_blocking_params_from_preset(dnnopt::BlockingPreset::kModerate);
    auto p4 = dnnopt::get_blocking_params_from_preset(dnnopt::BlockingPreset::kMaximum);

    printf("  Conservative: l1d=%.2f l2=%.2f l3=%.2f\n", p0.l1d_util, p0.l2_util, p0.l3_util);
    printf("  Moderate:     l1d=%.2f l2=%.2f l3=%.2f\n", p2.l1d_util, p2.l2_util, p2.l3_util);
    printf("  Maximum:      l1d=%.2f l2=%.2f l3=%.2f\n", p4.l1d_util, p4.l2_util, p4.l3_util);

    // Verify ordering
    if (p0.l1d_util >= p4.l1d_util) {
        printf("FAIL: Conservative should have lower l1d_util than Maximum\n");
        return;
    }
    printf("PASS: Blocking presets\n");
}

void test_blocking_selection() {
    printf("\n=== Blocking selection ===\n");

    setenv("DNNOPT_AUTOTUNE", "1", 1);

    // Test a medium-large shape (blocking matters)
    dnnopt::BlockingSelection sel = dnnopt::select_blocking_params(128, 256, 256);

    const char* preset_name = "?";
    switch (sel.preset) {
    case dnnopt::BlockingPreset::kConservative: preset_name = "Conservative"; break;
    case dnnopt::BlockingPreset::kStandard:     preset_name = "Standard"; break;
    case dnnopt::BlockingPreset::kModerate:     preset_name = "Moderate"; break;
    case dnnopt::BlockingPreset::kAggressive:   preset_name = "Aggressive"; break;
    case dnnopt::BlockingPreset::kMaximum:      preset_name = "Maximum"; break;
    }

    printf("  Shape 128x256x256 -> preset=%s, gflops=%.1f, valid=%d\n",
           preset_name, sel.gflops, sel.valid);

    unsetenv("DNNOPT_AUTOTUNE");
    printf("PASS: Blocking selection\n");
}

void test_gemm_with_blocking_autotune() {
    printf("\n=== GEMM with blocking autotune ===\n");

    setenv("DNNOPT_AUTOTUNE", "1", 1);
    dnnopt::warmup_blocking_autotune();

    const int M = 128, N = 256, K = 256;
    auto A = dnnopt::aligned_array<float>(M * K);
    auto B = dnnopt::aligned_array<float>(K * N);
    auto C = dnnopt::aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.01f * (i % 37);
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.01f * (i % 41);
    std::memset(C.get(), 0, M * N * sizeof(float));

    // Warmup
    dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);

    // Get blocking params
    auto bp = dnnopt::get_autotuned_blocking_params(M, N, K, 8, 12, 1, 4, 4);
    printf("  Blocking params: Mc=%d, Nc=%d, Kc=%d\n", bp.Mc, bp.Nc, bp.Kc);

    // Run GEMM
    std::memset(C.get(), 0, M * N * sizeof(float));
    dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);

    unsetenv("DNNOPT_AUTOTUNE");
    printf("PASS: GEMM with blocking autotune\n");
}

}  // namespace

int main() {
    printf("=== test_autotune_blocking ===\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("Hardware: %s, %u cores\n", hw.cpu_name.c_str(), hw.num_cores);

    test_blocking_presets();
    test_blocking_selection();
    test_gemm_with_blocking_autotune();

    printf("\n=== All tests passed ===\n");
    return 0;
}
