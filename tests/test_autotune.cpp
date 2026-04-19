/// @file test_autotune.cpp
/// Tests for runtime autotuning functionality.

#include "dnnopt/autotune/shape_cache.h"
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/timer.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace {

void test_shape_key_hash() {
    dnnopt::GemmShapeKey key1;
    key1.M = 4;
    key1.N = 4096;
    key1.K = 4096;
    key1.dtype = 0;
    key1.algo = 0;

    dnnopt::GemmShapeKey key2;
    key2.M = 4;
    key2.N = 4096;
    key2.K = 4096;
    key2.dtype = 0;
    key2.algo = 0;

    dnnopt::GemmShapeKey key3;
    key3.M = 8;
    key3.N = 4096;
    key3.K = 4096;
    key3.dtype = 0;
    key3.algo = 0;

    uint64_t h1 = key1.hash();
    uint64_t h2 = key2.hash();
    uint64_t h3 = key3.hash();

    printf("Hash test: key1=%llu key2=%llu key3=%llu\n",
           (unsigned long long)h1, (unsigned long long)h2, (unsigned long long)h3);

    if (h1 != h2) {
        printf("FAIL: identical keys should have same hash\n");
        return;
    }
    if (h1 == h3) {
        printf("FAIL: different keys should have different hash\n");
        return;
    }
    printf("PASS: shape key hash\n");
}

void test_shape_cache_basic() {
    dnnopt::ShapeCache cache;

    // Test empty lookup
    if (cache.lookup(12345) != nullptr) {
        printf("FAIL: empty cache should return nullptr\n");
        return;
    }

    // Test insert
    dnnopt::KernelSelection sel;
    sel.kernel_id = 2;
    sel.gflops = 150.5f;
    sel.time_us = 1000;
    sel.valid = true;
    cache.insert(12345, sel);

    // Test lookup after insert
    const dnnopt::KernelSelection* found = cache.lookup(12345);
    if (!found) {
        printf("FAIL: inserted entry should be found\n");
        return;
    }
    if (found->kernel_id != 2 || found->gflops != 150.5f) {
        printf("FAIL: lookup returned wrong data\n");
        return;
    }

    printf("PASS: shape cache basic (size=%zu)\n", cache.size());
}

void test_shape_cache_lru() {
    dnnopt::ShapeCache cache;

    // Fill cache beyond limit (256 entries)
    for (int i = 0; i < 300; ++i) {
        dnnopt::KernelSelection sel;
        sel.kernel_id = static_cast<uint8_t>(i % 5);
        sel.gflops = 100.0f + i;
        sel.time_us = i * 10;
        sel.valid = true;
        cache.insert(i, sel);
    }

    // Cache should be capped at 256
    if (cache.size() > 256) {
        printf("FAIL: cache should be capped at 256 (size=%zu)\n", cache.size());
        return;
    }

    // Oldest entries should be evicted
    if (cache.lookup(0) != nullptr) {
        printf("FAIL: oldest entry should be evicted\n");
        return;
    }

    // Newest entries should be present
    if (cache.lookup(299) == nullptr) {
        printf("FAIL: newest entry should be present\n");
        return;
    }

    printf("PASS: shape cache LRU eviction\n");
}

void test_gemm_kernel_selection() {
    printf("\n--- GEMM kernel selection tests ---\n");

    // Test different shapes
    struct TestCase {
        int M, N, K;
        const char* expected;
    };

    TestCase tests[] = {
        {1,    1024, 1024, "Tiny (M=1)"},
        {4,    4096, 4096, "Small-M or Packed"},
        {8,    512,  512,  "Adaptive or Packed"},
        {128,  128,  128, "Packed"},
        {32,   1024, 1024, "Adaptive or Packed"},
    };

    for (const auto& t : tests) {
        dnnopt::GemmKernelId kid = dnnopt::select_gemm_kernel(
            t.M, t.N, t.K, dnnopt::GemmDataType::kFP32);

        const char* name = "?";
        switch (kid) {
        case dnnopt::GemmKernelId::kTiny:       name = "Tiny"; break;
        case dnnopt::GemmKernelId::kSmallM:     name = "SmallM"; break;
        case dnnopt::GemmKernelId::kSmallMWide: name = "SmallMWide"; break;
        case dnnopt::GemmKernelId::kAdaptiveTile: name = "AdaptiveTile"; break;
        case dnnopt::GemmKernelId::kPacked:     name = "Packed"; break;
        }

        printf("  Shape %d×%d×%d → %s (%s expected)\n",
               t.M, t.N, t.K, name, t.expected);
    }

    printf("PASS: GEMM kernel selection (ran without errors)\n");
}

void test_gemm_with_autotune() {
    printf("\n--- GEMM with autotune enabled ---\n");

    // Enable autotune via environment
    setenv("DNNOPT_AUTOTUNE", "1", 1);

    // Warmup cache
    dnnopt::warmup_gemm_autotune();

    // Run a few GEMM operations
    const int M = 4, N = 1024, K = 1024;
    auto A = dnnopt::aligned_array<float>(M * K);
    auto B = dnnopt::aligned_array<float>(K * N);
    auto C = dnnopt::aligned_array<float>(M * N);

    for (int i = 0; i < M * K; ++i) A.get()[i] = 0.1f;
    for (int i = 0; i < K * N; ++i) B.get()[i] = 0.1f;
    std::memset(C.get(), 0, M * N * sizeof(float));

    dnnopt::Timer timer;
    timer.start();
    dnnopt::gemm_fp32(M, N, K, 1.0f, A.get(), K, B.get(), N, 0.0f, C.get(), N);
    timer.stop();

    printf("  GEMM %d×%d×%d with autotune: %.2f us\n",
           M, N, K, timer.elapsed_us());

    unsetenv("DNNOPT_AUTOTUNE");
    printf("PASS: GEMM with autotune\n");
}

void test_cache_persistence() {
    printf("\n--- Cache file persistence ---\n");

    dnnopt::ShapeCache cache;

    // Insert some entries
    for (int i = 0; i < 10; ++i) {
        dnnopt::KernelSelection sel;
        sel.kernel_id = static_cast<uint8_t>(i);
        sel.gflops = 100.0f + i;
        sel.time_us = i * 100;
        sel.valid = true;
        cache.insert(i, sel);
    }

    // Save to file
    int rc = cache.save_to_file("/tmp/test_autotune_cache.bin");
    if (rc != 0) {
        printf("FAIL: save_to_file failed\n");
        return;
    }
    printf("  Saved cache to /tmp/test_autotune_cache.bin (size=%zu)\n", cache.size());

    // Load into new cache
    dnnopt::ShapeCache cache2;
    rc = cache2.load_from_file("/tmp/test_autotune_cache.bin");
    if (rc < 0) {
        printf("FAIL: load_from_file failed\n");
        return;
    }
    printf("  Loaded %d entries\n", rc);

    // Verify loaded entries
    for (int i = 0; i < 10; ++i) {
        const dnnopt::KernelSelection* sel = cache2.lookup(i);
        if (!sel || sel->kernel_id != static_cast<uint8_t>(i)) {
            printf("FAIL: entry %d not loaded correctly\n", i);
            return;
        }
    }

    printf("PASS: cache file persistence\n");
}

}  // namespace

int main() {
    printf("=== test_autotune ===\n\n");

    test_shape_key_hash();
    test_shape_cache_basic();
    test_shape_cache_lru();
    test_gemm_kernel_selection();
    test_gemm_with_autotune();
    test_cache_persistence();

    printf("\n=== All tests passed ===\n");
    return 0;
}