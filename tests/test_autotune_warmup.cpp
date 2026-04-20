/// Test autotune warmup and persistence (minimal)
#include "dnnopt/gemm/gemm_autotune.h"
#include "dnnopt/autotune/shape_cache.h"
#include <cstdio>

using namespace dnnopt;

int main() {
    printf("=== Autotune Warmup & Persistence Test ===\n\n");

    // Test cache basic operations (without triggering benchmarks)
    printf("[1] Testing kernel cache operations...\n");
    auto& cache = get_gemm_shape_cache();

    KernelSelection sel;
    sel.kernel_id = static_cast<uint8_t>(GemmKernelId::kSmallM);
    sel.gflops = 20.0f;
    sel.time_us = 1000;
    sel.valid = true;

    cache.insert(12345, sel);
    printf("    Inserted entry. Cache size: %zu\n", cache.size());

    const KernelSelection* cached = cache.lookup(12345);
    if (cached && cached->kernel_id == sel.kernel_id) {
        printf("    PASS: Lookup found entry\n");
    } else {
        printf("    FAIL: Lookup failed\n");
        return 1;
    }

    // Test blocking cache
    printf("\n[2] Testing blocking cache operations...\n");
    auto& blocking_cache = get_blocking_cache();

    BlockingSelection bs;
    bs.preset = BlockingPreset::kModerate;
    bs.gflops = 40.0f;
    bs.time_us = 500;
    bs.valid = true;

    blocking_cache.insert(12345, bs);
    printf("    Inserted entry. Blocking cache size: %zu\n", blocking_cache.size());

    // Test file persistence
    printf("\n[3] Testing file persistence...\n");
    const char* path = "/tmp/test_cache.bin";
    int saved = cache.save_to_file(path);
    printf("    Saved: %d entries\n", saved);

    cache.clear();
    printf("    Cleared. Size: %zu\n", cache.size());

    int loaded = cache.load_from_file(path);
    printf("    Loaded: %d entries\n", loaded);

    if (loaded == 1) {
        printf("    PASS: Persistence works\n");
    } else {
        printf("    FAIL: Persistence failed\n");
        return 1;
    }

    // Test blocking cache persistence
    printf("\n[4] Testing blocking cache persistence...\n");
    const char* blocking_path = "/tmp/test_blocking.bin";
    blocking_cache.save_to_file(blocking_path);
    blocking_cache.clear();
    int blocking_loaded = blocking_cache.load_from_file(blocking_path);
    printf("    Blocking cache loaded: %d entries\n", blocking_loaded);

    // Test tile cache
    printf("\n[5] Testing tile cache operations...\n");
    auto& tile_cache = get_tile_cache();

    TileSelection ts;
    ts.preset = TilePreset::k8x12;
    ts.Mr = 8;
    ts.Nr = 12;
    ts.gflops = 45.0f;
    ts.valid = true;

    tile_cache.insert(12345, ts);
    printf("    Tile cache size: %zu\n", tile_cache.size());

    // Test clear all
    printf("\n[6] Testing clear_all_shape_caches()...\n");
    clear_all_shape_caches();
    printf("    Kernel cache: %zu, Blocking: %zu, Tile: %zu\n",
           cache.size(), blocking_cache.size(), tile_cache.size());

    if (cache.size() == 0 && blocking_cache.size() == 0 && tile_cache.size() == 0) {
        printf("    PASS: All caches cleared\n");
    } else {
        printf("    FAIL: Caches not fully cleared\n");
        return 1;
    }

    printf("\n=== All tests passed ===\n");
    return 0;
}