/*******************************************************************************
 * Test oneDNN+dnnopt GEMM dispatch via matmul primitive (C API)
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include <vector>
#include <iomanip>
#include <iostream>

// Simple matmul via oneDNN primitive (C API)
float test_matmul(dnnl_engine_t eng, dnnl_stream_t s, int M, int N, int K, int warmup, int iters) {
    dnnl_status_t status;

    // Create memory descriptors for row-major (ab format)
    dnnl_dims_t dims_A[2] = {M, K};
    dnnl_dims_t dims_B[2] = {K, N};
    dnnl_dims_t dims_C[2] = {M, N};

    dnnl_memory_desc_t md_A, md_B, md_C;
    status = dnnl_memory_desc_init_by_tag(&md_A, 2, dims_A, dnnl_f32, dnnl_ab);
    if (status != dnnl_success) { printf("md_A init failed: %d\n", status); return -1; }

    status = dnnl_memory_desc_init_by_tag(&md_B, 2, dims_B, dnnl_f32, dnnl_ab);
    if (status != dnnl_success) { printf("md_B init failed: %d\n", status); return -1; }

    status = dnnl_memory_desc_init_by_tag(&md_C, 2, dims_C, dnnl_f32, dnnl_ab);
    if (status != dnnl_success) { printf("md_C init failed: %d\n", status); return -1; }

    // Allocate buffers
    std::vector<float> A_data(M * K, 1.0f);
    std::vector<float> B_data(K * N, 1.0f);
    std::vector<float> C_data(M * N, 0.0f);

    // Create memory objects
    dnnl_memory_t mem_A, mem_B, mem_C;
    status = dnnl_memory_create(&mem_A, &md_A, eng, A_data.data());
    if (status != dnnl_success) { printf("mem_A create failed: %d\n", status); return -1; }

    status = dnnl_memory_create(&mem_B, &md_B, eng, B_data.data());
    if (status != dnnl_success) { printf("mem_B create failed: %d\n", status); return -1; }

    status = dnnl_memory_create(&mem_C, &md_C, eng, C_data.data());
    if (status != dnnl_success) { printf("mem_C create failed: %d\n", status); return -1; }

    // Create matmul primitive descriptor
    dnnl_primitive_desc_t pd;
    status = dnnl_matmul_primitive_desc_create(&pd, eng, nullptr, &md_A, &md_B, &md_C);
    if (status != dnnl_success) { printf("matmul pd create failed: %d\n", status); return -1; }

    // Create primitive
    dnnl_primitive_t prim;
    status = dnnl_primitive_create(&prim, pd);
    if (status != dnnl_success) { printf("primitive create failed: %d\n", status); return -1; }

    // Execute args
    dnnl_exec_arg_t args[3] = {
        {DNNL_ARG_SRC, mem_A},
        {DNNL_ARG_WEIGHTS, mem_B},
        {DNNL_ARG_DST, mem_C},
    };

    // Warmup
    for (int i = 0; i < warmup; i++) {
        status = dnnl_primitive_execute(prim, s, 3, args);
        if (status != dnnl_success) { printf("warmup exec failed: %d\n", status); return -1; }
    }
    dnnl_stream_wait(s);

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        dnnl_primitive_execute(prim, s, 3, args);
    }
    dnnl_stream_wait(s);
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / iters;
    float flops = 2.0f * M * N * K;
    float gflops = flops / (ms * 1e6);

    // Verify result
    float expected = K * 1.0f * 1.0f;
    float got = C_data[0];
    if (std::abs(got - expected) > 0.01f) {
        printf("ERROR: C[0]=%f, expected=%f\n", got, expected);
        gflops = -1;
    }

    // Cleanup
    dnnl_primitive_destroy(prim);
    dnnl_primitive_desc_destroy(pd);
    dnnl_memory_destroy(mem_A);
    dnnl_memory_destroy(mem_B);
    dnnl_memory_destroy(mem_C);

    return gflops;
}

int main() {
    dnnl_status_t status;

    // Create engine and stream
    dnnl_engine_t eng;
    status = dnnl_engine_create(&eng, dnnl_cpu, 0);
    if (status != dnnl_success) { printf("engine create failed: %d\n", status); return 1; }

    dnnl_stream_t s;
    status = dnnl_stream_create(&s, eng);
    if (status != dnnl_success) { printf("stream create failed: %d\n", status); return 1; }

    printf("oneDNN+dnnopt MatMul Primitive Benchmark (C API)\n");
    printf("=================================================\n\n");

    // Test shapes
    struct TestCase {
        const char *name;
        int M, N, K;
    };

    TestCase tests[] = {
        {"embedding_b1",     1, 256, 1024},
        {"embedding_b4",     4, 256, 1024},
        {"qkv_proj_b1",      1, 768,  768},
        {"ffn1_b1",          1, 3072, 768},
        {"q_proj_b1",        1, 4096, 4096},
        {"gate_proj_b1",     1, 11008, 4096},
    };

    printf("Shape                    M    N    K    GFLOPS    Status\n");
    printf("-------------------------------------------------------\n");

    float total_gflops = 0;
    int count = 0;

    for (const auto &tc : tests) {
        float gflops = test_matmul(eng, s, tc.M, tc.N, tc.K, 5, 30);
        printf("%-16s       %4d %4d %4d   ", tc.name, tc.M, tc.N, tc.K);
        if (gflops < 0) {
            printf("  ERROR\n");
        } else {
            printf("%7.2f    OK\n", gflops);
            total_gflops += gflops;
            count++;
        }
    }

    printf("-------------------------------------------------------\n");
    printf("Average GFLOPS: %.2f\n", total_gflops / count);

    // Cleanup
    dnnl_stream_destroy(s);
    dnnl_engine_destroy(eng);

    return 0;
}