/*******************************************************************************
 * Test oneDNN+dnnopt GEMM dispatch via matmul primitive
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

using namespace dnnl;

// Helper to create memory descriptor
memory::desc make_md(int ndims, const dim_t *dims, data_type_t dt, memory::format_tag_t tag) {
    return memory::desc(ndims, dims, dt, tag);
}

// Simple matmul via oneDNN primitive
float test_matmul(engine &eng, stream &s, int M, int N, int K, int warmup = 5, int iters = 30) {
    // Create memory descriptors for row-major layout
    dim_t dims_A[2] = {M, K};
    dim_t dims_B[2] = {K, N};
    dim_t dims_C[2] = {M, N};

    auto md_A = make_md(2, dims_A, data_type::f32, memory::format_tag::ab);
    auto md_B = make_md(2, dims_B, data_type::f32, memory::format_tag::ab);
    auto md_C = make_md(2, dims_C, data_type::f32, memory::format_tag::ab);

    // Allocate buffers
    std::vector<float> A_data(M * K, 1.0f);
    std::vector<float> B_data(K * N, 1.0f);
    std::vector<float> C_data(M * N, 0.0f);

    auto mem_A = memory(md_A, eng, A_data.data());
    auto mem_B = memory(md_B, eng, B_data.data());
    auto mem_C = memory(md_C, eng, C_data.data());

    // Create matmul primitive descriptor
    auto matmul_pd = matmul::primitive_desc(eng, md_A, md_B, md_C);
    auto matmul_prim = matmul(matmul_pd);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        matmul_prim.execute(s, {{DNNL_ARG_SRC, mem_A}, {DNNL_ARG_WEIGHTS, mem_B}, {DNNL_ARG_DST, mem_C}});
    }
    s.wait();

    // Benchmark
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; i++) {
        matmul_prim.execute(s, {{DNNL_ARG_SRC, mem_A}, {DNNL_ARG_WEIGHTS, mem_B}, {DNNL_ARG_DST, mem_C}});
    }
    s.wait();
    auto t1 = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / iters;
    float flops = 2.0f * M * N * K;
    float gflops = flops / (ms * 1e6);

    // Verify result
    float expected = K;
    float got = C_data[0];
    if (std::abs(got - expected) > 0.01f) {
        std::cerr << "ERROR: C[0]=" << got << ", expected=" << expected << std::endl;
        return -1;
    }

    return gflops;
}

int main() {
    engine eng(engine::kind::cpu);
    stream s(eng);

    std::cout << "oneDNN+dnnopt MatMul Primitive Benchmark" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Test shapes
    struct TestCase {
        std::string name;
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

    std::cout << "\nShape                    M    N    K    GFLOPS    Status" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    float total_gflops = 0;
    int count = 0;

    for (const auto &tc : tests) {
        float gflops = test_matmul(eng, s, tc.M, tc.N, tc.K);
        std::cout << tc.name << std::setw(20 - tc.name.length())
                  << " " << tc.M << " " << tc.N << " " << tc.K << "   ";
        if (gflops < 0) {
            std::cout << "  ERROR" << std::endl;
        } else {
            std::cout << std::fixed << std::setprecision(2) << gflops << "    OK" << std::endl;
            total_gflops += gflops;
            count++;
        }
    }

    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Average GFLOPS: " << (total_gflops / count) << std::endl;

    return 0;
}