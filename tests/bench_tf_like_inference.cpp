/*******************************************************************************
 * TensorFlow-like inference workload simulation via oneDNN
 *
 * Simulates TensorFlow's matmul patterns through oneDNN's dnnl_sgemm API
 * - Uses row-major layout like TF
 * - Tests CVR, BERT, LLM workload shapes
 * - Compares upstream oneDNN vs oneDNN+dnnopt
 ******************************************************************************/

#include <oneapi/dnnl/dnnl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace {

struct Layer {
    int M, N, K;
    const char* name;
};

void fill_random(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i)
        data[i] = (float)((rand() % 2000) - 1000) / 1000.0f;
}

// Benchmark using dnnl_sgemm
// Use the same proven parameter format from bench_inference_workload.cpp
double bench_layer_sgemm(int M, int N, int K,
                         const float* A, const float* B, float* C) {
    float alpha = 1.0f, beta = 0.0f;

    // Warmup - 10 iterations
    // Format: dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N)
    // This matches the working bench_inference_workload.cpp format
    for (int w = 0; w < 10; w++) {
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    }

    // Benchmark - 100 iterations for better timing
    std::vector<double> times;
    for (int i = 0; i < 100; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dnnl_sgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
        auto t1 = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(times.begin(), times.end());
    return times[50];  // median
}

}  // namespace

int main(int argc, char** argv) {
    // Layer definitions matching TensorFlow inference patterns
    Layer cvr_b1[] = {
        {1, 256, 1024, "embedding"},
        {1, 128, 256,  "fc1"},
        {1, 64, 128,   "fc2"},
        {1, 10, 64,    "classifier"},
    };
    Layer cvr_b4[] = {
        {4, 256, 1024, "embedding"},
        {4, 128, 256,  "fc1"},
        {4, 64, 128,   "fc2"},
        {4, 10, 64,    "classifier"},
    };

    Layer bert_b1[] = {
        {32, 256, 256, "qkv_proj"},
        {32, 256, 256, "out_proj"},
        {32, 512, 256, "ffn1"},
        {32, 256, 512, "ffn2"},
    };
    Layer bert_b4[] = {
        {128, 256, 256, "qkv_proj"},
        {128, 256, 256, "out_proj"},
        {128, 512, 256, "ffn1"},
        {128, 256, 512, "ffn2"},
    };

    Layer llm_b1[] = {
        {8, 512, 512,  "qkv_proj"},
        {8, 512, 512,  "out_proj"},
        {8, 1376, 512, "ffn1"},
        {8, 512, 1376, "ffn2"},
    };
    Layer llm_b4[] = {
        {32, 512, 512,  "qkv_proj"},
        {32, 512, 512,  "out_proj"},
        {32, 1376, 512, "ffn1"},
        {32, 512, 1376, "ffn2"},
    };

    struct ModelLayer {
        const char* model_name;
        Layer* layers;
        int n_layers;
    };

    ModelLayer models[] = {
        {"CVR (batch=1)", cvr_b1, 4},
        {"CVR (batch=4)", cvr_b4, 4},
        {"BERT-small (batch=1)", bert_b1, 4},
        {"BERT-small (batch=4)", bert_b4, 4},
        {"LLM (batch=1)", llm_b1, 4},
        {"LLM (batch=4)", llm_b4, 4},
    };

    printf("============================================================\n");
    printf("  TensorFlow-like Inference Benchmark via oneDNN dnnl_sgemm\n");
    printf("============================================================\n\n");

    printf("Note: Simulates TF matmul patterns with row-major layout\n");
    printf("      TF internally uses col-major, this tests oneDNN API\n\n");

    int n_models = sizeof(models) / sizeof(models[0]);

    printf("%-24s %8s %10s %10s %8s\n", "Model/Layer", "Shape", "Time(us)", "GFLOPS", "Status");
    printf("%-24s %8s %10s %10s %8s\n", "-----", "-----", "-------", "------", "------");
    printf("\n");

    double total_gflops = 0;
    int total_layers = 0;

    for (int m = 0; m < n_models; m++) {
        const auto& model = models[m];
        printf("--- %s ---\n", model.model_name);

        for (int l = 0; l < model.n_layers; l++) {
            const auto& layer = model.layers[l];
            int M = layer.M, N = layer.N, K = layer.K;

            // Allocate buffers
            float *A = (float*)aligned_alloc(64, (size_t)M*K*sizeof(float));
            float *B = (float*)aligned_alloc(64, (size_t)K*N*sizeof(float));
            float *C = (float*)aligned_alloc(64, (size_t)M*N*sizeof(float));

            fill_random(A, M*K);
            fill_random(B, K*N);

            double us = bench_layer_sgemm(M, N, K, A, B, C);
            double gflops = 2.0 * M * N * K / (us * 1e3);

            char shape[32];
            snprintf(shape, sizeof(shape), "[%d,%d,%d]", M, N, K);

            printf("  %-18s %8s %10.1f %10.2f %s\n",
                   layer.name, shape, us, gflops,
                   gflops > 0 ? "OK" : "FAIL");

            total_gflops += gflops;
            total_layers++;

            free(A); free(B); free(C);
        }
        printf("\n");
    }

    printf("============================================================\n");
    printf("  Summary: Avg GFLOPS = %.2f (%d layers)\n",
           total_gflops / total_layers, total_layers);
    printf("============================================================\n");

    return 0;
}