/// @file bench_conv3d.cpp
/// Conv3D benchmark suite for video processing.
/// Tests FP32, BF16, INT8 implementations with C3D/I3D shapes.

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"
#include "dnnopt/conv/conv3d.h"
#include "dnnopt/timer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

struct Conv3DShape {
    int N, IC, ID, IH, IW;
    int OC, KD, KH, KW;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    const char* label;

    int OD() const { return (ID + 2 * pad_d - KD) / stride_d + 1; }
    int OH() const { return (IH + 2 * pad_h - KH) / stride_h + 1; }
    int OW() const { return (IW + 2 * pad_w - KW) / stride_w + 1; }

    double flops() const {
        return 2.0 * N * OC * OD() * OH() * OW() * IC * KD * KH * KW;
    }
};

// C3D/I3D video model shapes
const Conv3DShape conv3d_shapes[] = {
    // Smaller shapes for quick tests
    {1, 16,   5,  16,  16,  32, 3, 3, 3, 1, 1, 1, 1, 1, 1, "Small-3x3x3"},
    {1, 32,   4,  14,  14,  64, 3, 3, 3, 1, 1, 1, 1, 1, 1, "Medium-3x3x3"},
    // Minimal C3D shapes
    {1,  3,   8,  56,  56,  64, 3, 3, 3, 1, 1, 1, 1, 1, 1, "C3D-mini"},
    {1, 64,   4,  28,  28, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1, "C3D-conv2-mini"},
    // Batch processing
    {2, 16,   5,  16,  16,  32, 3, 3, 3, 1, 1, 1, 1, 1, 1, "Small-batch2"},
};

dnnopt::Conv3DParams to_lib_params(const Conv3DShape& s) {
    dnnopt::Conv3DParams p;
    p.N = s.N; p.IC = s.IC;
    p.ID = s.ID; p.IH = s.IH; p.IW = s.IW;
    p.OC = s.OC; p.KD = s.KD; p.KH = s.KH; p.KW = s.KW;
    p.stride_d = s.stride_d; p.stride_h = s.stride_h; p.stride_w = s.stride_w;
    p.pad_d = s.pad_d; p.pad_h = s.pad_h; p.pad_w = s.pad_w;
    return p;
}

void fill_random(float* data, size_t n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

}  // namespace

int main(int argc, char** argv) {
    printf("==========================================================\n");
    printf("  Conv3D Benchmark Suite (NDHWC)\n");
    printf("==========================================================\n\n");

    const auto& hw = dnnopt::detect_arm_hwcaps();
    printf("CPU: %s @ %u MHz, %u cores\n\n", hw.cpu_name.c_str(), hw.freq_mhz, hw.num_cores);

    bool has_bf16 = (hw.hwcaps & static_cast<uint64_t>(dnnopt::HwCap::kBF16)) != 0;
    bool has_i8mm = (hw.hwcaps & static_cast<uint64_t>(dnnopt::HwCap::kI8MM)) != 0;
    printf("BF16: %s, INT8: %s\n\n", has_bf16 ? "YES" : "NO", has_i8mm ? "YES" : "NO");

    int warmup = 2;
    int runs = 5;
    if (argc > 1) runs = atoi(argv[1]);

    std::vector<dnnopt::BenchStats> all_results;

    for (const auto& shape : conv3d_shapes) {
        int OD = shape.OD(), OH = shape.OH(), OW = shape.OW();
        size_t in_size = (size_t)shape.N * shape.ID * shape.IH * shape.IW * shape.IC;
        size_t flt_size = (size_t)shape.OC * shape.KD * shape.KH * shape.KW * shape.IC;
        size_t out_size = (size_t)shape.N * OD * OH * OW * shape.OC;

        auto input = dnnopt::aligned_array<float>(in_size);
        auto filter = dnnopt::aligned_array<float>(flt_size);
        auto output = dnnopt::aligned_array<float>(out_size);

        fill_random(input.get(), in_size);
        fill_random(filter.get(), flt_size);

        double flops = shape.flops();
        double bytes = (in_size + flt_size + out_size) * sizeof(float);

        auto lp = to_lib_params(shape);

        // --- FP32 Conv3D ---
        {
            char name[128];
            snprintf(name, sizeof(name), "%s [%d,%d,%d,%d->%d] FP32",
                     shape.label, shape.IC, shape.ID, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                dnnopt::conv3d_fp32(lp, input.get(), filter.get(), nullptr,
                                    output.get(), dnnopt::Conv3DPostOp::kNone);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // --- BF16 Conv3D ---
        if (has_bf16) {
            char name[128];
            snprintf(name, sizeof(name), "%s [%d,%d,%d,%d->%d] BF16",
                     shape.label, shape.IC, shape.ID, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                dnnopt::conv3d_bf16(lp, input.get(), filter.get(), nullptr,
                                    output.get(), dnnopt::Conv3DPostOp::kNone);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        // --- INT8 Conv3D ---
        if (has_i8mm) {
            char name[128];
            snprintf(name, sizeof(name), "%s [%d,%d,%d,%d->%d] INT8",
                     shape.label, shape.IC, shape.ID, shape.IH, shape.IW, shape.OC);
            auto stats = dnnopt::benchmark(name, flops, bytes, warmup, runs, [&]() {
                dnnopt::conv3d_int8(lp, input.get(), filter.get(), nullptr,
                                    output.get(), dnnopt::Conv3DPostOp::kNone);
            });
            dnnopt::print_bench_stats(stats);
            all_results.push_back(stats);
        }

        printf("\n");
    }

    dnnopt::write_csv("bench_conv3d_results.csv", all_results);
    printf("\n[Done] %zu benchmark results collected.\n", all_results.size());
    return 0;
}