/// @file test_conv3d_correctness.cpp
/// Correctness tests for Conv3D implementations.

#include "dnnopt/aligned_alloc.h"
#include "dnnopt/conv/conv3d.h"
#include "test_utils.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

namespace {

struct Conv3DTestParams {
    int N, IC, ID, IH, IW, OC, KD, KH, KW;
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;
    const char* label;
    int OD() const { return (ID + 2 * pad_d - KD) / stride_d + 1; }
    int OH() const { return (IH + 2 * pad_h - KH) / stride_h + 1; }
    int OW() const { return (IW + 2 * pad_w - KW) / stride_w + 1; }
};

void fill_random(float* data, size_t n, unsigned seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < n; ++i) data[i] = dist(gen);
}

/// Reference NDHWC Conv3D (naive)
void conv3d_ref(const Conv3DTestParams& p,
                const float* input, const float* filter,
                const float* bias, float* output) {
    int OD = p.OD(), OH = p.OH(), OW = p.OW();
    for (int n = 0; n < p.N; ++n) {
        for (int od = 0; od < OD; ++od) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    for (int oc = 0; oc < p.OC; ++oc) {
                        float acc = bias ? bias[oc] : 0.0f;
                        for (int kd = 0; kd < p.KD; ++kd) {
                            int id = od * p.stride_d - p.pad_d + kd;
                            if (id < 0 || id >= p.ID) continue;
                            for (int kh = 0; kh < p.KH; ++kh) {
                                int ih = oh * p.stride_h - p.pad_h + kh;
                                if (ih < 0 || ih >= p.IH) continue;
                                for (int kw = 0; kw < p.KW; ++kw) {
                                    int iw = ow * p.stride_w - p.pad_w + kw;
                                    if (iw < 0 || iw >= p.IW) continue;
                                    for (int ic = 0; ic < p.IC; ++ic) {
                                        float in_val = input[((((n * p.ID + id) * p.IH + ih) * p.IW + iw) * p.IC + ic)];
                                        float flt_val = filter[((((oc * p.KD + kd) * p.KH + kh) * p.KW + kw) * p.IC + ic)];
                                        acc += in_val * flt_val;
                                    }
                                }
                            }
                        }
                        output[((((n * OD + od) * OH + oh) * OW + ow) * p.OC + oc)] = acc;
                    }
                }
            }
        }
    }
}

float max_diff(const float* a, const float* b, size_t n) {
    float md = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > md) md = d;
    }
    return md;
}

dnnopt::Conv3DParams to_lib_params(const Conv3DTestParams& p) {
    dnnopt::Conv3DParams lp;
    lp.N = p.N; lp.IC = p.IC;
    lp.ID = p.ID; lp.IH = p.IH; lp.IW = p.IW;
    lp.OC = p.OC; lp.KD = p.KD; lp.KH = p.KH; lp.KW = p.KW;
    lp.stride_d = p.stride_d; lp.stride_h = p.stride_h; lp.stride_w = p.stride_w;
    lp.pad_d = p.pad_d; lp.pad_h = p.pad_h; lp.pad_w = p.pad_w;
    return lp;
}

/// Test Conv3D FP32 vs reference
void test_conv3d_fp32(const Conv3DTestParams& p) {
    int OD = p.OD(), OH = p.OH(), OW = p.OW();
    size_t in_sz = (size_t)p.N * p.ID * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KD * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OD * OH * OW * p.OC;

    auto input = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_opt = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);

    conv3d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());

    auto lp = to_lib_params(p);
    dnnopt::conv3d_fp32(lp, input.get(), filter.get(), nullptr,
                        out_opt.get(), dnnopt::Conv3DPostOp::kNone);

    float tol = (float)(p.IC * p.KD * p.KH * p.KW) * 5e-5f;
    float md = max_diff(out_ref.get(), out_opt.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv3D %s FP32 max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

/// Test Conv3D BF16 vs FP32 reference (with relaxed tolerance)
void test_conv3d_bf16(const Conv3DTestParams& p) {
    int OD = p.OD(), OH = p.OH(), OW = p.OW();
    size_t in_sz = (size_t)p.N * p.ID * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KD * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OD * OH * OW * p.OC;

    auto input = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_bf16 = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);

    conv3d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());

    auto lp = to_lib_params(p);
    dnnopt::conv3d_bf16(lp, input.get(), filter.get(), nullptr,
                        out_bf16.get(), dnnopt::Conv3DPostOp::kNone);

    // BF16 has 7-bit mantissa, so tolerance is higher
    float tol = (float)(p.IC * p.KD * p.KH * p.KW) * 1e-2f;
    float md = max_diff(out_ref.get(), out_bf16.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv3D %s BF16 max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

/// Test Conv3D INT8 vs FP32 reference (with relaxed tolerance)
void test_conv3d_int8(const Conv3DTestParams& p) {
    int OD = p.OD(), OH = p.OH(), OW = p.OW();
    size_t in_sz = (size_t)p.N * p.ID * p.IH * p.IW * p.IC;
    size_t flt_sz = (size_t)p.OC * p.KD * p.KH * p.KW * p.IC;
    size_t out_sz = (size_t)p.N * OD * OH * OW * p.OC;

    auto input = dnnopt::aligned_array<float>(in_sz);
    auto filter = dnnopt::aligned_array<float>(flt_sz);
    auto out_ref = dnnopt::aligned_array<float>(out_sz);
    auto out_int8 = dnnopt::aligned_array<float>(out_sz);

    fill_random(input.get(), in_sz, 42);
    fill_random(filter.get(), flt_sz, 123);

    conv3d_ref(p, input.get(), filter.get(), nullptr, out_ref.get());

    auto lp = to_lib_params(p);
    dnnopt::conv3d_int8(lp, input.get(), filter.get(), nullptr,
                        out_int8.get(), dnnopt::Conv3DPostOp::kNone);

    // INT8 has 8-bit quantization, tolerance is higher
    float tol = (float)(p.IC * p.KD * p.KH * p.KW) * 5e-2f;
    float md = max_diff(out_ref.get(), out_int8.get(), out_sz);

    char msg[256];
    snprintf(msg, sizeof(msg), "Conv3D %s INT8 max_diff=%.6e tol=%.6e",
             p.label, md, tol);
    TEST_ASSERT(md < tol, msg);
}

}  // namespace

int main() {
    printf("=== test_conv3d_correctness ===\n");

    // Test shapes for video processing
    const Conv3DTestParams tests[] = {
        // Minimal test
        {1, 1, 3, 3, 3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, "minimal-3x3x3"},
        // Small C3D-like: 3 temporal frames
        {1, 3, 3, 8, 8, 4, 3, 3, 3, 1, 1, 1, 1, 1, 1, "c3d-small"},
        // Small with stride
        {1, 3, 4, 8, 8, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, "c3d-stride2"},
        // Medium shape
        {1, 16, 5, 16, 16, 32, 3, 3, 3, 1, 1, 1, 1, 1, 1, "medium-3x3x3"},
        // Larger temporal
        {1, 8, 8, 14, 14, 16, 5, 3, 3, 1, 1, 1, 2, 1, 1, "temporal-5"},
        // Batch test
        {2, 4, 4, 8, 8, 8, 3, 3, 3, 1, 1, 1, 1, 1, 1, "batch-2"},
    };

    // Section 1: FP32 tests
    printf("\n--- Conv3D FP32 vs ref ---\n");
    for (const auto& t : tests) {
        test_conv3d_fp32(t);
    }

    // Section 2: BF16 tests
    printf("\n--- Conv3D BF16 vs ref ---\n");
    for (const auto& t : tests) {
        test_conv3d_bf16(t);
    }

    // Section 3: INT8 tests
    printf("\n--- Conv3D INT8 vs ref ---\n");
    for (const auto& t : tests) {
        test_conv3d_int8(t);
    }

    TEST_SUMMARY();
}