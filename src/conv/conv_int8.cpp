/// @file conv_int8.cpp
/// INT8 Conv2D implementation using SMMLA instruction.
///
/// INT8 quantized convolution provides:
///   - 4x higher compute density: SMMLA does 4x more ops than FMLA
///   - 4x better memory bandwidth: 8-bit vs 32-bit data
///
/// Implementation:
///   1. Compute quantization scales (per-tensor dynamic)
///   2. Quantize FP32 input → INT8 on-the-fly
///   3. Quantize FP32 filter → INT8
///   4. Conv2D with SMMLA kernel (INT8 × INT8 → INT32 accumulate)
///   5. Dequantize output (INT32 → FP32 with scale)
///
/// Hardware: ARMv8.6-A I8MM extension (SMMLA, UMMLA, SDOT, UDOT)

#include "dnnopt/conv/conv.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"

#include <cstring>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>

#if defined(__ARM_FEATURE_MATMUL_INT8)
// SMMLA is available
#endif
#endif

namespace dnnopt {

// Forward declaration
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op);

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)

// ============================================================
// INT8 quantization helpers
// ============================================================

/// Compute quantization scale: scale = max_abs / 127.
static float compute_quant_scale(const float* data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 127.0f;
}

/// Quantize FP32 array to INT8 with given scale.
static void quantize_fp32_to_int8(const float* src, int8_t* dst, size_t n, float scale) {
    float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; ++i) {
        float q = src[i] * inv_scale;
        // Clamp to [-128, 127]
        int qi = static_cast<int>(std::round(q));
        if (qi > 127) qi = 127;
        else if (qi < -128) qi = -128;
        dst[i] = static_cast<int8_t>(qi);
    }
}

/// Dequantize INT32 accumulator to FP32 with scales.
/// output_val = acc * input_scale * filter_scale
static void dequantize_int32_to_fp32(const int32_t* src, float* dst, size_t n,
                                      float input_scale, float filter_scale) {
    float scale = input_scale * filter_scale;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

/// INT8 Conv2D kernel for 3x3 stride=1 pad=1.
static void conv2d_int8_3x3_s1p1_impl(const Conv2DParams& p,
                                       const int8_t* input_q,
                                       const int8_t* filter_q,
                                       int32_t* output_acc,
                                       float input_scale, float filter_scale) {
    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC, OC = p.OC;
    const int OH = p.OH(), OW = p.OW();
    const int K = IC * 3 * 3;
    const int M = N * OH * OW;

    // im2col INT8: [N*OH*OW, K]
    auto col_q = aligned_array<int8_t>((size_t)M * K);

    // Simplified im2col for 3x3 stride=1 pad=1
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int row = (n * OH + oh) * OW + ow;
                int col_idx = 0;

                for (int kh = 0; kh < 3; ++kh) {
                    const int ih = oh - 1 + kh;
                    if (ih < 0 || ih >= IH) {
                        for (int kw = 0; kw < 3; ++kw) {
                            for (int ic = 0; ic < IC; ++ic) {
                                col_q.get()[row * K + col_idx++] = 0;
                            }
                        }
                        continue;
                    }

                    for (int kw = 0; kw < 3; ++kw) {
                        const int iw = ow - 1 + kw;
                        if (iw < 0 || iw >= IW) {
                            for (int ic = 0; ic < IC; ++ic) {
                                col_q.get()[row * K + col_idx++] = 0;
                            }
                            continue;
                        }

                        for (int ic = 0; ic < IC; ++ic) {
                            col_q.get()[row * K + col_idx++] =
                                input_q[((n * IH + ih) * IW + iw) * IC + ic];
                        }
                    }
                }
            }
        }
    }

    // GEMM INT8: output_acc[M, OC] = col_q[M, K] × filter_q[K, OC]
    // Use gemm_int8 or SMMLA directly
    // For simplicity, call gemm_int8 which handles SMMLA dispatch
    gemm_int8(M, OC, K,
              1.0f, reinterpret_cast<const float*>(col_q.get()), K,
              reinterpret_cast<const float*>(filter_q), K,
              0.0f, reinterpret_cast<float*>(output_acc), OC);
}

#endif  // __ARM_FEATURE_MATMUL_INT8
#endif  // __ARM_NEON

/// INT8 Conv2D dispatch with dynamic quantization.
void conv2d_int8(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op) {
    const auto& hw = detect_arm_hwcaps();
    bool has_i8mm = (hw.hwcaps & static_cast<uint64_t>(HwCap::kI8MM)) != 0;

    if (!has_i8mm) {
        // Fallback to FP32
        conv2d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)
    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC, OC = p.OC;
    const int K = IC * p.KH * p.KW;
    const int OH = p.OH(), OW = p.OW();
    const int M = N * OH * OW;

    // Compute quantization scales
    float input_scale = compute_quant_scale(input, N * IH * IW * IC);
    float filter_scale = compute_quant_scale(filter, OC * K);

    // Quantize input and filter
    auto input_q = aligned_array<int8_t>((size_t)N * IH * IW * IC);
    auto filter_q = aligned_array<int8_t>((size_t)OC * K);

    quantize_fp32_to_int8(input, input_q.get(), N * IH * IW * IC, input_scale);
    quantize_fp32_to_int8(filter, filter_q.get(), OC * K, filter_scale);

    // INT32 accumulator buffer
    auto output_acc = aligned_array<int32_t>((size_t)M * OC);

    // Select kernel
    if (p.KH == 3 && p.KW == 3 && p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 1 && p.pad_w == 1) {
        conv2d_int8_3x3_s1p1_impl(p, input_q.get(), filter_q.get(),
                                  output_acc.get(), input_scale, filter_scale);
    } else {
        // General path: im2col + gemm_int8
        // For simplicity, fallback to FP32 for now
        conv2d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

    // Dequantize to FP32
    dequantize_int32_to_fp32(output_acc.get(), output, M * OC, input_scale, filter_scale);

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        apply_conv_postops(output, M, OC, bias, post_op);
    }
    return;
#endif
#endif

    // Fallback
    conv2d_fp32(p, input, filter, bias, output, post_op);
}

}  // namespace dnnopt