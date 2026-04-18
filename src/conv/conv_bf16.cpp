/// @file conv_bf16.cpp
/// BF16 Conv2D implementation using BFMMLA instruction.
///
/// BF16 (BFloat16) has 7-bit mantissa (vs FP32's 23-bit), allowing:
///   - 2x higher compute density: BFMMLA does 2x more ops than FMLA
///   - Better memory bandwidth: 16-bit vs 32-bit data
///
/// Implementation:
///   1. Convert FP32 input → BF16 on-the-fly during compute
///   2. Convert FP32 filter → BF16 (can be pre-computed for static weights)
///   3. Conv2D with BFMMLA kernel
///   4. Output in FP32 (accumulation is always FP32)
///
/// Hardware: ARMv8.6-A BF16 extension (BFMMLA, BFDOT)

#include "dnnopt/conv/conv.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"
#include "dnnopt/arm_hwcaps.h"

#include <cstring>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>

#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#include <arm_bf16.h>
#endif
#endif

namespace dnnopt {

// Forward declaration
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op);

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

// ============================================================
// BF16 conversion helpers
// ============================================================

/// Convert 4 FP32 values to 4 BF16 values.
static inline bfloat16x4_t fp32_to_bf16_4(const float* ptr) {
    float32x4_t f32 = vld1q_f32(ptr);
    return vcvt_bf16_f32(f32);
}

/// Convert FP32 array to BF16 array.
static void fp32_to_bf16_array(const float* src, bfloat16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        bfloat16x4_t bf16 = fp32_to_bf16_4(src + i);
        vst1_bf16(reinterpret_cast<__bf16*>(dst + i), bf16);
    }
    // Tail
    for (; i < n; ++i) {
        dst[i] = bfloat16_t(src[i]);
    }
}

/// Compute max absolute value for quantization scale.
static float compute_max_abs(const float* data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    return max_abs;
}

/// BF16 Conv2D kernel for 3x3 stride=1 pad=1.
/// Uses im2col + BFMMLA GEMM.
static void conv2d_bf16_3x3_s1p1_impl(const Conv2DParams& p,
                                       const float* input,
                                       const bfloat16_t* filter_bf16,
                                       float* output) {
    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC, OC = p.OC;
    const int OH = p.OH(), OW = p.OW();
    const int K = IC * 3 * 3;
    const int M = N * OH * OW;

    // im2col: [N*OH*OW, K] FP32
    auto col = aligned_array<float>((size_t)M * K);

    // Simplified im2col for 3x3 stride=1 pad=1
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int row = (n * OH + oh) * OW + ow;
                int col_idx = 0;

                for (int kh = 0; kh < 3; ++kh) {
                    const int ih = oh - 1 + kh;  // stride=1, pad=1
                    if (ih < 0 || ih >= IH) {
                        for (int kw = 0; kw < 3; ++kw) {
                            for (int ic = 0; ic < IC; ++ic) {
                                col.get()[row * K + col_idx++] = 0.0f;
                            }
                        }
                        continue;
                    }

                    for (int kw = 0; kw < 3; ++kw) {
                        const int iw = ow - 1 + kw;
                        if (iw < 0 || iw >= IW) {
                            for (int ic = 0; ic < IC; ++ic) {
                                col.get()[row * K + col_idx++] = 0.0f;
                            }
                            continue;
                        }

                        for (int ic = 0; ic < IC; ++ic) {
                            col.get()[row * K + col_idx++] =
                                input[((n * IH + ih) * IW + iw) * IC + ic];
                        }
                    }
                }
            }
        }
    }

    // Convert col to BF16
    auto col_bf16 = aligned_array<bfloat16_t>((size_t)M * K);
    fp32_to_bf16_array(col.get(), col_bf16.get(), M * K);

    // GEMM with BFMMLA: output[M, OC] = col_bf16[M, K] × filter_bf16[K, OC]
    // Call gemm_bf16 with BF16 inputs
    gemm_bf16(M, OC, K,
              1.0f, reinterpret_cast<const float*>(col_bf16.get()), K,
              reinterpret_cast<const float*>(filter_bf16), K,  // Filter layout adjusted
              0.0f, output, OC);
}

#endif  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif  // __ARM_NEON

/// BF16 Conv2D dispatch.
void conv2d_bf16(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op) {
    const auto& hw = detect_arm_hwcaps();
    bool has_bf16 = (hw.hwcaps & static_cast<uint64_t>(HwCap::kBF16)) != 0;

    if (!has_bf16) {
        // Fallback to FP32
        conv2d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    const int OC = p.OC;
    const int K = p.IC * p.KH * p.KW;

    // Convert filter to BF16 (one-time cost)
    auto filter_bf16 = aligned_array<bfloat16_t>((size_t)OC * K);
    fp32_to_bf16_array(filter, filter_bf16.get(), OC * K);

    // Select kernel based on shape
    if (p.KH == 3 && p.KW == 3 && p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 1 && p.pad_w == 1) {
        conv2d_bf16_3x3_s1p1_impl(p, input, filter_bf16.get(), output);
    } else {
        // General path: im2col + gemm_bf16
        const int N = p.N;
        const int OH = p.OH(), OW = p.OW();
        const int M = N * OH * OW;

        // im2col FP32 → convert to BF16
        auto col = aligned_array<float>((size_t)M * K);

        // Call im2col helper (from conv2d.cpp)
        Conv2DParams p_im2col = p;
        p_im2col.IC = p.IC;
        // Need external im2col function...

        // For now, use FP32 conv as fallback
        conv2d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        const int M = p.N * p.OH() * p.OW();
        apply_conv_postops(output, M, p.OC, bias, post_op);
    }
    return;
#endif
#endif

    // Fallback
    conv2d_fp32(p, input, filter, bias, output, post_op);
}

}  // namespace dnnopt