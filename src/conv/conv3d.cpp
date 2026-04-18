/// @file conv3d.cpp
/// 3D Convolution for video processing.
///
/// Conv3D extends Conv2D with a temporal dimension:
///   - Input:  [N, ID, IH, IW, IC] (batch, time, height, width, channels)
///   - Filter: [OC, KD, KH, KW, IC] (output channels, temporal kernel, spatial kernel, input channels)
///   - Output: [N, OD, OH, OW, OC]
///
/// Output dimensions:
///   - OD = (ID + 2*pad_d - KD) / stride_d + 1
///   - OH = (IH + 2*pad_h - KH) / stride_h + 1
///   - OW = (IW + 2*pad_w - KW) / stride_w + 1
///
/// Use cases:
///   - Video classification (C3D, I3D models)
///   - Action recognition (temporal feature extraction)
///   - Medical imaging (3D CT/MRI analysis)
///
/// Implementation:
///   - im2col3d + GEMM for general kernels
///   - BF16: BFMMLA for higher compute density
///   - INT8: SMMLA with dynamic quantization + native INT8 GEMM
///
/// TODO: Winograd F(2x2, 3x3x3) for temporal+spatial optimization
///       - Apply spatial Winograd (KH=3, KW=3) per temporal slice
///       - Combine temporal dimension in final reduction

#include "dnnopt/conv/conv3d.h"
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

/// Transpose filter from [OC, K] to [K, OC] for GEMM B matrix.
static void transpose_filter_3d(const float* filter, float* filter_T,
                                int OC, int K) {
    for (int oc = 0; oc < OC; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T[k * OC + oc] = filter[oc * K + k];
        }
    }
}

/// im2col3d for Conv3D: extracts receptive fields into column matrix.
/// Input: [N, ID, IH, IW, IC] → col [N*OD*OH*OW, IC*KD*KH*KW]
static void im2col_3d_nhwc(const Conv3DParams& p,
                           const float* input,
                           float* col) {
    const int N = p.N;
    const int ID = p.ID, IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KD = p.KD, KH = p.KH, KW = p.KW;
    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int stride_d = p.stride_d, stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_d = p.pad_d, pad_h = p.pad_h, pad_w = p.pad_w;

    const int col_rows = N * OD * OH * OW;
    const int col_cols = IC * KD * KH * KW;

    for (int n = 0; n < N; ++n) {
        for (int od = 0; od < OD; ++od) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int row = ((n * OD + od) * OH + oh) * OW + ow;
                    int col_idx = 0;

                    for (int kd = 0; kd < KD; ++kd) {
                        const int id = od * stride_d - pad_d + kd;
                        if (id < 0 || id >= ID) {
                            // Zero-padding for out-of-bounds temporal
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col[row * col_cols + col_idx++] = 0.0f;
                                    }
                                }
                            }
                            continue;
                        }

                        for (int kh = 0; kh < KH; ++kh) {
                            const int ih = oh * stride_h - pad_h + kh;
                            if (ih < 0 || ih >= IH) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col[row * col_cols + col_idx++] = 0.0f;
                                    }
                                }
                                continue;
                            }

                            for (int kw = 0; kw < KW; ++kw) {
                                const int iw = ow * stride_w - pad_w + kw;
                                if (iw < 0 || iw >= IW) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col[row * col_cols + col_idx++] = 0.0f;
                                    }
                                    continue;
                                }

                                for (int ic = 0; ic < IC; ++ic) {
                                    col[row * col_cols + col_idx++] =
                                        input[((((n * ID + id) * IH + ih) * IW + iw) * IC + ic)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Apply post-ops to Conv3D output.
static void apply_conv3d_postops(float* output, int num_rows, int OC,
                                  const float* bias, Conv3DPostOp op) {
    for (int idx = 0; idx < num_rows; ++idx) {
        for (int oc = 0; oc < OC; ++oc) {
            float val = output[idx * OC + oc];

            if (bias) val += bias[oc];

            switch (op) {
                case Conv3DPostOp::kRelu:
                    val = val > 0 ? val : 0;
                    break;
                case Conv3DPostOp::kRelu6:
                    val = val > 0 ? (val < 6 ? val : 6) : 0;
                    break;
                default:
                    break;
            }

            output[idx * OC + oc] = val;
        }
    }
}

/// Conv3D FP32: im2col3d + GEMM.
void conv3d_fp32(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op) {
    if (p.N <= 0 || p.OC <= 0 || p.IC <= 0) return;

    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int M = p.N * OD * OH * OW;
    const int K = p.IC * p.KD * p.KH * p.KW;
    const int N_gemm = p.OC;

    // im2col3d
    auto col = aligned_array<float>((size_t)M * K);
    im2col_3d_nhwc(p, input, col.get());

    // Transpose filter: [OC, K] → [K, OC]
    auto filter_T = aligned_array<float>((size_t)K * N_gemm);
    transpose_filter_3d(filter, filter_T.get(), N_gemm, K);

    // GEMM: output[M, OC] = col[M, K] × filter_T[K, OC]
    gemm_fp32(M, N_gemm, K,
              1.0f, col.get(), K,
              filter_T.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != Conv3DPostOp::kNone) {
        apply_conv3d_postops(output, M, N_gemm, bias, post_op);
    }
}

// ============================================================
// BF16 Conv3D Implementation
// ============================================================

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)

/// Convert 4 FP32 values to 4 BF16 values.
static inline bfloat16x4_t fp32_to_bf16_4_3d(const float* ptr) {
    float32x4_t f32 = vld1q_f32(ptr);
    return vcvt_bf16_f32(f32);
}

/// Convert FP32 array to BF16 array for Conv3D.
static void fp32_to_bf16_array_3d(const float* src, bfloat16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        bfloat16x4_t bf16 = fp32_to_bf16_4_3d(src + i);
        vst1_bf16(reinterpret_cast<__bf16*>(dst + i), bf16);
    }
    // Tail
    for (; i < n; ++i) {
        dst[i] = bfloat16_t(src[i]);
    }
}

/// im2col3d for BF16 Conv3D: extracts receptive fields and converts to BF16.
static void im2col_3d_nhwc_bf16(const Conv3DParams& p,
                                const float* input,
                                bfloat16_t* col_bf16) {
    const int N = p.N;
    const int ID = p.ID, IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KD = p.KD, KH = p.KH, KW = p.KW;
    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int stride_d = p.stride_d, stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_d = p.pad_d, pad_h = p.pad_h, pad_w = p.pad_w;

    const int col_rows = N * OD * OH * OW;
    const int col_cols = IC * KD * KH * KW;

    for (int n = 0; n < N; ++n) {
        for (int od = 0; od < OD; ++od) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int row = ((n * OD + od) * OH + oh) * OW + ow;
                    int col_idx = 0;

                    for (int kd = 0; kd < KD; ++kd) {
                        const int id = od * stride_d - pad_d + kd;
                        if (id < 0 || id >= ID) {
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_bf16[row * col_cols + col_idx++] = bfloat16_t(0.0f);
                                    }
                                }
                            }
                            continue;
                        }

                        for (int kh = 0; kh < KH; ++kh) {
                            const int ih = oh * stride_h - pad_h + kh;
                            if (ih < 0 || ih >= IH) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_bf16[row * col_cols + col_idx++] = bfloat16_t(0.0f);
                                    }
                                }
                                continue;
                            }

                            for (int kw = 0; kw < KW; ++kw) {
                                const int iw = ow * stride_w - pad_w + kw;
                                if (iw < 0 || iw >= IW) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_bf16[row * col_cols + col_idx++] = bfloat16_t(0.0f);
                                    }
                                    continue;
                                }

                                // Convert FP32 input directly to BF16
                                const float* input_ptr = &input[((((n * ID + id) * IH + ih) * IW + iw) * IC)];
                                for (int ic = 0; ic < IC; ++ic) {
                                    col_bf16[row * col_cols + col_idx++] = bfloat16_t(input_ptr[ic]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif  // __ARM_FEATURE_BF16_VECTOR_ARITHMETIC
#endif  // __ARM_NEON

/// BF16 Conv3D: im2col3d + BFMMLA GEMM.
void conv3d_bf16(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op) {
    const auto& hw = detect_arm_hwcaps();
    bool has_bf16 = (hw.hwcaps & static_cast<uint64_t>(HwCap::kBF16)) != 0;

    if (!has_bf16) {
        // Fallback to FP32
        conv3d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    if (p.N <= 0 || p.OC <= 0 || p.IC <= 0) return;

    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int M = p.N * OD * OH * OW;
    const int K = p.IC * p.KD * p.KH * p.KW;
    const int N_gemm = p.OC;

    // Convert filter to BF16 (one-time cost)
    auto filter_bf16 = aligned_array<bfloat16_t>((size_t)N_gemm * K);
    fp32_to_bf16_array_3d(filter, filter_bf16.get(), N_gemm * K);

    // Transpose filter: [OC, K] → [K, OC] in BF16
    auto filter_T_bf16 = aligned_array<bfloat16_t>((size_t)K * N_gemm);
    for (int oc = 0; oc < N_gemm; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T_bf16.get()[k * N_gemm + oc] = filter_bf16.get()[oc * K + k];
        }
    }

    // im2col3d with BF16 conversion
    auto col_bf16 = aligned_array<bfloat16_t>((size_t)M * K);
    im2col_3d_nhwc_bf16(p, input, col_bf16.get());

    // GEMM BF16: output[M, OC] = col_bf16[M, K] × filter_T_bf16[K, OC]
    gemm_bf16_bf16bf16f32(M, N_gemm, K,
              1.0f, col_bf16.get(), K,
              filter_T_bf16.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != Conv3DPostOp::kNone) {
        apply_conv3d_postops(output, M, N_gemm, bias, post_op);
    }
    return;
#endif
#endif

    // Fallback
    conv3d_fp32(p, input, filter, bias, output, post_op);
}

// ============================================================
// INT8 Conv3D Implementation
// ============================================================

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)

/// Compute quantization scale for Conv3D: scale = max_abs / 127.
static float compute_quant_scale_3d(const float* data, size_t n) {
    float max_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float abs_val = std::fabs(data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 127.0f;
}

/// Quantize FP32 array to INT8 with given scale for Conv3D.
static void quantize_fp32_to_int8_3d(const float* src, int8_t* dst, size_t n, float scale) {
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

/// Dequantize INT32 accumulator to FP32 with scales for Conv3D.
static void dequantize_int32_to_fp32_3d(const int32_t* src, float* dst, size_t n,
                                        float input_scale, float filter_scale) {
    float scale = input_scale * filter_scale;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = static_cast<float>(src[i]) * scale;
    }
}

/// im2col3d for INT8 Conv3D: extracts receptive fields and quantizes to INT8.
static void im2col_3d_nhwc_int8(const Conv3DParams& p,
                                const int8_t* input_q,
                                int8_t* col_q) {
    const int N = p.N;
    const int ID = p.ID, IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KD = p.KD, KH = p.KH, KW = p.KW;
    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int stride_d = p.stride_d, stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_d = p.pad_d, pad_h = p.pad_h, pad_w = p.pad_w;

    const int col_rows = N * OD * OH * OW;
    const int col_cols = IC * KD * KH * KW;

    for (int n = 0; n < N; ++n) {
        for (int od = 0; od < OD; ++od) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int row = ((n * OD + od) * OH + oh) * OW + ow;
                    int col_idx = 0;

                    for (int kd = 0; kd < KD; ++kd) {
                        const int id = od * stride_d - pad_d + kd;
                        if (id < 0 || id >= ID) {
                            for (int kh = 0; kh < KH; ++kh) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_q[row * col_cols + col_idx++] = 0;
                                    }
                                }
                            }
                            continue;
                        }

                        for (int kh = 0; kh < KH; ++kh) {
                            const int ih = oh * stride_h - pad_h + kh;
                            if (ih < 0 || ih >= IH) {
                                for (int kw = 0; kw < KW; ++kw) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_q[row * col_cols + col_idx++] = 0;
                                    }
                                }
                                continue;
                            }

                            for (int kw = 0; kw < KW; ++kw) {
                                const int iw = ow * stride_w - pad_w + kw;
                                if (iw < 0 || iw >= IW) {
                                    for (int ic = 0; ic < IC; ++ic) {
                                        col_q[row * col_cols + col_idx++] = 0;
                                    }
                                    continue;
                                }

                                for (int ic = 0; ic < IC; ++ic) {
                                    col_q[row * col_cols + col_idx++] =
                                        input_q[((((n * ID + id) * IH + ih) * IW + iw) * IC + ic)];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif  // __ARM_FEATURE_MATMUL_INT8
#endif  // __ARM_NEON

/// INT8 Conv3D: im2col3d + SMMLA GEMM with dynamic quantization.
void conv3d_int8(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op) {
    const auto& hw = detect_arm_hwcaps();
    bool has_i8mm = (hw.hwcaps & static_cast<uint64_t>(HwCap::kI8MM)) != 0;

    if (!has_i8mm) {
        // Fallback to FP32
        conv3d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

#ifdef __ARM_NEON
#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (p.N <= 0 || p.OC <= 0 || p.IC <= 0) return;

    const int N = p.N;
    const int ID = p.ID, IH = p.IH, IW = p.IW;
    const int IC = p.IC, OC = p.OC;
    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int K = IC * p.KD * p.KH * p.KW;
    const int M = N * OD * OH * OW;

    // Compute quantization scales
    float input_scale = compute_quant_scale_3d(input, N * ID * IH * IW * IC);
    float filter_scale = compute_quant_scale_3d(filter, OC * K);

    // Quantize input and filter to INT8
    auto input_q = aligned_array<int8_t>((size_t)N * ID * IH * IW * IC);
    auto filter_q = aligned_array<int8_t>((size_t)OC * K);

    quantize_fp32_to_int8_3d(input, input_q.get(), N * ID * IH * IW * IC, input_scale);
    quantize_fp32_to_int8_3d(filter, filter_q.get(), OC * K, filter_scale);

    // im2col3d INT8
    auto col_q = aligned_array<int8_t>((size_t)M * K);
    im2col_3d_nhwc_int8(p, input_q.get(), col_q.get());

    // Transpose filter: [OC, K] → [K, OC] in INT8
    auto filter_T_q = aligned_array<int8_t>((size_t)K * OC);
    for (int oc = 0; oc < OC; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T_q.get()[k * OC + oc] = filter_q.get()[oc * K + k];
        }
    }

    // INT32 accumulator buffer
    auto output_acc = aligned_array<int32_t>((size_t)M * OC);

    // Direct INT8 GEMM: output_acc[M, OC] = col_q[M, K] × filter_T_q[K, OC]
    gemm_int8_int8int8int32(M, OC, K, col_q.get(), K, filter_T_q.get(), OC, output_acc.get(), OC);

    // Dequantize to FP32
    float dequant_scale = input_scale * filter_scale;
    for (int i = 0; i < M * OC; ++i) {
        output[i] = static_cast<float>(output_acc.get()[i]) * dequant_scale;
    }

    // Apply bias + post-ops
    if (bias || post_op != Conv3DPostOp::kNone) {
        apply_conv3d_postops(output, M, OC, bias, post_op);
    }
    return;
#endif
#endif

    // Fallback
    conv3d_fp32(p, input, filter, bias, output, post_op);
}

}  // namespace dnnopt