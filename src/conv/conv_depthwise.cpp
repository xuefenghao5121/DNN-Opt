/// @file conv_depthwise.cpp
/// Depthwise separable convolution (FP32) with NEON optimization.
///
/// Depthwise convolution: each input channel has its own filter.
/// No cross-channel computation (unlike standard conv).
/// Output channels = Input channels = groups.
///
/// MobileNet/EfficientNet use depthwise + pointwise (1x1 conv) for efficiency:
///   - Standard conv 3x3: IC*OC*9 operations per output pixel
///   - Depthwise + 1x1: IC*9 + IC*OC operations (much fewer for OC >> IC)
///
/// This implementation:
///   - Vectorized per-channel compute (NEON)
///   - No im2col overhead (direct sliding window)
///   - Fused ReLU/ReLU6 post-ops

#include "dnnopt/conv/conv.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

#ifdef __ARM_NEON

// ============================================================
// Depthwise convolution kernels
// ============================================================

/// Depthwise conv 3x3 kernel for a single output pixel.
/// Computes: out[c] = sum over (kh,kw) of input[h+kh,w+kw,c] * filter[kh,kw,c]
static inline float depthwise_3x3_single(
    const float* input, int IH, int IW, int IC,
    const float* filter,  // [IC, 3, 3]
    int oh, int ow, int c,
    int stride_h, int stride_w,
    int pad_h, int pad_w) {

    float acc = 0.0f;
    const int in_h_base = oh * stride_h - pad_h;
    const int in_w_base = ow * stride_w - pad_w;

    for (int kh = 0; kh < 3; ++kh) {
        const int in_h = in_h_base + kh;
        if (in_h < 0 || in_h >= IH) continue;

        for (int kw = 0; kw < 3; ++kw) {
            const int in_w = in_w_base + kw;
            if (in_w < 0 || in_w >= IW) continue;

            // Filter layout: [KH, KW, IC] or [IC, KH, KW]
            // Assuming [IC, KH, KW] = filter[c * 9 + kh * 3 + kw]
            const float in_val = input[(in_h * IW + in_w) * IC + c];
            const float f_val = filter[c * 9 + kh * 3 + kw];
            acc += in_val * f_val;
        }
    }
    return acc;
}

/// Vectorized depthwise 3x3 kernel processing 4 channels at once.
static inline void depthwise_3x3_4ch(
    const float* input, int IH, int IW, int IC,
    const float* filter,
    int oh, int ow, int c_start,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float* output, int OH, int OW) {

    float32x4_t acc = vdupq_n_f32(0.0f);
    const int in_h_base = oh * stride_h - pad_h;
    const int in_w_base = ow * stride_w - pad_w;

    // Load 4 filters
    // filter[c] stored at filter[c * 9 + kh*kw] for c in [c_start, c_start+4)
    for (int kh = 0; kh < 3; ++kh) {
        const int in_h = in_h_base + kh;
        if (in_h < 0 || in_h >= IH) continue;

        for (int kw = 0; kw < 3; ++kw) {
            const int in_w = in_w_base + kw;
            if (in_w < 0 || in_w >= IW) continue;

            // Load 4 input values (consecutive channels)
            float32x4_t in_vals = vld1q_f32(&input[(in_h * IW + in_w) * IC + c_start]);

            // Load 4 filter values
            float f_vals[4];
            for (int ci = 0; ci < 4; ++ci) {
                f_vals[ci] = filter[(c_start + ci) * 9 + kh * 3 + kw];
            }
            float32x4_t f_vec = vld1q_f32(f_vals);

            acc = vfmaq_f32(acc, in_vals, f_vec);
        }
    }

    // Store 4 output values
    vst1q_f32(&output[(oh * OW + ow) * IC + c_start], acc);
}

/// Depthwise conv 3x3 stride=1 pad=1 with full NEON vectorization.
void conv2d_depthwise_3x3_s1p1(
    const Conv2DParams& p,
    const float* input,
    const float* filter,
    float* output) {

    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int OH = p.OH(), OW = p.OW();
    const int stride = 1;
    const int pad = 1;

    // Process each batch
    for (int n = 0; n < N; ++n) {
        const float* in_batch = input + n * IH * IW * IC;
        float* out_batch = output + n * OH * OW * IC;

        // Process each output pixel
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {

                // Vectorized path: process 4 channels at once
                int c = 0;
                for (; c + 3 < IC; c += 4) {
                    depthwise_3x3_4ch(in_batch, IH, IW, IC, filter,
                                      oh, ow, c, stride, stride, pad, pad,
                                      out_batch, OH, OW);
                }

                // Scalar tail: remaining channels
                for (; c < IC; ++c) {
                    float val = depthwise_3x3_single(in_batch, IH, IW, IC, filter,
                                                     oh, ow, c, stride, stride, pad, pad);
                    out_batch[(oh * OW + ow) * IC + c] = val;
                }
            }
        }
    }
}

/// General depthwise convolution kernel.
void conv2d_depthwise_general(
    const Conv2DParams& p,
    const float* input,
    const float* filter,
    float* output) {

    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KH = p.KH, KW = p.KW;
    const int OH = p.OH(), OW = p.OW();
    const int stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_h = p.pad_h, pad_w = p.pad_w;

    // Process each batch
    for (int n = 0; n < N; ++n) {
        const float* in_batch = input + n * IH * IW * IC;
        float* out_batch = output + n * OH * OW * IC;

        // Process each output pixel
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int in_h_base = oh * stride_h - pad_h;
                const int in_w_base = ow * stride_w - pad_w;

                // Process each channel
                int c = 0;
                // Vectorized: process 4 channels at once (if KH*KW allows vectorization)
                if (KH * KW >= 3) {
                    for (; c + 3 < IC; c += 4) {
                        float32x4_t acc = vdupq_n_f32(0.0f);

                        for (int kh = 0; kh < KH; ++kh) {
                            const int in_h = in_h_base + kh;
                            if (in_h < 0 || in_h >= IH) continue;

                            for (int kw = 0; kw < KW; ++kw) {
                                const int in_w = in_w_base + kw;
                                if (in_w < 0 || in_w >= IW) continue;

                                float32x4_t in_vals = vld1q_f32(&in_batch[(in_h * IW + in_w) * IC + c]);

                                float f_vals[4];
                                for (int ci = 0; ci < 4; ++ci) {
                                    f_vals[ci] = filter[(c + ci) * KH * KW + kh * KW + kw];
                                }
                                float32x4_t f_vec = vld1q_f32(f_vals);

                                acc = vfmaq_f32(acc, in_vals, f_vec);
                            }
                        }
                        vst1q_f32(&out_batch[(oh * OW + ow) * IC + c], acc);
                    }
                }

                // Scalar tail
                for (; c < IC; ++c) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int in_h = in_h_base + kh;
                        if (in_h < 0 || in_h >= IH) continue;

                        for (int kw = 0; kw < KW; ++kw) {
                            const int in_w = in_w_base + kw;
                            if (in_w < 0 || in_w >= IW) continue;

                            acc += in_batch[(in_h * IW + in_w) * IC + c] *
                                   filter[c * KH * KW + kh * KW + kw];
                        }
                    }
                    out_batch[(oh * OW + ow) * IC + c] = acc;
                }
            }
        }
    }
}

#endif  // __ARM_NEON

// ============================================================
// Public API
// ============================================================

void conv2d_depthwise_fp32(const Conv2DParams& p,
                            const float* input,
                            const float* filter,
                            const float* bias,
                            float* output,
                            ConvPostOp post_op) {
    // Validate depthwise parameters
    if (!p.is_depthwise()) {
        // Not depthwise - this function should not be called
        // Caller should use conv2d_fp32 instead
        // For safety, we do nothing and return
        return;
    }

#ifdef __ARM_NEON
    // Select specialized kernel
    if (p.KH == 3 && p.KW == 3 && p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 1 && p.pad_w == 1) {
        conv2d_depthwise_3x3_s1p1(p, input, filter, output);
    } else {
        conv2d_depthwise_general(p, input, filter, output);
    }

    // Apply bias + post-ops
    const int OH = p.OH(), OW = p.OW();

    if (bias || post_op != ConvPostOp::kNone) {
        for (int n = 0; n < p.N; ++n) {
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    for (int c = 0; c < p.IC; ++c) {
                        int idx = (n * OH * OW + oh * OW + ow) * p.IC + c;
                        float val = output[idx];

                        if (bias) val += bias[c];

                        switch (post_op) {
                            case ConvPostOp::kRelu:
                                val = val > 0 ? val : 0;
                                break;
                            case ConvPostOp::kRelu6:
                                val = val > 0 ? (val < 6 ? val : 6) : 0;
                                break;
                            default:
                                break;
                        }

                        output[idx] = val;
                    }
                }
            }
        }
    }
#else
    // Fallback: scalar implementation for non-NEON platforms
    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KH = p.KH, KW = p.KW;
    const int OH = p.OH(), OW = p.OW();
    const int stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_h = p.pad_h, pad_w = p.pad_w;

    for (int n = 0; n < N; ++n) {
        const float* in_batch = input + n * IH * IW * IC;
        float* out_batch = output + n * OH * OW * IC;

        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int in_h_base = oh * stride_h - pad_h;
                const int in_w_base = ow * stride_w - pad_w;

                for (int c = 0; c < IC; ++c) {
                    float acc = 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int in_h = in_h_base + kh;
                        if (in_h < 0 || in_h >= IH) continue;

                        for (int kw = 0; kw < KW; ++kw) {
                            const int in_w = in_w_base + kw;
                            if (in_w < 0 || in_w >= IW) continue;

                            acc += in_batch[(in_h * IW + in_w) * IC + c] *
                                   filter[c * KH * KW + kh * KW + kw];
                        }
                    }

                    // Apply bias and post-ops inline
                    if (bias) acc += bias[c];
                    switch (post_op) {
                        case ConvPostOp::kRelu:  acc = acc > 0 ? acc : 0; break;
                        case ConvPostOp::kRelu6: acc = acc > 0 ? (acc < 6 ? acc : 6) : 0; break;
                        default: break;
                    }

                    out_batch[(oh * OW + ow) * IC + c] = acc;
                }
            }
        }
    }
#endif
}

}  // namespace dnnopt