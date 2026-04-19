/// @file conv2d.cpp
/// Conv2D dispatch: im2col + GEMM for general convolutions,
/// direct GEMM for 1×1 pointwise convolutions,
/// Winograd F(2x2, 3x3) for 3x3 stride=1 convolutions,
/// Depthwise kernel for depthwise separable convolutions.

#include "dnnopt/conv/conv.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>

namespace dnnopt {

// Forward declarations
void im2col_nhwc(const Conv2DParams& p, const float* input, float* col);
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op);

#ifdef __ARM_NEON
// Winograd F(2x2, 3x3) for stride=1, padding=1
void conv2d_winograd_3x3_s1p1(const Conv2DParams& p,
                               const float* input,
                               const float* filter,
                               float* output);

// Winograd F(4x4, 3x3) for stride=1, padding=1 (larger tiles)
void conv2d_winograd_4x4_3x3_s1p1(const Conv2DParams& p,
                                   const float* input,
                                   const float* filter,
                                   float* output);

// Depthwise convolution kernels
void conv2d_depthwise_3x3_s1p1(const Conv2DParams& p,
                                const float* input,
                                const float* filter,
                                float* output);
void conv2d_depthwise_general(const Conv2DParams& p,
                                const float* input,
                                const float* filter,
                                float* output);
#endif

// Grouped convolution kernel (defined in conv_grouped.cpp)
void conv2d_grouped_fp32(const Conv2DParams& p,
                          const float* input,
                          const float* filter,
                          const float* bias,
                          float* output,
                          ConvPostOp post_op);

/// Transpose filter from [OC, K] to [K, OC] for GEMM B matrix.
static void transpose_filter(const float* filter, float* filter_T,
                              int OC, int K) {
    for (int oc = 0; oc < OC; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T[k * OC + oc] = filter[oc * K + k];
        }
    }
}

/// Direct 1×1 convolution: input [N*H*W, IC] × filter^T [IC, OC] → output [N*H*W, OC]
/// No im2col needed — NHWC input is already in GEMM-ready layout.
static void conv2d_1x1_direct(const Conv2DParams& p,
                               const float* input,
                               const float* filter,
                               const float* bias,
                               float* output,
                               ConvPostOp post_op) {
    const int M = p.N * p.IH * p.IW;  // For 1×1 s1 p0: OH=IH, OW=IW
    const int K = p.IC;
    const int N_gemm = p.OC;

    // Transpose filter once: [OC, IC] → [IC, OC]
    auto filter_T = aligned_array<float>((size_t)K * N_gemm);
    transpose_filter(filter, filter_T.get(), N_gemm, K);

    // GEMM: C[M, OC] = A[M, IC] × B[IC, OC]
    gemm_fp32(M, N_gemm, K,
              1.0f, input, K,
              filter_T.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        apply_conv_postops(output, M, N_gemm, bias, post_op);
    }
}

/// im2col + GEMM convolution for general kernels.
static void conv2d_im2col_gemm(const Conv2DParams& p,
                                const float* input,
                                const float* filter,
                                const float* bias,
                                float* output,
                                ConvPostOp post_op) {
    const int OH = p.OH(), OW = p.OW();
    const int M = p.N * OH * OW;           // Output spatial elements
    const int K = p.IC * p.KH * p.KW;     // Flattened receptive field
    const int N_gemm = p.OC;

    // Allocate im2col buffer
    auto col = aligned_array<float>((size_t)M * K);

    // im2col: input [N,IH,IW,IC] → col [M, K]
    im2col_nhwc(p, input, col.get());

    // Transpose filter: [OC, K] → [K, OC]
    auto filter_T = aligned_array<float>((size_t)K * N_gemm);
    transpose_filter(filter, filter_T.get(), N_gemm, K);

    // GEMM: output[M, OC] = col[M, K] × filter_T[K, OC]
    gemm_fp32(M, N_gemm, K,
              1.0f, col.get(), K,
              filter_T.get(), N_gemm,
              0.0f, output, N_gemm);

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        apply_conv_postops(output, M, N_gemm, bias, post_op);
    }
}

void conv2d_fp32(const Conv2DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 ConvPostOp post_op) {
#ifdef __ARM_NEON
    // Depthwise path: groups=IC, OC=IC
    // Use specialized kernel that avoids cross-channel GEMM
    if (p.is_depthwise()) {
        if (p.KH == 3 && p.KW == 3 && p.stride_h == 1 && p.stride_w == 1 &&
            p.pad_h == 1 && p.pad_w == 1) {
            conv2d_depthwise_3x3_s1p1(p, input, filter, output);
        } else {
            conv2d_depthwise_general(p, input, filter, output);
        }

        // Apply bias + post-ops
        if (bias || post_op != ConvPostOp::kNone) {
            const int M = p.N * p.OH() * p.OW();
            apply_conv_postops(output, M, p.OC, bias, post_op);
        }
        return;
    }

    // Grouped path: groups > 1, groups < IC
    // Used in ResNeXt, ShuffleNet for efficiency
    if (p.is_grouped()) {
        conv2d_grouped_fp32(p, input, filter, bias, output, post_op);
        return;
    }
#endif

    // Fast path: 1×1 conv with stride=1, no padding
    if (p.KH == 1 && p.KW == 1 &&
        p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 0 && p.pad_w == 0) {
        conv2d_1x1_direct(p, input, filter, bias, output, post_op);
        return;
    }

#ifdef __ARM_NEON
    // Winograd path: 3×3 conv with stride=1, padding=1
    // Reduces multiplications:
    //   - F(2x2, 3x3): 2.25x fewer (9 → 4)
    // F(4x4, 3x3) is disabled pending correct transform matrices
    if (p.KH == 3 && p.KW == 3 &&
        p.stride_h == 1 && p.stride_w == 1 &&
        p.pad_h == 1 && p.pad_w == 1) {
        const int OH = p.OH(), OW = p.OW();

        if (OH >= 8 && OW >= 8) {
            // F(2x2, 3x3): 2.25x fewer multiplications
            conv2d_winograd_3x3_s1p1(p, input, filter, output);
        } else {
            // Small spatial dims: im2col is better (Winograd overhead not amortized)
            conv2d_im2col_gemm(p, input, filter, bias, output, post_op);
            return;
        }

        // Apply bias + post-ops after Winograd
        if (bias || post_op != ConvPostOp::kNone) {
            const int M = p.N * p.OH() * p.OW();
            apply_conv_postops(output, M, p.OC, bias, post_op);
        }
        return;
    }
#endif

    // General path: im2col + GEMM
    conv2d_im2col_gemm(p, input, filter, bias, output, post_op);
}

}  // namespace dnnopt
