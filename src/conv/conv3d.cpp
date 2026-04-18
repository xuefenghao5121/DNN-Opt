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
///   - Optimized paths for common shapes (1x3x3, 3x3x3 stride=1)

#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>
#include <cmath>

namespace dnnopt {

/// Conv3D parameters.
struct Conv3DParams {
    int N;             // Batch size
    int IC;            // Input channels
    int ID, IH, IW;   // Input: temporal depth, height, width
    int OC;            // Output channels
    int KD, KH, KW;   // Kernel: temporal, height, width
    int stride_d, stride_h, stride_w;
    int pad_d, pad_h, pad_w;

    int OD() const { return (ID + 2 * pad_d - KD) / stride_d + 1; }
    int OH() const { return (IH + 2 * pad_h - KH) / stride_h + 1; }
    int OW() const { return (IW + 2 * pad_w - KW) / stride_w + 1; }
};

/// Post-operation applied after Conv3D.
enum class Conv3DPostOp {
    kNone,
    kRelu,
    kRelu6,
};

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
                 Conv3DPostOp post_op = Conv3DPostOp::kNone) {
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

}  // namespace dnnopt