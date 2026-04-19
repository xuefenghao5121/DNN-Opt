/// @file conv_grouped.cpp
/// Grouped convolution implementation for ResNeXt, ShuffleNet.
///
/// Grouped convolution splits channels into groups:
///   - Input: IC channels → groups × (IC/groups) channels per group
///   - Output: OC channels → groups × (OC/groups) channels per group
///   - Each group independently computes: in_group[g] × filter_group[g] → out_group[g]
///
/// Filter layout: [groups, OC/groups, KH, KW, IC/groups]
///   - Can also be [OC, KH, KW, IC] with implicit grouping
///
/// Use cases:
///   - ResNeXt: groups=32, IC=256, OC=256 → 8 channels per group
///   - ShuffleNet: groups=g, followed by channel shuffle
///
/// This implementation:
///   - Loops over groups, calls optimized kernels per group
///   - Could be further optimized with fused batch processing

#include "dnnopt/conv/conv.h"
#include "dnnopt/gemm/gemm.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>

namespace dnnopt {

// Forward declaration of internal helpers
void im2col_nhwc(const Conv2DParams& p, const float* input, float* col);
void apply_conv_postops(float* output, int num_rows, int OC,
                        const float* bias, ConvPostOp op);

#ifdef __ARM_NEON
void conv2d_winograd_3x3_s1p1(const Conv2DParams& p,
                               const float* input,
                               const float* filter,
                               float* output);
void conv2d_winograd_4x4_3x3_s1p1(const Conv2DParams& p,
                                   const float* input,
                                   const float* filter,
                                   float* output);
#endif

/// Transpose filter from [OC_g, K] to [K, OC_g] for GEMM B matrix.
static void transpose_filter_group(const float* filter, float* filter_T,
                                    int OC_g, int K) {
    for (int oc = 0; oc < OC_g; ++oc) {
        for (int k = 0; k < K; ++k) {
            filter_T[k * OC_g + oc] = filter[oc * K + k];
        }
    }
}

/// im2col for a single group.
/// Extracts input [N, OH, OW, IC_g] from full input [N, IH, IW, IC].
static void im2col_group_nhwc(const Conv2DParams& p, int group_id,
                               const float* input, float* col) {
    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int KH = p.KH, KW = p.KW;
    const int OH = p.OH(), OW = p.OW();
    const int stride_h = p.stride_h, stride_w = p.stride_w;
    const int pad_h = p.pad_h, pad_w = p.pad_w;
    const int groups = p.groups;

    const int IC_g = IC / groups;  // Channels per group
    const int ic_start = group_id * IC_g;

    const int col_rows = N * OH * OW;
    const int col_cols = IC_g * KH * KW;

    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                const int row = (n * OH + oh) * OW + ow;
                int col_idx = 0;

                for (int kh = 0; kh < KH; ++kh) {
                    const int ih = oh * stride_h - pad_h + kh;
                    if (ih < 0 || ih >= IH) {
                        // Zero-padding for out-of-bounds
                        for (int kw = 0; kw < KW; ++kw) {
                            for (int ic = 0; ic < IC_g; ++ic) {
                                col[row * col_cols + col_idx++] = 0.0f;
                            }
                        }
                        continue;
                    }

                    for (int kw = 0; kw < KW; ++kw) {
                        const int iw = ow * stride_w - pad_w + kw;
                        if (iw < 0 || iw >= IW) {
                            // Zero-padding for out-of-bounds
                            for (int ic = 0; ic < IC_g; ++ic) {
                                col[row * col_cols + col_idx++] = 0.0f;
                            }
                            continue;
                        }

                        // Copy input values for this group's channels
                        for (int ic = 0; ic < IC_g; ++ic) {
                            col[row * col_cols + col_idx++] =
                                input[((n * IH + ih) * IW + iw) * IC + (ic_start + ic)];
                        }
                    }
                }
            }
        }
    }
}

/// Grouped convolution via per-group dispatch.
/// Each group is processed independently, reducing cross-group data movement.
void conv2d_grouped_fp32(const Conv2DParams& p,
                          const float* input,
                          const float* filter,
                          const float* bias,
                          float* output,
                          ConvPostOp post_op) {
    if (!p.is_grouped()) {
        // Not grouped, fall back to standard conv
        conv2d_fp32(p, input, filter, bias, output, post_op);
        return;
    }

    const int N = p.N;
    const int OH = p.OH(), OW = p.OW();
    const int groups = p.groups;
    const int IC = p.IC, OC = p.OC;
    const int IC_g = IC / groups;  // Input channels per group
    const int OC_g = OC / groups;  // Output channels per group
    const int KH = p.KH, KW = p.KW;
    const int K = IC_g * KH * KW;  // Flattened receptive field per group

    // Process each group
    for (int g = 0; g < groups; ++g) {
        // Filter offset for this group: [groups, OC_g, KH, KW, IC_g]
        // Or standard layout with implicit grouping: [OC, KH, KW, IC]
        // Assuming grouped layout: filter + g * OC_g * KH * KW * IC_g
        const float* filter_g = filter + g * OC_g * K;

        // Bias offset for this group (if present)
        const float* bias_g = bias ? (bias + g * OC_g) : nullptr;

        // Output offset for this group
        float* output_g = output + g * OC_g;

        // Check for 1x1 conv optimization (no im2col needed)
        if (KH == 1 && KW == 1 && p.stride_h == 1 && p.stride_w == 1 &&
            p.pad_h == 0 && p.pad_w == 0) {
            // Direct GEMM: input_group[M, IC_g] × filter^T[IC_g, OC_g]
            const int M = N * OH * OW;

            // Transpose filter
            auto filter_T = aligned_array<float>((size_t)K * OC_g);
            transpose_filter_group(filter_g, filter_T.get(), OC_g, K);

            // Extract input for this group
            // NHWC layout: input[..., ic_start:ic_start+IC_g]
            // For GEMM, we need contiguous input
            // Input is already in correct layout if we adjust the GEMM parameters
            // GEMM: C_g[M, OC_g] = A_g[M, IC_g] × B_g[IC_g, OC_g]

            // Special handling: input stride for this group
            // A: input[n*IH*IW*IC + ... + ic_start:ic_start+IC_g]
            // Need to adjust lda for GEMM

            // Simple approach: extract contiguous input for this group
            auto input_g = aligned_array<float>((size_t)M * IC_g);
            for (int n = 0; n < N; ++n) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        int row = (n * OH + oh) * OW + ow;
                        const float* in_ptr = &input[((n * OH + oh) * OW + ow) * IC + g * IC_g];
                        for (int ic = 0; ic < IC_g; ++ic) {
                            input_g.get()[row * IC_g + ic] = in_ptr[ic];
                        }
                    }
                }
            }

            gemm_fp32(M, OC_g, IC_g,
                      1.0f, input_g.get(), IC_g,
                      filter_T.get(), OC_g,
                      0.0f, output_g, OC);  // Note: output stride is OC (full)

            continue;
        }

#ifdef __ARM_NEON
        // Winograd path for 3x3 stride=1 pad=1 (per group)
        if (KH == 3 && KW == 3 && p.stride_h == 1 && p.stride_w == 1 &&
            p.pad_h == 1 && p.pad_w == 1) {
            // Create per-group Conv2DParams
            Conv2DParams p_g = p;
            p_g.IC = IC_g;
            p_g.OC = OC_g;

            // Extract input for this group (contiguous IC_g channels)
            auto input_g = aligned_array<float>((size_t)N * p.IH * p.IW * IC_g);
            for (int idx = 0; idx < N * p.IH * p.IW; ++idx) {
                for (int ic = 0; ic < IC_g; ++ic) {
                    input_g.get()[idx * IC_g + ic] = input[idx * IC + g * IC_g + ic];
                }
            }

            // Remap filter to OIHW layout [OC_g, KH, KW, IC_g]
            // Explicit grouped layout: [groups, OC_g, KH, KW, IC_g]
            // Winograd expects: [OC_g, KH, KW, IC_g]
            auto filter_g_oihw = aligned_array<float>((size_t)OC_g * KH * KW * IC_g);
            for (int oc_g = 0; oc_g < OC_g; ++oc_g) {
                for (int kh = 0; kh < KH; ++kh) {
                    for (int kw = 0; kw < KW; ++kw) {
                        for (int ic_g = 0; ic_g < IC_g; ++ic_g) {
                            // Explicit grouped: [g, oc_g, kh, kw, ic_g]
                            // Index: g * OC_g * K + oc_g * K + kh * KW * IC_g + kw * IC_g + ic_g
                            // Where K = KH * KW * IC_g
                            int src_idx = oc_g * K + kh * KW * IC_g + kw * IC_g + ic_g;
                            // OIHW: [oc_g, kh, kw, ic_g]
                            // Index: ((oc_g * KH + kh) * KW + kw) * IC_g + ic_g
                            int dst_idx = ((oc_g * KH + kh) * KW + kw) * IC_g + ic_g;
                            filter_g_oihw.get()[dst_idx] = filter_g[src_idx];
                        }
                    }
                }
            }

            // Temporary output buffer for this group (will scatter to final output)
            auto output_g_buf = aligned_array<float>((size_t)N * OH * OW * OC_g);

            if (OH >= 16 && OW >= 16) {
                conv2d_winograd_4x4_3x3_s1p1(p_g, input_g.get(), filter_g_oihw.get(), output_g_buf.get());
            } else if (OH >= 8 && OW >= 8) {
                conv2d_winograd_3x3_s1p1(p_g, input_g.get(), filter_g_oihw.get(), output_g_buf.get());
            } else {
                // Fallback: im2col + GEMM for small spatial dims
                auto col = aligned_array<float>((size_t)N * OH * OW * K);
                im2col_group_nhwc(p, g, input, col.get());

                auto filter_T = aligned_array<float>((size_t)K * OC_g);
                transpose_filter_group(filter_g, filter_T.get(), OC_g, K);

                gemm_fp32(N * OH * OW, OC_g, K,
                          1.0f, col.get(), K,
                          filter_T.get(), OC_g,
                          0.0f, output_g_buf.get(), OC_g);
            }

            // Scatter output to final position
            for (int idx = 0; idx < N * OH * OW; ++idx) {
                for (int oc = 0; oc < OC_g; ++oc) {
                    output[idx * OC + g * OC_g + oc] = output_g_buf.get()[idx * OC_g + oc];
                }
            }

            continue;
        }
#endif

        // General path: im2col + GEMM for each group
        {
            const int M = N * OH * OW;

            // im2col for this group
            auto col = aligned_array<float>((size_t)M * K);
            im2col_group_nhwc(p, g, input, col.get());

            // Transpose filter
            auto filter_T = aligned_array<float>((size_t)K * OC_g);
            transpose_filter_group(filter_g, filter_T.get(), OC_g, K);

            // GEMM: output[M, OC_g] = col[M, K] × filter_T[K, OC_g]
            // But we need to scatter output to correct position
            auto output_g_buf = aligned_array<float>((size_t)M * OC_g);
            gemm_fp32(M, OC_g, K,
                      1.0f, col.get(), K,
                      filter_T.get(), OC_g,
                      0.0f, output_g_buf.get(), OC_g);

            // Scatter to final output [N*OH*OW, OC]
            for (int idx = 0; idx < M; ++idx) {
                for (int oc = 0; oc < OC_g; ++oc) {
                    output[idx * OC + g * OC_g + oc] = output_g_buf.get()[idx * OC_g + oc];
                }
            }
        }
    }

    // Apply bias + post-ops
    if (bias || post_op != ConvPostOp::kNone) {
        const int M = N * OH * OW;
        apply_conv_postops(output, M, OC, bias, post_op);
    }
}

}  // namespace dnnopt