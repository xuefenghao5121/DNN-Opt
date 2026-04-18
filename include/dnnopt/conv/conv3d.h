#pragma once
/// @file conv3d.h
/// Public Conv3D API for DNN-Opt (video processing).
///
/// Data layouts:
///   input:  [N, ID, IH, IW, IC] (NDHWC)
///   filter: [OC, KD, KH, KW, IC] (ODIHW with IC innermost)
///   bias:   [OC] or nullptr
///   output: [N, OD, OH, OW, OC] (NDHWC)

#include <cstddef>

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

    /// Output temporal depth.
    int OD() const { return (ID + 2 * pad_d - KD) / stride_d + 1; }

    /// Output height.
    int OH() const { return (IH + 2 * pad_h - KH) / stride_h + 1; }

    /// Output width.
    int OW() const { return (IW + 2 * pad_w - KW) / stride_w + 1; }
};

/// Post-operation applied after Conv3D.
enum class Conv3DPostOp {
    kNone,       // No post-op
    kRelu,       // max(0, x)
    kRelu6,      // min(6, max(0, x))
};

/// FP32 Conv3D with Winograd F(2x2, 3x3x3) optimization.
/// Dispatches to Winograd when KD=KH=KW=3, stride=1, pad=1, OH>=4, OW>=4.
/// Falls back to im2col3d + GEMM for other configurations.
///
/// @param p      Conv3D parameters
/// @param input  Input tensor [N, ID, IH, IW, IC] (NDHWC)
/// @param filter Filter tensor [OC, KD, KH, KW, IC]
/// @param bias   Bias vector [OC], or nullptr for no bias
/// @param output Output tensor [N, OD, OH, OW, OC] (NDHWC)
/// @param post_op Post-operation to apply
void conv3d_fp32(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op = Conv3DPostOp::kNone);

/// Winograd F(2x2, 3x3x3) Conv3D for stride=1, padding=1.
/// Applies spatial Winograd per temporal slice, then accumulates temporal.
/// Reduces spatial multiplications by 2.25x.
///
/// @param p      Conv3D params (must have KD=KH=KW=3, stride=1, pad=1)
/// @param input  [N, ID, IH, IW, IC] NDHWC layout
/// @param filter [OC, KD, KH, KW, IC] filter
/// @param output [N, OD, OH, OW, OC] NDHWC layout
void conv3d_winograd_3x3x3_s1p1(
    const Conv3DParams& p,
    const float* input,
    const float* filter,
    float* output);

/// Winograd dispatch wrapper for Conv3D.
/// Checks conditions and falls back to im2col3d if not met.
///
/// @param p      Conv3D parameters
/// @param input  Input tensor [N, ID, IH, IW, IC] (NDHWC)
/// @param filter Filter tensor [OC, KD, KH, KW, IC]
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OD, OH, OW, OC] (NDHWC)
/// @param post_op Post-operation to apply
void conv3d_winograd_dispatch(
    const Conv3DParams& p,
    const float* input,
    const float* filter,
    const float* bias,
    float* output,
    Conv3DPostOp post_op);

/// BF16 Conv3D: FP32 input/filter converted to BF16 for compute.
/// Uses BFMMLA for higher compute density (2x vs FP32).
///
/// @param p      Conv3D parameters
/// @param input  Input tensor [N, ID, IH, IW, IC] (NDHWC, FP32)
/// @param filter Filter tensor [OC, KD, KH, KW, IC] (FP32)
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OD, OH, OW, OC] (NDHWC, FP32)
/// @param post_op Post-operation to apply
void conv3d_bf16(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op = Conv3DPostOp::kNone);

/// INT8 Conv3D: FP32 input/filter dynamically quantized to INT8.
/// Uses SMMLA for maximum compute density (4x vs FP32).
/// Per-tensor quantization with dynamic scale computation.
///
/// @param p      Conv3D parameters
/// @param input  Input tensor [N, ID, IH, IW, IC] (NDHWC, FP32)
/// @param filter Filter tensor [OC, KD, KH, KW, IC] (FP32)
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OD, OH, OW, OC] (NDHWC, FP32)
/// @param post_op Post-operation to apply
void conv3d_int8(const Conv3DParams& p,
                 const float* input,
                 const float* filter,
                 const float* bias,
                 float* output,
                 Conv3DPostOp post_op = Conv3DPostOp::kNone);

}  // namespace dnnopt