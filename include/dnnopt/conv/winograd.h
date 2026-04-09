#pragma once
/// @file winograd.h
/// Winograd convolution kernels for 3x3 filters.
///
/// Winograd minimal filtering algorithm reduces multiplication count:
///   F(2x2, 3x3): 2.25x fewer multiplications (36 mul -> 16 mul per 2x2 output)
///   F(4x4, 3x3): 1.56x fewer multiplications (144 mul -> 36 mul per 4x4 output)
///
/// Algorithm:
///   1. Input transform:    U = B^T * d * B
///   2. Filter transform:   V = G * g * G^T
///   3. Element-wise mul:   M = U * V (Hadamard product)
///   4. Output transform:   Y = A^T * M * A
///
/// Where:
///   d: input tile (4x4 for F2x2, 6x6 for F4x4)
///   g: 3x3 filter
///   Y: output tile (2x2 for F2x2, 4x4 for F4x4)

#include "dnnopt/conv/conv.h"

namespace dnnopt {

//=============================================================================
// F(2x2, 3x3) Transform Matrices
//=============================================================================

/// Input transform: B^T (4x4)
/// Maps 4x4 input tile to 4x4 transformed space
constexpr float WINO_F22_BT[4][4] = {
    { 1.0f,  0.0f, -1.0f,  0.0f},
    { 0.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f, -1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f,  0.0f, -1.0f}
};

/// Filter transform: G (4x3)
/// Maps 3x3 filter to 4x3 transformed space
constexpr float WINO_F22_G[4][3] = {
    { 1.0f,  0.0f,  0.0f},
    { 0.5f,  0.5f,  0.5f},
    { 0.5f, -0.5f,  0.5f},
    { 0.0f,  0.0f,  1.0f}
};

/// Output transform: A^T (2x4)
/// Maps 4x4 transformed space to 2x2 output tile
constexpr float WINO_F22_AT[2][4] = {
    { 1.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  1.0f}
};

//=============================================================================
// F(4x4, 3x3) Transform Matrices
//=============================================================================

/// Input transform: B^T (6x6)
/// Maps 6x6 input tile to 6x6 transformed space
constexpr float WINO_F44_BT[6][6] = {
    { 4.0f,  0.0f, -5.0f,  0.0f,  1.0f,  0.0f},
    { 0.0f, -4.0f, -4.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  4.0f, -4.0f, -1.0f,  1.0f,  0.0f},
    { 0.0f, -2.0f, -1.0f,  2.0f,  1.0f,  0.0f},
    { 0.0f,  2.0f, -1.0f, -2.0f,  1.0f,  0.0f},
    { 0.0f,  4.0f,  0.0f, -5.0f,  0.0f,  1.0f}
};

/// Filter transform: G (6x3)
/// Maps 3x3 filter to 6x3 transformed space
constexpr float WINO_F44_G[6][3] = {
    { 1.0f/6,  0.0f,    0.0f},
    {-1.0f/6, -1.0f/6, -1.0f/6},
    {-1.0f/6,  1.0f/6, -1.0f/6},
    { 1.0f/24, 1.0f/12, 1.0f/6},
    { 1.0f/24,-1.0f/12, 1.0f/6},
    { 0.0f,    0.0f,    1.0f}
};

/// Output transform: A^T (4x6)
/// Maps 6x6 transformed space to 4x4 output tile
constexpr float WINO_F44_AT[4][6] = {
    { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  2.0f, -2.0f,  1.0f},
    { 0.0f,  1.0f,  1.0f,  4.0f,  4.0f,  0.0f},
    { 0.0f,  1.0f, -1.0f,  8.0f, -8.0f,  1.0f}
};

//=============================================================================
// Winograd Convolution API
//=============================================================================

/// Winograd F(2x2, 3x3) convolution.
/// Processes 2x2 output tiles at a time.
///
/// Requirements:
///   - KH = KW = 3
///   - stride_h = stride_w = 1
///   - pad_h = pad_w = 1
///
/// @param p      Convolution parameters (must satisfy requirements above)
/// @param input  Input tensor [N, IH, IW, IC] (NHWC)
/// @param filter Filter tensor [OC, 3, 3, IC] (OHWI)
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OH, OW, OC] (NHWC)
/// @param post_op Post-operation to apply
void winograd_f2x2_3x3_fp32(const Conv2DParams& p,
                            const float* input,
                            const float* filter,
                            const float* bias,
                            float* output,
                            ConvPostOp post_op = ConvPostOp::kNone);

/// Winograd F(4x4, 3x3) convolution.
/// Processes 4x4 output tiles at a time.
///
/// Requirements:
///   - KH = KW = 3
///   - stride_h = stride_w = 1
///   - pad_h = pad_w = 1
///
/// @param p      Convolution parameters (must satisfy requirements above)
/// @param input  Input tensor [N, IH, IW, IC] (NHWC)
/// @param filter Filter tensor [OC, 3, 3, IC] (OHWI)
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OH, OW, OC] (NHWC)
/// @param post_op Post-operation to apply
void winograd_f4x4_3x3_fp32(const Conv2DParams& p,
                            const float* input,
                            const float* filter,
                            const float* bias,
                            float* output,
                            ConvPostOp post_op = ConvPostOp::kNone);

/// Dispatch to best Winograd variant based on output dimensions.
/// Uses F(4x4,3x3) for large outputs, F(2x2,3x3) for small outputs.
///
/// @param p      Convolution parameters
/// @param input  Input tensor [N, IH, IW, IC] (NHWC)
/// @param filter Filter tensor [OC, 3, 3, IC] (OHWI)
/// @param bias   Bias vector [OC], or nullptr
/// @param output Output tensor [N, OH, OW, OC] (NHWC)
/// @param post_op Post-operation to apply
void winograd_3x3_fp32(const Conv2DParams& p,
                       const float* input,
                       const float* filter,
                       const float* bias,
                       float* output,
                       ConvPostOp post_op = ConvPostOp::kNone);

}  // namespace dnnopt
