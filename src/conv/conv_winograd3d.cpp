/// @file conv_winograd3d.cpp
/// Winograd F(2x2, 3x3x3) algorithm for Conv3D video convolution.
///
/// Algorithm: Spatial Winograd per temporal slice + temporal accumulation
///   - For each output temporal position (od), compute:
///     Sum over kd of Winograd-spatial convolution at input position id
///   - This reduces spatial multiplications by 2.25x while handling temporal dim
///
/// Benefits:
///   - F(2x2, 3x3x3): 2.25x fewer spatial MACs than im2col3d + GEMM
///   - Efficient for C3D/I3D video models with 3x3x3 kernels
///
/// Conditions:
///   - KD=3, KH=3, KW=3
///   - stride_d=1, stride_h=1, stride_w=1
///   - pad_d=1, pad_h=1, pad_w=1
///   - OH >= 4, OW >= 4 (enough tiles for amortization)

#include "dnnopt/conv/conv3d.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>
#include <cmath>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace dnnopt {

// ============================================================
// Winograd F(2x2, 3x3) spatial transform constants
// ============================================================

// Input transform: B^T * d * B (4x4 input tile → 4x4 transformed)
// B^T = [[1, 0,-1, 0],
//        [0, 1, 1, 0],
//        [0,-1, 1, 0],
//        [0, 1, 0,-1]]
// B = [[1, 0, 0, 0],
//      [0, 1,-1, 1],
//      [-1, 1, 1, 0],
//      [0, 0, 0,-1]]

// Filter transform: G * g * G^T (3x3 filter → 4x4 transformed)
// G = [[  1,   0,   0],
//      [0.5, 0.5, 0.5],
//      [-0.5, 0.5,-0.5],
//      [  0,   0,   1]]

// Output transform: A^T * m * A (4x4 → 2x2)
// A = [[1, 0],
//      [1, 1],
//      [1,-1],
//      [0,-1]]
// A^T = [[1, 1, 1, 0],
//        [0, 1,-1,-1]]

// ============================================================
// Winograd spatial transforms (from conv_winograd.cpp)
// ============================================================

/// Transform 3x3 spatial filter to 4x4 Winograd domain.
static inline void winograd_filter_transform_3x3(
    const float* filter,  // [3x3] row-major
    float* GgGT) {        // [4x4] output (16 elements)

    float g00 = filter[0];
    float g01 = filter[1];
    float g02 = filter[2];
    float g10 = filter[3];
    float g11 = filter[4];
    float g12 = filter[5];
    float g20 = filter[6];
    float g21 = filter[7];
    float g22 = filter[8];

    // G * g (4x3)
    float Gg00 = g00;
    float Gg01 = g01;
    float Gg02 = g02;
    float Gg10 = 0.5f * (g00 + g10 + g20);
    float Gg11 = 0.5f * (g01 + g11 + g21);
    float Gg12 = 0.5f * (g02 + g12 + g22);
    float Gg20 = 0.5f * (g00 - g10 + g20);
    float Gg21 = 0.5f * (g01 - g11 + g21);
    float Gg22 = 0.5f * (g02 - g12 + g22);
    float Gg30 = g20;
    float Gg31 = g21;
    float Gg32 = g22;

    // (G * g) * G^T
    GgGT[0] = Gg00;
    GgGT[1] = 0.5f * (Gg00 + Gg01 + Gg02);
    GgGT[2] = 0.5f * (Gg00 - Gg01 + Gg02);
    GgGT[3] = Gg02;

    GgGT[4] = Gg10;
    GgGT[5] = 0.5f * (Gg10 + Gg11 + Gg12);
    GgGT[6] = 0.5f * (Gg10 - Gg11 + Gg12);
    GgGT[7] = Gg12;

    GgGT[8] = Gg20;
    GgGT[9] = 0.5f * (Gg20 + Gg21 + Gg22);
    GgGT[10] = 0.5f * (Gg20 - Gg21 + Gg22);
    GgGT[11] = Gg22;

    GgGT[12] = Gg30;
    GgGT[13] = 0.5f * (Gg30 + Gg31 + Gg32);
    GgGT[14] = 0.5f * (Gg30 - Gg31 + Gg32);
    GgGT[15] = Gg32;
}

/// Input transform for 4x4 spatial tile.
/// Computes U = B^T * d * B.
/// B = [[  1,   0,  -1,   0],
///      [  0,   1,   1,   0],
///      [  -1,   1,   1,   0],  <- Actually B^T, need to verify
///      [  0,   1,   0,  -1]]
///
/// Standard Winograd F(2x2, 3x3) matrices:
/// B = [[1,  0, -1,  0],
///      [0,  1,  1,  0],
///      [0, -1,  1,  0],
///      [0,  1,  0, -1]]
/// B^T = [[1,  0,  0,  0],
///        [0,  1, -1,  1],
///        [-1, 1,  1,  0],
///        [0,  0,  0, -1]]
static inline void winograd_input_transform_4x4(
    const float* input_tile,  // [4x4] row-major
    float* U) {               // [4x4] output (16 elements)

    // Load 4x4 input tile (d[row][col] indexed as input_tile[row*4 + col])
    float d00 = input_tile[0];
    float d01 = input_tile[1];
    float d02 = input_tile[2];
    float d03 = input_tile[3];
    float d10 = input_tile[4];
    float d11 = input_tile[5];
    float d12 = input_tile[6];
    float d13 = input_tile[7];
    float d20 = input_tile[8];
    float d21 = input_tile[9];
    float d22 = input_tile[10];
    float d23 = input_tile[11];
    float d30 = input_tile[12];
    float d31 = input_tile[13];
    float d32 = input_tile[14];
    float d33 = input_tile[15];

    // B^T * d (row transformation)
    // B^T = [[   1,    0,   -1,    0],
    //        [   0,    1,    1,    0],
    //        [   0,   -1,    1,    0],
    //        [   0,    1,    0,   -1]]
    float Bd[4][4];
    // Row 0 of B^T: [1, 0, -1, 0] -> Bd[0] = d[0] - d[2]
    Bd[0][0] = d00 - d20; Bd[0][1] = d01 - d21; Bd[0][2] = d02 - d22; Bd[0][3] = d03 - d23;
    // Row 1 of B^T: [0, 1, 1, 0] -> Bd[1] = d[1] + d[2]
    Bd[1][0] = d10 + d20; Bd[1][1] = d11 + d21; Bd[1][2] = d12 + d22; Bd[1][3] = d13 + d23;
    // Row 2 of B^T: [0, -1, 1, 0] -> Bd[2] = -d[1] + d[2] = d[2] - d[1]
    Bd[2][0] = d20 - d10; Bd[2][1] = d21 - d11; Bd[2][2] = d22 - d12; Bd[2][3] = d23 - d13;
    // Row 3 of B^T: [0, 1, 0, -1] -> Bd[3] = d[1] - d[3]
    Bd[3][0] = d10 - d30; Bd[3][1] = d11 - d31; Bd[3][2] = d12 - d32; Bd[3][3] = d13 - d33;

    // (B^T * d) * B (column transformation)
    // B = transpose(B^T) = [[   1,    0,    0,    0],
    //                       [   0,    1,   -1,    1],
    //                       [  -1,    1,    1,    0],
    //                       [   0,    0,    0,   -1]]
    // Columns of B (same as rows of B^T due to transpose):
    // Col 0: [1, 0, -1, 0]    -> U[i][0] = Bd[i][0] - Bd[i][2]
    // Col 1: [0, 1, 1, 0]     -> U[i][1] = Bd[i][1] + Bd[i][2]
    // Col 2: [0, -1, 1, 0]    -> U[i][2] = -Bd[i][1] + Bd[i][2]
    // Col 3: [0, 1, 0, -1]    -> U[i][3] = Bd[i][1] - Bd[i][3]
    for (int i = 0; i < 4; ++i) {
        U[i*4 + 0] = Bd[i][0] - Bd[i][2];
        U[i*4 + 1] = Bd[i][1] + Bd[i][2];
        U[i*4 + 2] = -Bd[i][1] + Bd[i][2];
        U[i*4 + 3] = Bd[i][1] - Bd[i][3];
    }
}

/// Output transform for 4x4 → 2x2.
/// Computes out = A^T * M * A.
static inline void winograd_output_transform_2x2(
    const float* M,     // [4x4] transformed output
    float* out_tile) {  // [2x2] output (4 elements)

    // A^T * M (2x4 intermediate)
    float Atm00 = M[0] + M[4] + M[8];
    float Atm01 = M[1] + M[5] + M[9];
    float Atm02 = M[2] + M[6] + M[10];
    float Atm03 = M[3] + M[7] + M[11];
    float Atm10 = M[4] - M[8] - M[12];
    float Atm11 = M[5] - M[9] - M[13];
    float Atm12 = M[6] - M[10] - M[14];
    float Atm13 = M[7] - M[11] - M[15];

    // (A^T * M) * A → 2x2 output
    out_tile[0] = Atm00 + Atm01 + Atm02;
    out_tile[1] = Atm01 - Atm02 - Atm03;
    out_tile[2] = Atm10 + Atm11 + Atm12;
    out_tile[3] = Atm11 - Atm12 - Atm13;
}

// ============================================================
// Conv3D Winograd F(2x2, 3x3x3) implementation
// ============================================================

/// Winograd F(2x2, 3x3x3) Conv3D for stride=1, padding=1.
/// Applies spatial Winograd per temporal slice, then accumulates temporal.
///
/// @param p      Conv3D params (must have KD=KH=KW=3, stride=1, pad=1)
/// @param input  [N, ID, IH, IW, IC] NDHWC layout
/// @param filter [OC, KD, KH, KW, IC] filter
/// @param output [N, OD, OH, OW, OC] NDHWC layout
void conv3d_winograd_3x3x3_s1p1(
    const Conv3DParams& p,
    const float* input,
    const float* filter,
    float* output) {

    const int N = p.N;
    const int ID = p.ID, IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int OC = p.OC;
    const int OD = p.OD(), OH = p.OH(), OW = p.OW();
    const int KD = p.KD;

    // Validate conditions
    if (p.KD != 3 || p.KH != 3 || p.KW != 3) return;  // Only 3x3x3 kernel
    if (p.stride_d != 1 || p.stride_h != 1 || p.stride_w != 1) return;
    if (p.pad_d != 1 || p.pad_h != 1 || p.pad_w != 1) return;

    // Pad input spatial dimensions for tile boundaries
    const int padded_H = ((OH + 1) / 2) * 2 + 2;
    const int padded_W = ((OW + 1) / 2) * 2 + 2;

    // Allocate padded input buffer [N, ID, padded_H, padded_W, IC]
    auto padded_input = aligned_array<float>((size_t)N * ID * padded_H * padded_W * IC);
    std::memset(padded_input.get(), 0, N * ID * padded_H * padded_W * IC * sizeof(float));

    // Copy input with spatial padding (temporal stays same)
    for (int n = 0; n < N; ++n) {
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    for (int ic = 0; ic < IC; ++ic) {
                        padded_input.get()[(((n * ID + id) * padded_H + ih) * padded_W + iw) * IC + ic] =
                            input[((((n * ID + id) * IH + ih) * IW + iw) * IC + ic)];
                    }
                }
            }
        }
    }

    // Temporal padding: for temporal dimension, pad_d=1 means we need
    // access to id = od - 1, od, od + 1 for each output position
    // So we pad temporal at boundaries with zeros

    // Allocate transformed filter buffer [OC, IC, KD, 16]
    // For each (oc, ic, kd), we have a 4x4 transformed spatial filter
    auto transformed_filter = aligned_array<float>((size_t)OC * IC * KD * 16);

    // Transform all spatial filters per temporal position
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            for (int kd = 0; kd < KD; ++kd) {
                // Extract 3x3 spatial filter at this temporal slice
                float g[9];
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        // Filter layout: [OC, KD, KH, KW, IC]
                        g[kh * 3 + kw] = filter[((((oc * KD + kd) * 3 + kh) * 3 + kw) * IC + ic)];
                    }
                }
                // Transform and store
                winograd_filter_transform_3x3(g,
                    &transformed_filter.get()[(((oc * IC + ic) * KD + kd) * 16)]);
            }
        }
    }

    // Process output tiles (2x2 spatial each)
    const int tile_H = (OH + 1) / 2;
    const int tile_W = (OW + 1) / 2;

    // Allocate buffers for tile processing
    auto M_tile = aligned_array<float>((size_t)OC * KD * 16);  // Accumulator per temporal slice
    auto M_sum = aligned_array<float>((size_t)OC * 16);        // Final accumulator across temporal

    for (int n = 0; n < N; ++n) {
        for (int od = 0; od < OD; ++od) {
            for (int th = 0; th < tile_H; ++th) {
                for (int tw = 0; tw < tile_W; ++tw) {
                    const int out_h_base = th * 2;
                    const int out_w_base = tw * 2;

                    // Clear final accumulator
                    std::memset(M_sum.get(), 0, OC * 16 * sizeof(float));

                    // For each temporal kernel position
                    for (int kd = 0; kd < KD; ++kd) {
                        const int id = od - 1 + kd;  // stride_d=1, pad_d=1

                        // Skip if temporal out of bounds (zero-padding)
                        if (id < 0 || id >= ID) continue;

                        // Clear M_tile for this temporal slice
                        std::memset(M_tile.get(), 0, OC * 16 * sizeof(float));

                        // Accumulate over input channels for this temporal slice
                        for (int ic = 0; ic < IC; ++ic) {
                            // Extract 4x4 input tile at this (id, spatial position)
                            // For Winograd F(2x2, 3x3) with pad=1, stride=1:
                            // Input tile for output positions (oh, ow) covers:
                            // oh-1 to oh+2, ow-1 to ow+2 (4x4 tile for 3x3 kernel)
                            float U_tile[16];
                            for (int i = 0; i < 4; ++i) {
                                for (int j = 0; j < 4; ++j) {
                                    const int in_h = out_h_base - 1 + i;  // offset by -1 for pad=1
                                    const int in_w = out_w_base - 1 + j;

                                    if (in_h >= 0 && in_h < IH && in_w >= 0 && in_w < IW) {
                                        U_tile[i * 4 + j] = padded_input.get()
                                            [(((n * ID + id) * padded_H + in_h) * padded_W + in_w) * IC + ic];
                                    } else {
                                        U_tile[i * 4 + j] = 0.0f;  // zero-padding
                                    }
                                }
                            }

                            // Input transform
                            float U[16];
                            winograd_input_transform_4x4(U_tile, U);

                            // Accumulate: M_tile += U * transformed_filter
                            for (int oc = 0; oc < OC; ++oc) {
                                const float* GgGT = &transformed_filter.get()[(((oc * IC + ic) * KD + kd) * 16)];
                                for (int idx = 0; idx < 16; ++idx) {
                                    M_tile.get()[oc * 16 + idx] += U[idx] * GgGT[idx];
                                }
                            }
                        }

                        // Add this temporal slice's contribution to final accumulator
                        for (int oc = 0; oc < OC; ++oc) {
                            for (int idx = 0; idx < 16; ++idx) {
                                M_sum.get()[oc * 16 + idx] += M_tile.get()[oc * 16 + idx];
                            }
                        }
                    }

                    // Output transform for final result
                    for (int oc = 0; oc < OC; ++oc) {
                        float out_tile[4];
                        winograd_output_transform_2x2(&M_sum.get()[oc * 16], out_tile);

                        // Store 2x2 output tile
                        const int out_h0 = out_h_base;
                        const int out_h1 = out_h_base + 1;
                        const int out_w0 = out_w_base;
                        const int out_w1 = out_w_base + 1;

                        if (out_h0 < OH && out_w0 < OW)
                            output[((((n * OD + od) * OH + out_h0) * OW + out_w0) * OC + oc)] = out_tile[0];
                        if (out_h0 < OH && out_w1 < OW)
                            output[((((n * OD + od) * OH + out_h0) * OW + out_w1) * OC + oc)] = out_tile[1];
                        if (out_h1 < OH && out_w0 < OW)
                            output[((((n * OD + od) * OH + out_h1) * OW + out_w0) * OC + oc)] = out_tile[2];
                        if (out_h1 < OH && out_w1 < OW)
                            output[((((n * OD + od) * OH + out_h1) * OW + out_w1) * OC + oc)] = out_tile[3];
                    }
                }
            }
        }
    }
}

/// Dispatch wrapper for Conv3D Winograd.
/// Checks conditions and falls back to im2col3d if not met.
void conv3d_winograd_dispatch(
    const Conv3DParams& p,
    const float* input,
    const float* filter,
    const float* bias,
    float* output,
    Conv3DPostOp post_op) {

    const int OH = p.OH(), OW = p.OW();

    // Dispatch conditions:
    // 1. KD=KH=KW=3 (3x3x3 kernel)
    // 2. stride=1, pad=1
    // 3. OH >= 4, OW >= 4 (enough tiles for amortization)
    bool can_winograd = (p.KD == 3 && p.KH == 3 && p.KW == 3) &&
                         (p.stride_d == 1 && p.stride_h == 1 && p.stride_w == 1) &&
                         (p.pad_d == 1 && p.pad_h == 1 && p.pad_w == 1) &&
                         (OH >= 4 && OW >= 4);

    if (can_winograd) {
        conv3d_winograd_3x3x3_s1p1(p, input, filter, output);

        // Apply bias + post-ops
        if (bias || post_op != Conv3DPostOp::kNone) {
            const int M = p.N * p.OD() * OH * OW;
            for (int idx = 0; idx < M; ++idx) {
                for (int oc = 0; oc < p.OC; ++oc) {
                    float val = output[idx * p.OC + oc];
                    if (bias) val += bias[oc];
                    switch (post_op) {
                        case Conv3DPostOp::kRelu:
                            val = val > 0 ? val : 0;
                            break;
                        case Conv3DPostOp::kRelu6:
                            val = val > 0 ? (val < 6 ? val : 6) : 0;
                            break;
                        default:
                            break;
                    }
                    output[idx * p.OC + oc] = val;
                }
            }
        }
    } else {
        // Fallback to im2col3d + GEMM
        conv3d_fp32(p, input, filter, bias, output, post_op);
    }
}

}  // namespace dnnopt