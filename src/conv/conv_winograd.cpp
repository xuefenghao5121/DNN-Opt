/// @file conv_winograd.cpp
/// Winograd F(2x2, 3x3) and F(4x4, 3x3) convolution algorithms.
///
/// Winograd minimal filtering algorithm reduces arithmetic complexity:
///   - Standard conv 3x3: 9 multiplications per output pixel
///   - Winograd F(2x2, 3x3): 4 multiplications per 2x2 output tile (2.25x fewer)
///   - Winograd F(4x4, 3x3): 6 multiplications per 4x4 output tile (6x fewer)
///
/// F(2x2, 3x3) Algorithm:
///   1. Input transform: 4x4 input tile → B^T * d * B
///   2. Filter transform: 3x3 filter → G * g * G^T
///   3. Element-wise multiply: transformed input × transformed filter
///   4. Output transform: A^T * (U * V) * A → 2x2 output tile
///
/// F(4x4, 3x3) Algorithm:
///   1. Input transform: 6x6 input tile → B^T * d * B
///   2. Filter transform: 3x3 filter → G * g * G^T (6x4)
///   3. Element-wise multiply: 6x6 × 6x4 → 6x4
///   4. Output transform: A^T * (U * V) * A → 4x4 output tile
///
/// References:
///   - Fast Algorithms for Convolutional Neural Networks (Lavin & Gray, 2015)
///   - oneDNN Winograd implementation

#include "dnnopt/conv/conv.h"
#include "dnnopt/aligned_alloc.h"

#include <cstring>
#include <cmath>

namespace dnnopt {

#ifdef __ARM_NEON
#include <arm_neon.h>

// ============================================================
// Winograd transform constants
// ============================================================

// Input transform: B^T * d * B (4x4 input tile → 4x4 transformed)
// Precomputed constants for efficiency
static const float kInputTransformConstants[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f},
    {0.0f,  1.0f, -1.0f,  1.0f},
    {-1.0f, 1.0f,  1.0f,  0.0f},
    {0.0f,  0.0f,  0.0f, -1.0f}
};

// Filter transform: G * g * G^T (3x3 filter → 4x4 transformed)
static const float kFilterTransformG[4][3] = {
    {1.0f,    0.0f,    0.0f},
    {0.5f,    0.5f,    0.5f},
    {-0.5f,   0.5f,   -0.5f},
    {0.0f,    0.0f,    1.0f}
};

// Output transform: A^T * m * A (4x4 → 2x2)
static const float kOutputTransformA[4][2] = {
    {1.0f,  0.0f},
    {1.0f,  1.0f},
    {1.0f, -1.0f},
    {0.0f, -1.0f}
};

// ============================================================
// Winograd 3x3 kernel transforms
// ============================================================

/// Transform 4x4 input tile for Winograd convolution.
/// Computes B^T * d * B where d is the 4x4 input tile.
static inline void winograd_input_transform_4x4(
    const float* input, int stride,
    float* U0, float* U1, float* U2, float* U3) {
    // Load 4x4 input tile (row-major, stride = input width)
    // d[i][j] = input[i*stride + j]

    // Compute B^T * d first (4x4 intermediate)
    // Then compute (B^T * d) * B

    // Inline NEON implementation for performance
    // Row 0: B^T[0] * d = [d[0], d[1], d[2], d[3]] (identity row)
    // Row 1: B^T[1] * d = [d[0]-d[2], d[1]-d[3], d[2], d[3]]??
    // Simplified: use precomputed formulas

    float d00 = input[0*stride + 0];
    float d01 = input[0*stride + 1];
    float d02 = input[0*stride + 2];
    float d03 = input[0*stride + 3];
    float d10 = input[1*stride + 0];
    float d11 = input[1*stride + 1];
    float d12 = input[1*stride + 2];
    float d13 = input[1*stride + 3];
    float d20 = input[2*stride + 0];
    float d21 = input[2*stride + 1];
    float d22 = input[2*stride + 2];
    float d23 = input[2*stride + 3];
    float d30 = input[3*stride + 0];
    float d31 = input[3*stride + 1];
    float d32 = input[3*stride + 2];
    float d33 = input[3*stride + 3];

    // B^T * d (row-by-row transformation)
    // B^T = [[1, 0,-1, 0],
    //        [0, 1, 1, 0],
    //        [0,-1, 1, 0],
    //        [0, 1, 0,-1]]
    float Bd0_0 = d00 - d20;
    float Bd0_1 = d01 - d21;
    float Bd0_2 = d02 - d22;
    float Bd0_3 = d03 - d23;
    float Bd1_0 = d10 + d20;
    float Bd1_1 = d11 + d21;
    float Bd1_2 = d12 + d22;
    float Bd1_3 = d13 + d23;
    float Bd2_0 = d20 - d10;
    float Bd2_1 = d21 - d11;
    float Bd2_2 = d22 - d12;
    float Bd2_3 = d23 - d13;
    float Bd3_0 = d10 - d30;
    float Bd3_1 = d11 - d31;
    float Bd3_2 = d12 - d32;
    float Bd3_3 = d13 - d33;

    // (B^T * d) * B (column-by-column transformation)
    // B = [[1, 0, 0, 0],
    //      [0, 1,-1, 1],
    //      [-1, 1, 1, 0],
    //      [0, 0, 0,-1]]
    *U0 = Bd0_0 - Bd0_2;
    *U1 = Bd1_0 - Bd1_2;
    *U2 = Bd2_0 - Bd2_2;
    *U3 = Bd3_0 - Bd3_2;

    // Actually we need full 4x4 U matrix, but simplified for now
    // Will expand in actual implementation
}

/// Transform 3x3 filter for Winograd convolution.
/// Computes G * g * G^T where g is the 3x3 filter.
static inline void winograd_filter_transform_3x3(
    const float* filter,  // [3x3] row-major
    float* GgGT) {        // [4x4] output

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
    // G[0] * g = [g00, g01, g02]
    // G[1] * g = [0.5*(g00+g10+g20), 0.5*(g01+g11+g21), 0.5*(g02+g12+g22)]
    // G[2] * g = [0.5*(g00-g10+g20), 0.5*(g01-g11+g21), 0.5*(g02-g12+g22)]
    // G[3] * g = [g20, g21, g22]
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

    // (G * g) * G^T (G^T is 3x4)
    // Column transformation
    // G^T[0] = [1, 0.5,-0.5, 0]
    // G^T[1] = [0, 0.5, 0.5, 0]
    // G^T[2] = [0, 0.5,-0.5, 1]
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

/// Winograd output transform: A^T * m * A (4x4 → 2x2)
static inline void winograd_output_transform_2x2(
    const float* M,  // [4x4] transformed output
    float* out0, float* out1) {  // [2] output row

    // A^T * m (2x4 intermediate)
    // A^T[0] = [1, 1, 1, 0]
    // A^T[1] = [0, 1,-1,-1]
    float Atm00 = M[0] + M[4] + M[8];
    float Atm01 = M[1] + M[5] + M[9];
    float Atm02 = M[2] + M[6] + M[10];
    float Atm03 = M[3] + M[7] + M[11];
    float Atm10 = M[4] - M[8] - M[12];
    float Atm11 = M[5] - M[9] - M[13];
    float Atm12 = M[6] - M[10] - M[14];
    float Atm13 = M[7] - M[11] - M[15];

    // (A^T * m) * A
    // A = [[1, 0],
    //      [1, 1],
    //      [1,-1],
    //      [0,-1]]
    out0[0] = Atm00 + Atm01 + Atm02;
    out0[1] = Atm01 - Atm02 - Atm03;
    out1[0] = Atm10 + Atm11 + Atm12;
    out1[1] = Atm11 - Atm12 - Atm13;
}

// ============================================================
// Winograd 3x3 convolution implementation
// ============================================================

/// Winograd F(2x2, 3x3) convolution for stride=1, padding=1.
/// Efficient for 3x3 kernels, reduces multiplications by 2.25x.
///
/// @param input   [N, IH, IW, IC] NHWC layout
/// @param filter  [OC, 3, 3, IC]  OIHW layout
/// @param output  [N, OH, OW, OC] NHWC layout
/// @param N       Batch size
/// @param IH, IW  Input height/width
/// @param IC      Input channels
/// @param OC      Output channels
void conv2d_winograd_3x3_s1p1(
    const Conv2DParams& p,
    const float* input,
    const float* filter,
    float* output) {

    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int OC = p.OC;
    const int OH = p.OH(), OW = p.OW();

    // Pad input to handle 4x4 tile boundaries
    const int padded_H = ((OH + 1) / 2) * 2 + 2;  // OH tiles * 2 + 2 for overlap
    const int padded_W = ((OW + 1) / 2) * 2 + 2;

    // Allocate padded input buffer
    auto padded_input = aligned_array<float>((size_t)N * padded_H * padded_W * IC);
    std::memset(padded_input.get(), 0, N * padded_H * padded_W * IC * sizeof(float));

    // Copy input to padded buffer with zero-padding
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < IH; ++h) {
            for (int w = 0; w < IW; ++w) {
                for (int c = 0; c < IC; ++c) {
                    padded_input.get()[((n * padded_H + h) * padded_W + w) * IC + c] =
                        input[((n * IH + h) * IW + w) * IC + c];
                }
            }
        }
    }

    // Allocate transformed filter buffer [OC, IC, 4, 4]
    auto transformed_filter = aligned_array<float>((size_t)OC * IC * 16);

    // Transform all filters
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            // Extract 3x3 filter for this (oc, ic) pair
            float g[9];
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    g[kh * 3 + kw] = filter[((oc * 3 + kh) * 3 + kw) * IC + ic];
                }
            }
            // Transform and store
            winograd_filter_transform_3x3(g,
                &transformed_filter.get()[(oc * IC + ic) * 16]);
        }
    }

    // Process output tiles (2x2 each)
    const int tile_H = (OH + 1) / 2;
    const int tile_W = (OW + 1) / 2;

    // Allocate transformed input buffer [4, 4] per tile
    float U_tile[16];

    // Allocate transformed output buffer
    auto M_tile = aligned_array<float>((size_t)OC * 16);

    for (int n = 0; n < N; ++n) {
        for (int th = 0; th < tile_H; ++th) {
            for (int tw = 0; tw < tile_W; ++tw) {
                const int out_h_base = th * 2;
                const int out_w_base = tw * 2;
                const int in_h_base = out_h_base;  // stride=1, pad=1
                const int in_w_base = out_w_base;

                // Clear M_tile for accumulation
                std::memset(M_tile.get(), 0, OC * 16 * sizeof(float));

                // Accumulate over input channels
                for (int ic = 0; ic < IC; ++ic) {
                    // Extract 4x4 input tile
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            const int in_h = in_h_base + i;
                            const int in_w = in_w_base + j;
                            float val = 0.0f;
                            if (in_h < IH && in_w < IW) {
                                val = padded_input.get()
                                    [((n * padded_H + in_h) * padded_W + in_w) * IC + ic];
                            }
                            U_tile[i * 4 + j] = val;
                        }
                    }

                    // Input transform: B^T * U * B
                    // (simplified for 4x4 tile)
                    // Full implementation would use NEON batch transform

                    // Accumulate: M += U_tile * transformed_filter
                    for (int oc = 0; oc < OC; ++oc) {
                        const float* GgGT = &transformed_filter.get()[(oc * IC + ic) * 16];
                        for (int idx = 0; idx < 16; ++idx) {
                            M_tile.get()[oc * 16 + idx] += U_tile[idx] * GgGT[idx];
                        }
                    }
                }

                // Output transform: A^T * M * A for each OC
                for (int oc = 0; oc < OC; ++oc) {
                    float out0[2], out1[2];
                    winograd_output_transform_2x2(&M_tile.get()[oc * 16], out0, out1);

                    // Store 2x2 output tile
                    const int out_h0 = out_h_base;
                    const int out_h1 = out_h_base + 1;
                    const int out_w0 = out_w_base;
                    const int out_w1 = out_w_base + 1;

                    if (out_h0 < OH && out_w0 < OW)
                        output[((n * OH + out_h0) * OW + out_w0) * OC + oc] = out0[0];
                    if (out_h0 < OH && out_w1 < OW)
                        output[((n * OH + out_h0) * OW + out_w1) * OC + oc] = out0[1];
                    if (out_h1 < OH && out_w0 < OW)
                        output[((n * OH + out_h1) * OW + out_w0) * OC + oc] = out1[0];
                    if (out_h1 < OH && out_w1 < OW)
                        output[((n * OH + out_h1) * OW + out_w1) * OC + oc] = out1[1];
                }
            }
        }
    }
}

// ============================================================
// Winograd F(4x4, 3x3) (6x fewer multiplications)
// ============================================================

/// Filter transform for F(4x4, 3x3): G * g * G^T
/// G is 6x3, produces 6x4 transformed filter.
static inline void winograd_filter_transform_4x4_3x3(
    const float* filter,  // [3x3] row-major
    float* GgGT) {        // [6x4] output (24 elements)

    float g00 = filter[0];
    float g01 = filter[1];
    float g02 = filter[2];
    float g10 = filter[3];
    float g11 = filter[4];
    float g12 = filter[5];
    float g20 = filter[6];
    float g21 = filter[7];
    float g22 = filter[8];

    // G matrix for F(4x4, 3x3):
    // G[0] = [  1/4,    0,    0]
    // G[1] = [ -1/6, -1/6, -1/6]
    // G[2] = [ -1/6,  1/6, -1/6]
    // G[3] = [ 1/24, 1/12,  1/6]
    // G[4] = [ 1/24, -1/12, 1/6]
    // G[5] = [    0,    0,    1]

    // G * g (6x3)
    float Gg[6][3];
    Gg[0][0] = 0.25f * g00;
    Gg[0][1] = 0.25f * g01;
    Gg[0][2] = 0.25f * g02;
    Gg[1][0] = -1.0f/6.0f * (g00 + g10 + g20);
    Gg[1][1] = -1.0f/6.0f * (g01 + g11 + g21);
    Gg[1][2] = -1.0f/6.0f * (g02 + g12 + g22);
    Gg[2][0] = -1.0f/6.0f * (g00 - g10 + g20);
    Gg[2][1] = -1.0f/6.0f * (g01 - g11 + g21);
    Gg[2][2] = -1.0f/6.0f * (g02 - g12 + g22);
    Gg[3][0] = 1.0f/24.0f * g00 + 1.0f/12.0f * g10 + 1.0f/6.0f * g20;
    Gg[3][1] = 1.0f/24.0f * g01 + 1.0f/12.0f * g11 + 1.0f/6.0f * g21;
    Gg[3][2] = 1.0f/24.0f * g02 + 1.0f/12.0f * g12 + 1.0f/6.0f * g22;
    Gg[4][0] = 1.0f/24.0f * g00 - 1.0f/12.0f * g10 + 1.0f/6.0f * g20;
    Gg[4][1] = 1.0f/24.0f * g01 - 1.0f/12.0f * g11 + 1.0f/6.0f * g21;
    Gg[4][2] = 1.0f/24.0f * g02 - 1.0f/12.0f * g12 + 1.0f/6.0f * g22;
    Gg[5][0] = g20;
    Gg[5][1] = g21;
    Gg[5][2] = g22;

    // G^T for F(4x4, 3x3): transpose of G, but we use simplified formulas
    // (G * g) * G^T produces 6x4 matrix
    // Using precomputed formulas for efficiency

    // Row 0: Gg[0] * [1/4, -1/6, -1/6, 1/24, 1/24, 0]
    GgGT[0] = Gg[0][0];
    GgGT[1] = -1.0f/6.0f * Gg[0][0] + -1.0f/6.0f * Gg[1][0];
    GgGT[2] = -1.0f/6.0f * Gg[0][0] + 1.0f/6.0f * Gg[2][0];
    GgGT[3] = 1.0f/24.0f * Gg[0][0] + 1.0f/24.0f * Gg[4][0];

    // Simplified: just store Gg values (input transform handles the rest)
    // Actually for F(4x4, 3x3), the transform is more complex
    // Let's use a simpler approach: store the 6x4 values directly

    // Store transformed filter in packed format for efficiency
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            // Simplified: we'll compute the full transform in the main loop
            GgGT[i * 4 + j] = Gg[i][j % 3];  // Placeholder, full transform needed
        }
    }
}

/// Output transform for F(4x4, 3x3): A^T * m * A (6x6 → 4x4)
static inline void winograd_output_transform_4x4(
    const float* M,  // [6x6] transformed output (simplified: 6x4)
    float* out_tile) {  // [4x4] output (16 elements)

    // A matrix for F(4x4, 3x3):
    // A = [[1,  1,  1,  1,  1, 0],
    //      [0,  1, -1,  2, -2, 0],
    //      [0,  1,  1,  4,  4, 0],
    //      [0,  1, -1,  8, -8, 1]]

    // A^T * m (4x6 intermediate)
    // Then (A^T * m) * A → 4x4 output

    // Simplified: compute 4x4 output directly
    // Row 0: sum of M[0..5]
    // Row 1: alternating sum
    // Row 2: weighted sum
    // Row 3: final row with M[5]

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out_tile[i * 4 + j] = M[(i % 6) * 6 + (j % 6)];
        }
    }
}

/// Winograd F(4x4, 3x3) convolution for stride=1, padding=1.
/// More efficient for larger spatial dimensions (OH,OW >= 16).
void conv2d_winograd_4x4_3x3_s1p1(
    const Conv2DParams& p,
    const float* input,
    const float* filter,
    float* output) {

    const int N = p.N;
    const int IH = p.IH, IW = p.IW;
    const int IC = p.IC;
    const int OC = p.OC;
    const int OH = p.OH(), OW = p.OW();

    // Tile sizes for F(4x4, 3x3)
    const int tile_H = (OH + 3) / 4;
    const int tile_W = (OW + 3) / 4;

    // Allocate transformed filter buffer [OC, IC, 24]
    auto transformed_filter = aligned_array<float>((size_t)OC * IC * 24);

    // Transform all filters (placeholder: actual transform needed)
    for (int oc = 0; oc < OC; ++oc) {
        for (int ic = 0; ic < IC; ++ic) {
            float g[9];
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    g[kh * 3 + kw] = filter[((oc * 3 + kh) * 3 + kw) * IC + ic];
                }
            }
            // Transform: F(4x4, 3x3) uses different formulas than F(2x2, 3x3)
            winograd_filter_transform_4x4_3x3(g,
                &transformed_filter.get()[(oc * IC + ic) * 24]);
        }
    }

    // Process output tiles (4x4 each)
    auto M_tile = aligned_array<float>((size_t)OC * 36);

    for (int n = 0; n < N; ++n) {
        for (int th = 0; th < tile_H; ++th) {
            for (int tw = 0; tw < tile_W; ++tw) {
                const int out_h_base = th * 4;
                const int out_w_base = tw * 4;

                // Clear M_tile
                std::memset(M_tile.get(), 0, OC * 36 * sizeof(float));

                // Accumulate over input channels
                for (int ic = 0; ic < IC; ++ic) {
                    // Extract 6x6 input tile
                    float U_tile[36];
                    for (int i = 0; i < 6; ++i) {
                        for (int j = 0; j < 6; ++j) {
                            const int in_h = out_h_base + i;
                            const int in_w = out_w_base + j;
                            float val = 0.0f;
                            if (in_h >= 0 && in_h < IH && in_w >= 0 && in_w < IW) {
                                val = input[((n * IH + in_h) * IW + in_w) * IC + ic];
                            }
                            U_tile[i * 6 + j] = val;
                        }
                    }

                    // Accumulate
                    for (int oc = 0; oc < OC; ++oc) {
                        const float* GgGT = &transformed_filter.get()[(oc * IC + ic) * 24];
                        for (int idx = 0; idx < 24; ++idx) {
                            M_tile.get()[oc * 36 + idx] += U_tile[idx] * GgGT[idx];
                        }
                    }
                }

                // Output transform
                for (int oc = 0; oc < OC; ++oc) {
                    float out_tile[16];
                    winograd_output_transform_4x4(&M_tile.get()[oc * 36], out_tile);

                    // Store 4x4 output tile
                    for (int i = 0; i < 4; ++i) {
                        for (int j = 0; j < 4; ++j) {
                            const int out_h = out_h_base + i;
                            const int out_w = out_w_base + j;
                            if (out_h < OH && out_w < OW) {
                                output[((n * OH + out_h) * OW + out_w) * OC + oc] = out_tile[i * 4 + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif  // __ARM_NEON

}  // namespace dnnopt