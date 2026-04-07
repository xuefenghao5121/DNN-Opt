/// @file cblas_saxpy.cpp
/// CBLAS saxpy: Y = alpha * X + Y
/// NEON-accelerated for contiguous case (incX == incY == 1).

#include "dnnopt/blas/cblas.h"

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

extern "C" {

void cblas_saxpy(int N, float alpha, const float *X, int incX,
                 float *Y, int incY) {
    if (N <= 0 || alpha == 0.0f) return;

    if (incX == 1 && incY == 1) {
        // Contiguous case — NEON accelerated
        int i = 0;
#ifdef __ARM_NEON
        float32x4_t va = vdupq_n_f32(alpha);
        for (; i + 15 < N; i += 16) {
            float32x4_t x0 = vld1q_f32(X + i);
            float32x4_t x1 = vld1q_f32(X + i + 4);
            float32x4_t x2 = vld1q_f32(X + i + 8);
            float32x4_t x3 = vld1q_f32(X + i + 12);
            float32x4_t y0 = vld1q_f32(Y + i);
            float32x4_t y1 = vld1q_f32(Y + i + 4);
            float32x4_t y2 = vld1q_f32(Y + i + 8);
            float32x4_t y3 = vld1q_f32(Y + i + 12);
            vst1q_f32(Y + i,      vfmaq_f32(y0, va, x0));
            vst1q_f32(Y + i + 4,  vfmaq_f32(y1, va, x1));
            vst1q_f32(Y + i + 8,  vfmaq_f32(y2, va, x2));
            vst1q_f32(Y + i + 12, vfmaq_f32(y3, va, x3));
        }
        for (; i + 3 < N; i += 4) {
            float32x4_t x0 = vld1q_f32(X + i);
            float32x4_t y0 = vld1q_f32(Y + i);
            vst1q_f32(Y + i, vfmaq_f32(y0, va, x0));
        }
#endif
        for (; i < N; ++i) {
            Y[i] += alpha * X[i];
        }
    } else {
        // Strided case — scalar
        for (int i = 0; i < N; ++i) {
            Y[i * incY] += alpha * X[i * incX];
        }
    }
}

}  // extern "C"
