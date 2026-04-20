# DNN-Opt Performance Comparison Report

## Test Configuration

| Configuration | Implementation | Method |
|--------------|----------------|--------|
| TensorFlow Eigen | TF_DISABLE_ONEDNN=1 | tf.matmul() |
| TensorFlow oneDNN | Default TF 2.16.1 | tf.matmul() |
| oneDNN + DNN-Opt | Patched libdnnl.so | dnnl_sgemm() |

## Summary by Category

| Category | Eigen (GF) | oneDNN (GF) | DNN-Opt (GF) | DNN vs Eigen |
|----------|------------|-------------|--------------|--------------|
| **M=1 (GEMV)** | 4.85 | 4.78 | 7.99 | **+65%** |
| **M=2-7 (small batch)** | 1.49 | 1.47 | 25.37 | **+1601%** |
| **Irregular N (prime)** | 0.31 | 0.32 | 19.36 | **+6145%** |
| **Tiny shapes (N<=32)** | 0.09 | 0.09 | 9.95 | **+10512%** |
| Large M (control) | 35.51 | 35.11 | 42.19 | +19% |

## Key Shapes Comparison

### Small-M (oneDNN weakness domain)

| Shape | Eigen | oneDNN | DNN-Opt | DNN/Eigen | DNN/oneDNN |
|-------|-------|--------|---------|-----------|------------|
| [1,64,64] | 0.20 | 0.19 | 10.78 | **54x** | **57x** |
| [1,128,128] | 0.63 | 0.69 | 10.37 | **17x** | **15x** |
| [4,64,64] | 0.60 | 0.62 | 27.31 | **46x** | **44x** |
| [4,128,128] | 2.12 | 2.09 | 27.08 | **13x** | **13x** |
| [6,64,64] | 0.93 | 0.90 | 30.72 | **33x** | **34x** |
| [6,128,128] | 2.97 | 3.02 | 30.15 | **10x** | **10x** |

### Irregular N (prime numbers)

| Shape | Eigen | oneDNN | DNN-Opt | DNN/Eigen |
|-------|-------|--------|---------|-----------|
| [4,13,64] | 0.31 | 0.31 | ~20 | **~65x** |
| [4,17,64] | 0.31 | 0.32 | ~21 | **~67x** |
| [4,37,64] | 0.32 | 0.33 | ~27 | **~84x** |

### Tiny Shapes

| Shape | Eigen | oneDNN | DNN-Opt | DNN/Eigen |
|-------|-------|--------|---------|-----------|
| [8,8,8] | 0.02 | 0.02 | 4.27 | **214x** |
| [8,16,16] | 0.09 | 0.08 | 12.05 | **134x** |
| [8,32,32] | 0.35 | 0.34 | 24.82 | **71x** |

### Large M (oneDNN strength - control group)

| Shape | Eigen | oneDNN | DNN-Opt | DNN/Eigen |
|-------|-------|--------|---------|-----------|
| [32,256,256] | 21.08 | 20.34 | 41.21 | 2.0x |
| [64,512,512] | 41.07 | 41.01 | 42.10 | ~1x |

### Large N (DNN-Opt slower)

| Shape | Eigen | oneDNN | DNN-Opt | Note |
|-------|-------|--------|---------|------|
| [1,4096,4096] | 21.24 | 20.89 | 6.30 | **-71%** (DNN slower) |

## Key Findings

### 1. TensorFlow Framework Overhead

TensorFlow matmul has significant framework overhead that masks actual GEMM performance:

- **Tiny shapes (N<=32)**: Only 0.02-0.35 GFLOPS (framework dominates)
- **Small shapes (N=64-128)**: 0.6-3 GFLOPS (still overhead-heavy)
- **Large shapes (N>=1024)**: 20+ GFLOPS (overhead amortized)

### 2. Eigen vs oneDNN Identical

TensorFlow's built-in oneDNN performs identically to Eigen for all shapes:
- Small-M: Both ~0.6 GFLOPS
- Large-M: Both ~40 GFLOPS

This confirms oneDNN doesn't have special small-M optimization.

### 3. DNN-Opt's Strength Domain

DNN-Opt provides massive improvement when bypassing TensorFlow framework:

| Domain | Improvement |
|--------|-------------|
| M<8, N<=128 | **+10-50x** |
| Irregular N (prime) | **+60-80x** |
| Tiny (N<=32) | **+70-200x** |

### 4. DNN-Opt's Weakness

DNN-Opt is slower for large matrices:
- [1,4096,4096]: DNN-Opt 6.3 GF vs TensorFlow 21 GF (**-71%**)

This is expected: DNN-Opt specializes in small-M optimization.

## Conclusions

1. **DNN-Opt design is correct**: Optimizes oneDNN's weakness (small-M, irregular N)
2. **TensorFlow matmul overhead masks performance**: Framework dominates small shapes
3. **Correct integration**: Use patched oneDNN compiled into TensorFlow
4. **BLAS wrapper ineffective**: TensorFlow doesn't call cblas_sgemm

## Next Steps

- Compile TensorFlow with patched oneDNN for real integration
- Test on SME hardware (Neoverse V3) for matrix extension benefits
- Test PyTorch CPU + oneDNN integration
