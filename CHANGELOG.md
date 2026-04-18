# Changelog

All notable changes to DNN-Opt will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.9.18-dev] - 2026-04-18

### Added
- **Enhanced Autotuning** (`gemm_autotune.cpp`)
  - Expanded search grid: 5 candidates (Conservative → Maximum)
  - Multi-shape testing: Large, Small, Tall-skinny, Square, Batch1
  - Shape-specific tuning with weighted scoring
  - Total autotune cost: ~10-15ms (5 shapes × 5 candidates)

- **Winograd F(2x2, 3x3) Convolution** (`conv_winograd.cpp`)
  - Reduces multiplications by 2.25x (9 → 4 per output pixel)
  - Input/Filter/Output transforms with optimized formulas
  - Dispatch: 3x3 stride=1 padding=1 with OH,OW >= 8
  - Efficient for ResNet/MobileNet 3x3 convolutions

### Changed
- **Conv2D Dispatch**
  - Added Winograd path for 3x3 stride=1 padding=1 convolutions
  - Dispatch order: 1x1 direct → Winograd 3x3 → im2col+GEMM

### Performance
- Winograd 3x3: ~2x speedup vs im2col+GEMM for medium-large spatial dims
- Autotuning: Better cache blocking for unknown ARM CPUs

### Tests
- test_conv_correctness: Passed with Winograd dispatch
- test_gemm_correctness: Passed

## [0.9.17-dev] - 2026-04-18

### Added
- **NEON-SVE Bridge** (`arm_neon_sve_bridge.h`)
  - `neon_to_sve_f32()` / `sve_to_neon_f32()` for vector conversion
  - `prefer_neon_over_sve(M,N,K)` dispatch helper for SVE-128 optimization
  - `sve_fmla_neon_prefix()` for hybrid NEON-in-SVE compute

- **BF16 Small-M Kernels** (`gemm_smallm_bf16.cpp`)
  - M=1: `gemm_mx1_bf16` GEMV with BFMMLA 2-col pairing
  - M=2-8: `gemm_smallm_bf16_mops` row-pair compute, inline FP32→BF16 conversion
  - No packing overhead for small matrices

- **INT8 Small-M Kernels** (`gemm_smallm_int8.cpp`)
  - M=1: `gemm_mx1_int8` GEMV with dynamic quantization
  - M=2-8: `gemm_smallm_int8_smmla` SMMLA with per-tensor scale
  - Dynamic quantization avoids pre-packing complexity

- **FP8 Framework** (`gemm_fp8.cpp`, `gemm_types.h`)
  - `fp8_e4m3_t`: E4M3FN format, range [-448, 448], precision-focused
  - `fp8_e5m2_t`: E5M2 format, range [-57K, 57K], supports inf/nan
  - API: `gemm_fp8_e4m3()`, `gemm_fp8_e5m2()`, `gemm_fp8()`, `has_fp8_support()`
  - Software fallback (converts FP8→FP32, uses FP32 GEMM)
  - Hardware detection: `kFP8`, `kFP8FMA` hwcaps flags

- **Hardware Capabilities**
  - `kFP8` flag for FP8 arithmetic (ARMv9-A)
  - `kFP8FMA` flag for FP8 FMA instructions
  - FP8 algorithm enums: `kFp8E4m3Fdot`, `kSveFp8E4m3`, etc.

### Changed
- **GEMM Dispatch Integration**
  - `gemm_bf16()` calls `gemm_smallm_bf16()` when M≤8 and BF16 hardware
  - `gemm_int8()` calls `gemm_smallm_int8()` when M≤8 and I8MM hardware
  - Falls back to FP32 small-M driver when hardware unavailable

### Documentation
- Updated README.md with multi-precision section and FP8 specifications
- Added supported data types table
- Added v0.9.17-dev development log entry

### Tests
- BF16/INT8 correctness tests pass (existing test coverage)
- ctest: 100% tests passed

## [0.9.16-dev] - 2026-04-18

### Added
- Shape-based dispatch strategy for oneDNN integration
- OpenBLAS fallback for large regular matrices (M≥32)
- Small-M (M<32) and irregular shapes → dnnopt optimized kernels

### Changed
- oneDNN integration finalized with shape-based dispatch
- ACL integration abandoned (API mismatch with bazel cache)

### Performance
- CVR embedding b1: 11.91 GF (2.7x vs upstream)
- CVR embedding b4: 28.63 GF (6.1x vs upstream)
- LLM qkv b4: 38.70 GF (1.31x vs upstream)
- Average: 31.18 GFLOPS (24% overall improvement)

## [0.9.15-dev] - 2026-04-17

### Added
- TensorFlow 2.16.1 ARM build with oneDNN
- XLA MatMul vs oneDNN threadpool interface mismatch fix
- ARM64 TLSLE linker error solution (LLD linker)

### Fixed
- Use `mkl_aarch64_threadpool` instead of `mkl_aarch64` for XLA compatibility
- Install LLD and use `--linkopt=-fuse-ld=lld` for ARM64 TLSLE

### Performance
- Pre-built tensorflow-aarch64: 13.24 GFLOPS
- Compiled TF + oneDNN: 20.29 GFLOPS (1.53x speedup)

## [0.9.14-dev] - 2026-04-14

### Fixed
- OpenMP conditional `#include <omp.h>` in `gemm_smallm_fp32.cpp`
- OpenMP for loop condition rewrite

### Added
- TensorFlow 2.16.1 ARM build exploration
- dnnopt+oneDNN integration patch preparation

## [0.9.13] - 2026-04-13

### Added
- Packed 6x16 kernel registered in kernel registry
- M=6 large-shape routing: N*K > 4M uses 8x16 M-padding packed path
- N-tail vectorization via `load_b_narrow_tail<N>()` templates
- edge_buf optimization: removed unnecessary memset

### Performance
- 54/1 wins vs oneDNN
- M16_N23: 1.28→1.68x
- M32_N47: 1.63→1.95x

## [0.9.12] - 2026-04-13

### Added
- Clang-15 compiler migration (enables `.s[N]` fused FMLA)
- npo2 kernels: M=3,5,7 dedicated `.s[N]` kernels
- Tall-skinny kernels: N=2-7 template-specialized
- OpenMP N-parallelism for adaptive tile
- oneDNN patch integration via `dnnl_sgemm` injection

### Performance
- 54/1 wins vs oneDNN-native
- CVR 13-22x, BERT 7-12x, LLM 4-6x speedup

## [0.9.0] - 2026-04-12

### Added
- autoGEMM integration
- 6x16 2x K-unrolling, prefetch optimization
- 8x16 packed kernel
- Batch GEMM dispatch: M=4-7 large N*K uses packed+threaded path

## [0.8.0] - 2026-04-11

### Added
- Inline assembly 4x16/6x16 kernels
- autoGEMM dynamic tile selection
- Kc blocking for small shapes

### Performance
- vs oneDNN: 35/17 wins

## [0.5.0] - 2026-04-07

### Added
- CBLAS/BLAS interface
- LD_PRELOAD support
- Drop-in BLAS replacement

### Performance
- vs OpenBLAS: 1.5-1.6x on large matrices

## [0.4.0] - 2026-04-07

### Added
- Conv2D: im2col + GEMM

### Performance
- Up to 17.7x speedup over naive implementation

## [0.3.0] - 2026-04-07

### Added
- Per-CPU tuning profiles (11 ARM families)
- SVE/SME support
- 2D OpenMP threading
- Huge pages

## [0.2.0] - 2026-04-06

### Added
- FP32 8x12 FMLA kernel
- BF16 8x8 BFMMLA kernel
- INT8 8x8 SMMLA kernel

### Performance
- FP32: 93% peak
- BF16: 86.5% peak
- INT8: 70.8% peak

## [0.1.0] - 2026-04-06

### Added
- Build system (CMake)
- Hardware capability detection (HWCAP/CPUID)
- Test framework
- Benchmark framework