# Changelog

All notable changes to DNN-Opt will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.9.28-dev] - 2026-04-20

### Added
- **Runtime Autotuning for Kernel Selection** (`src/gemm/gemm_autotune.cpp`)
  - `select_gemm_kernel()`: Micro-benchmark comparison for optimal kernel selection
  - `warmup_gemm_autotune()`: Pre-populate cache for common inference shapes
  - `ShapeCache`: LRU cache (256 entries) with file persistence
  - `GemmShapeKey`: 64-bit hash for GEMM/Conv2D shapes
  - Environment variable `DNNOPT_AUTOTUNE=1` enables autotune-guided dispatch

### Kernel Candidates
- `kTiny`: M=1 or N=1 (GEMV/vector operations)
- `kSmallM`: M<8, no packing overhead
- `kSmallMWide`: M<8, N>=48, 48-col macro-tiling
- `kAdaptiveTile`: M=4-32, unpacked + Kc blocking
- `kPacked`: M>=8, packing + threading

### Performance
- batch-1 LLM (1×4096×4096): +8.8% vs heuristic
- batch-4 LLM (4×4096×4096): +8.8% vs heuristic
- batch-8 LLM (8×4096×4096): +8.5% vs heuristic

### Files
- `include/dnnopt/autotune/shape_cache.h`: ShapeKey, KernelSelection, ShapeCache
- `src/autotune/shape_cache.cpp`: LRU cache implementation
- `src/gemm/gemm_autotune.cpp`: Kernel selection + warmup API
- `tests/test_autotune.cpp`: Autotune correctness tests

### Tests
- test_autotune: All passed (shape cache, kernel selection, persistence)
- test_gemm_correctness: 74/74 passed
- test_conv_correctness: 31/31 passed

## [0.9.23-dev] - 2026-04-19

### Added
- **INT8 SMMLA Native GEMM** (`gemm_int8_native.cpp`)
  - 8×8 tile processing using vmmlaq_s32 SMMLA instruction
  - Correct B matrix layout: [OC, K] column-major for transpose product
  - 4x compute density improvement vs FP32

### Fixed
- **INT8 Packed GEMM** (`gemm_driver_int8.cpp`, `gemm_pack_int8.cpp`, `gemm_smallm_int8.cpp`)
  - Global quantization scale instead of per-panel for consistency
  - Fixed B tile layout: SMMLA format [K-group][col-pair]
  - Fixed A row-pair loading: combine two rows' K values correctly
  - Local debug test passes (max_diff=0.008)

### Changed
- **Conv3D INT8 Performance**
  - Small-3x3x3: 74 GFLOPS (3.4x vs FP32)
  - Medium-3x3x3: 148 GFLOPS (4.2x vs FP32)
  - C3D-conv2-mini: 174 GFLOPS (4.2x vs FP32)

### Performance
- INT8 Conv3D achieves 74-174 GFLOPS on Neoverse N2
- 2-4x improvement over FP32 baseline

### Known Issues
- INT8 packed GEMM test program has memory corruption for large shapes
- Needs further debugging

### Tests
- test_conv3d_correctness: 18/18 passed (INT8 SMMLA)

## [0.9.22-dev] - 2026-04-19

### Added
- **Native INT8 GEMM** (`gemm_int8_native.cpp`)
  - `gemm_int8_int8int8int32()`: Direct INT8×INT8→INT32 GEMM
  - No dequantization overhead before compute
  - Used in Conv3D INT8 for better memory efficiency
  - TODO: SMMLA optimization for 8×8 tile processing

- **Conv3D Benchmark** (`bench_conv3d.cpp`)
  - 5 shapes for C3D/I3D video models
  - FP32, BF16, INT8 performance comparison
  - BF16 shows 2x improvement (34-106 GFLOPS)
  - INT8 baseline (scalar), needs SMMLA optimization

### Changed
- **Conv3D INT8**: Uses native INT8 GEMM instead of dequantize fallback
  - Reduces memory overhead (no FP32 intermediate buffers)
  - Correctness verified (18/18 tests passed)

## [0.9.21-dev] - 2026-04-19

### Added
- **Conv3D BF16/INT8** (`conv3d.cpp`)
  - `conv3d_bf16()`: BFMMLA-based 3D convolution for video
  - `conv3d_int8()`: SMMLA-based 3D convolution with dynamic quantization
  - C3D/I3D video models with mixed precision support

- **Depthwise INT8** (`conv_depthwise.cpp`)
  - `conv2d_depthwise_int8()`: SMMLA depthwise separable convolution
  - Per-tensor dynamic quantization for MobileNet-style layers
  - `conv2d_depthwise_bf16()`: API available (uses FP32 fallback due to compiler backend issues)

- **Conv3D Tests** (`test_conv3d_correctness.cpp`)
  - 6 test shapes for C3D, temporal, batch configurations
  - FP32, BF16, INT8 correctness verification

### Tests
- test_conv3d_correctness: 18/18 passed

## [0.9.20-dev] - 2026-04-19

### Added
- **Grouped Convolution** (`conv_grouped.cpp`)
  - Per-group dispatch for ResNeXt/ShuffleNet (groups > 1, groups < IC)
  - Winograd F(2x2)/F(4x4) per-group optimization
  - 1x1 grouped direct GEMM path
  - API: `conv2d_grouped_fp32()`

- **BF16 Conv2D** (`conv_bf16.cpp`)
  - FP32 → BF16 conversion for input/filter
  - BFMMLA-based convolution for 2x compute density
  - Dynamic filter quantization (one-time)
  - API: `conv2d_bf16()`

- **INT8 Conv2D** (`conv_int8.cpp`)
  - Per-tensor dynamic quantization
  - SMMLA-based convolution (4x compute density vs FP32)
  - INT32 accumulation → FP32 dequantization
  - API: `conv2d_int8()`

- **Conv3D** (`conv3d.cpp`, `conv3d.h`)
  - 3D temporal+spatial convolution for video processing
  - im2col3d + GEMM implementation
  - NDHWC layout support
  - C3D/I3D video model support
  - API: `conv3d_fp32()`

### Changed
- **Conv2D Dispatch Complete**
  - Full dispatch chain: Depthwise → Grouped → 1x1 → Winograd → im2col
  - Automatic precision selection based on hardware capabilities

### Tests
- test_conv_correctness: Passed
- test_gemm_correctness: Passed

## [0.9.19-dev] - 2026-04-18

### Added
- **Depthwise Separable Convolution** (`conv_depthwise.cpp`)
  - Dedicated kernel for MobileNet/EfficientNet (groups=IC, OC=IC)
  - Vectorized 4-channel processing with NEON FMLA
  - 3x3 stride=1 pad=1 specialized path
  - No im2col overhead (direct sliding window compute)
  - Fused ReLU/ReLU6 post-ops

- **Winograd F(4x4, 3x3)** (`conv_winograd.cpp`)
  - 6x fewer multiplications (9 → 1.5 per output pixel)
  - Larger tiles: 6x6 input → 4x4 output
  - Better efficiency for VGG/ResNet large spatial dims

### Changed
- **Conv2D Dispatch Strategy**
  - Priority: Depthwise → 1x1 → Winograd F(4x4) → F(2x2) → im2col
  - Winograd F(4x4) for OH,OW >= 16, F(2x2) for >= 8
  - Depthwise kernel auto-selected for groups=IC

- **Convolution Benchmark Suite** (`bench_conv.cpp`)
  - Expanded to 38 shapes
  - Added MobileNetV2, EfficientNet, VGG shapes
  - Added Winograd test shapes (8x8, 16x16, 32x32)
  - Added batch-4 inference shapes

### Performance
- Depthwise 3x3: ~2-3x speedup vs im2col+GEMM for channel-wise compute
- Winograd F(4x4): ~3x speedup vs F(2x2) for large spatial dims

### Tests
- test_conv_correctness: Passed

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