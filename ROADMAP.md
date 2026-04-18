# DNN-Opt Roadmap

## Current Version: v0.9.20-dev

**Status**: Complete Conv2D dispatch + BF16/INT8 Conv + Conv3D

## Completed: v0.9.20 Tasks

### ✅ Grouped Convolution
- ResNeXt/ShuffleNet support (groups > 1)
- Per-group Winograd optimization
- 1x1 grouped GEMM path

### ✅ BF16/INT8 Conv2D
- BFMMLA convolution (2x compute density)
- SMMLA convolution (4x compute density)
- Dynamic quantization support

### ✅ Conv3D (Video)
- im2col3d + GEMM implementation
- NDHWC layout for video models

## Completed: v0.9.19 Tasks

### ✅ Depthwise Convolution
- Dedicated kernel for groups=IC (MobileNet-style)
- NEON vectorized 4-channel processing
- 3x3 stride=1 pad=1 specialized path

### ✅ Winograd F(4x4, 3x3)
- 6x fewer multiplications
- Larger tiles for better amortization
- Dispatch for OH,OW >= 16

### ✅ Convolution Benchmark
- 38 shapes (ResNet, MobileNet, EfficientNet, VGG)
- Winograd test cases

## Completed: v0.9.18 Tasks

### ✅ Autotuning Improvements
- Expanded search grid (5 candidates)
- Multi-shape testing (5 shapes)
- Shape-specific weighted scoring

### ✅ Winograd 3x3 Convolution
- F(2x2, 3x3) algorithm implemented
- 2.25x fewer multiplications
- Dispatch integrated in conv2d_fp32()

### ✅ Benchmark Suite
- BF16/INT8 benchmarks verified in bench_gemm.cpp

## Phase 1: FP8 Hardware Support (Pending ARMv9-A)

**Target Hardware**: Neoverse V3, Cortex-X4, Cortex-A725

### Prerequisites
- ARMv9-A CPU with FP8 extension
- GCC 13+ or Clang 16+ compiler with `__ARM_FEATURE_FP8` support

### Tasks
1. **FP8 NEON Kernel** (`gemm_ukernel_fp8_neon.cpp`)
   - FDOT/FMLA for E4M3 × E5M2 compute
   - 8x8 or 16x8 tile dimensions
   - Inline FP8→FP32 conversion for accumulation

2. **FP8 SVE Kernel** (`gemm_ukernel_fp8_sve.cpp`)
   - VLA kernel for SVE-256+
   - Predicate-based edge handling
   - SVE FP8 FMLA instructions

3. **FP8 Small-M Kernel**
   - M=1-8 FP8 kernels without packing
   - Dynamic FP32→FP8 quantization

4. **Registry Integration**
   - Register FP8 kernels in `GemmUkernelRegistry`
   - Hardware capability detection for FP8
   - Dispatch integration in `gemm_fp8()`

### Expected Performance
- FP8 E4M3: 2x theoretical throughput vs FP32 (same compute units)
- FP8 E5M2: Similar to E4M3 with extended range

### Timeline
- Waiting for ARMv9-A hardware availability
- Implementation ready when compiler support arrives

## Phase 2: SME (Scalable Matrix Extension)

**Target Hardware**: Neoverse V3, Cortex-X4

### Prerequisites
- ARMv9-A CPU with SME extension
- SME-capable assembler (GCC 12+ / Clang 15+)

### Current Status
- SME framework exists (`gemm_sme.cpp`, `gemm_ukernel_fp32_sme.cpp`)
- Compile-only, requires SME hardware to run

### Tasks
1. **SME FP32 Kernel**
   - 2D streaming mode for matrix multiply
   - Tile storage in ZA0-ZA15 registers
   - Zero software pipelining overhead

2. **SME BF16 Kernel**
   - BFMMLA in streaming mode
   - Tile-based compute with zero packing

3. **SME INT8 Kernel**
   - SMMLA in streaming mode
   - Maximum compute density

### Expected Performance
- SME eliminates microkernel packing overhead
- 1.5-2x speedup for large regular matrices
- Zero L2 cache pressure from packed buffers

### Timeline
- SME kernels require SME hardware (Neoverse V3)
- Framework ready, implementation pending hardware

## Phase 3: Convolution Optimization

### Current Status
- Conv2D via im2col + GEMM
- Basic implementation, no specialization

### Tasks
1. **Direct Convolution**
   - Winograd algorithm for 3x3 kernels
   - Avoid im2col memory overhead

2. **Depthwise Convolution**
   - Specialized kernels for depthwise separable
   - Mobile inference optimization

3. **Grouped Convolution**
   - Batch processing for grouped conv
   - Efficient for ResNeXt, ShuffleNet

4. **3D Convolution**
   - Video processing kernels
   - Temporal dimension optimization

### Timeline
- Post-GEMM optimization phase
- Depends on oneDNN integration feedback

## Phase 4: Autotuning & Benchmarking

### Tasks
1. **Runtime Autotuning**
   - Online shape profiling
   - Adaptive kernel selection
   - Cache for repeated shapes

2. **Benchmark Suite Expansion**
   - Add BF16/INT8/FP8 benchmarks
   - Model-level benchmarks (BERT, LLM, CVR)
   - Roofline analysis integration

3. **Performance Database**
   - Shape × Kernel performance matrix
   - Automatic best-kernel selection
   - Version-controlled performance data

### Timeline
- Continuous improvement
- Per-release benchmark updates

## Phase 5: Integration & Deployment

### oneDNN Integration
- **Status**: Patch ready (`integration/onednn/0001-dnnopt-integration.patch`)
- **Next**: Submit as oneDNN contribution PR

### TensorFlow Integration
- **Status**: Build tested (TF 2.16.1 + oneDNN)
- **Next**: dnnopt patch integration in TF oneDNN fork

### PyTorch Integration
- **Status**: Not started
- **Plan**: Similar approach as TensorFlow
- **Dependency**: PyTorch oneDNN integration path

### BLAS Drop-in
- **Status**: Working (`libdnnopt_blas.so`)
- **Next**: NumPy/SciPy compatibility testing

## Hardware Roadmap

| CPU | Release | BF16 | INT8 | FP8 | SME | DNN-Opt Support |
|-----|---------|------|------|-----|-----|-----------------|
| Neoverse N1 | 2019 | ❌ | ❌ | ❌ | ❌ | FP32 only |
| Neoverse N2 | 2020 | ✅ | ✅ | ❌ | ❌ | FP32 + BF16 + INT8 |
| Neoverse V1 | 2021 | ✅ | ✅ | ❌ | ✅ | FP32 + BF16 + SME |
| Neoverse V2 | 2023 | ✅ | ✅ | ❌ | ✅ | FP32 + BF16 + SME |
| Neoverse V3 | 2024 | ✅ | ✅ | ✅ | ✅ | Full (FP8+SME) |
| Cortex-X4 | 2024 | ✅ | ✅ | ✅ | ✅ | Full (FP8+SME) |
| Cortex-A725 | 2024 | ✅ | ✅ | ✅ | ✅ | Full (FP8+SME) |

## Version Milestones

### v0.9.21 (Next)
- Conv3D BF16/INT8 kernels
- Depthwise BF16/INT8 kernels
- Conv benchmark with BF16/INT8 shapes

### v0.10.0 (Major)
- FP8 kernels when ARMv9-A hardware available
- SME kernels fully implemented
- PyTorch integration

### v1.0.0 (Stable)
- Full ARMv9-A support
- oneDNN upstream integration
- Production-ready API freeze