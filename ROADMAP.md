# DNN-Opt Roadmap

## 设计定位

**核心原则**: DNN-Opt 是 oneDNN supplementary patch，不是 replacement。

- **专注弱点**: 只优化 oneDNN 慢的形状 (M<8, 不规则N, batch-1)
- **不抢风头**: 大矩阵 (M>=64) 直接 fallback 给 oneDNN
- **轻量级**: 保持简单，避免过度优化

## oneDNN 弱点 vs 强项

| 场景 | oneDNN | DNN-Opt 定位 |
|------|--------|--------------|
| M>=64 大矩阵 | 很好 (接近峰值) | **不竞争，fallback** |
| M=8-32 中型 | 较好 | 补充优化 |
| **M<8 小矩阵** | 慢几十倍 | **核心目标** |
| **不规则 N (质数)** | 慢 | **核心目标** |
| **batch-1 推理** | 很慢 | **核心目标** |

## Current Version: v2.0

**Status**: Autotune v2 完成，但范围过大，需要收敛

### 需要收敛的功能

| 功能 | 当前状态 | 收敛方向 |
|------|----------|----------|
| Blocking autotune | 所有形状 | **只针对 M<32** |
| Tile autotune | 所有形状 | **删除** (大矩阵领域) |
| Threshold autotune | 所有形状 | **简化** (只 small_m_bound) |

### 保留的功能

| 功能 | 说明 |
|------|------|
| Small-M kernel selection | 核心价值 (M=1-7) |
| Shape cache for small-M | 避免重复 benchmark |
| Tiny kernel (M=1) | GEMV 优化 |
| SmallMWide (M<8, N>=48) | batch-1 推理 |

## Next: v2.1 - Small-M 深度优化

### 专注 small-M (oneDNN 弱点)

**目标**: M<8 场景极致优化

| 任务 | 优先级 | 说明 |
|------|--------|------|
| Small-M SVE2 kernel | P0 | SVE2 128-bit vector for M=1-7 |
| Small-M BF16 kernel | P1 | BFMMLA for M=1-7 |
| Small-M INT8 kernel | P1 | SMMLA for M=1-7 |
| 不规则 N 处理 | P2 | 质数 N, N%8!=0 优化 |

### 删除/简化

| 功能 | 操作 |
|------|------|
| Blocking autotune (大矩阵) | **删除**，fallback oneDNN |
| Tile autotune | **删除**，over-engineering |
| Threshold autotune | **简化**，只保留 small_m_bound |

## Future: 等待硬件

### SME (Neoverse V3)

- SME 对大矩阵效果好 → **让 oneDNN 处理**
- SME small-M 可能有价值 → 待验证

### FP8 (ARMv9-A)

- FP8 small-M 有价值 → 待硬件

## Version History

### v2.0 (2026-04-20)
- Autotune 三方向完成
- **发现问题**: 范围过大，抢 oneDNN 风头
- **调整方向**: 收敛到 small-M

### v0.9.28 (2026-04-19)
- Runtime autotuning for kernel selection
- Performance: batch-1 +13.1x vs oneDNN

## Completed: v0.9.23 Tasks

### ✅ INT8 SMMLA Native GEMM
- `gemm_int8_int8int8int32()`: 8×8 tile vmmlaq_s32 processing
- Correct B layout: [OC, K] column-major for transpose product
- Conv3D INT8 uses SMMLA native GEMM
- Performance: 74-174 GFLOPS (2-4x vs FP32)

### ⏳ Conv3D Winograd F(2x2, 3x3x3)
- Framework created (`conv_winograd3d.cpp`)
- Input/filter/output transform functions implemented
- Currently disabled pending correctness validation
- Expected: 2.25x fewer spatial MACs

### ✅ Conv3D INT8 Performance Validation
- test_conv3d_correctness: 18/18 passed
- Benchmark: INT8 74-174 GFLOPS on Neoverse N2

## Completed: v0.9.22 Tasks

### ✅ Direct INT8 GEMM
- `gemm_int8_int8int8int32()`: INT8×INT8→INT32 compute
- No dequantization overhead before GEMM
- Conv3D INT8 uses native INT8 GEMM
- TODO: SMMLA optimization for 8×8 tile

### ✅ Conv3D Benchmark
- 5 shapes for C3D/I3D video models
- FP32/BF16/INT8 performance comparison
- BF16: 2x improvement (34-106 GFLOPS)
- INT8: Scalar baseline, needs SMMLA

## Completed: v0.9.21 Tasks

### ✅ Conv3D BF16/INT8
- BFMMLA-based 3D convolution for video
- SMMLA-based 3D convolution with dynamic quantization
- NDHWC layout for C3D/I3D models

### ✅ Depthwise INT8
- SMMLA depthwise separable convolution
- Per-tensor dynamic quantization for MobileNet
- API: `conv2d_depthwise_int8()`

### ✅ Conv3D Tests
- 6 shapes for C3D, temporal, batch configurations
- FP32, BF16, INT8 correctness tests

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

## Integration

| 框架 | 状态 | 说明 |
|------|------|------|
| oneDNN patch | Ready | 作为 oneDNN 弱点补充 |
| TensorFlow | Tested | TF 2.16.1 + oneDNN + dnnopt |
| PyTorch | TODO | 等需求 |
| BLAS drop-in | Working | libdnnopt_blas.so |

## Hardware Support

| CPU | DNN-Opt 重点 |
|-----|--------------|
| Neoverse N2 | Small-M FP32/BF16/INT8 |
| Neoverse V3 | FP8+SME (等硬件) |

---

**核心原则**: 只补 oneDNN 弱点，不抢 oneDNN 风头