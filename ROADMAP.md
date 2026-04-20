# DNN-Opt Roadmap

## 设计定位

**核心原则**: DNN-Opt 是 oneDNN supplementary patch，不是 replacement。

| oneDNN | DNN-Opt |
|--------|---------|
| 大矩阵 (M>=64) 很好 | 补充优化 (Autotune 可帮助) |
| **M<8 很慢** | **核心目标** |
| **不规则 N 很慢** | **核心目标** |

## Current Version: v2.2

**Status**: SME 编译 + Autotune 集成完成

### SME Kernel 支持 ✅

cmake 选项: `-DDNNOPT_ENABLE_SME=ON -DCMAKE_C_COMPILER=clang-15`

| 文件 | 指令 | Priority |
|------|------|----------|
| gemm_ukernel_fp32_sme.cpp | FMOPA | 300 |
| gemm_ukernel_bf16_sme.cpp | BFMOPA | 300 |
| gemm_ukernel_int8_sme.cpp | SMOPA | 300 |

**编译 Flags:** `-march=armv9-a+sme+bf16+dotprod+fp16+i8mm`

### SME Autotune 集成 ✅

GemmKernelId 新增 `kSME`:
- SME 硬件上 SME 作为候选 benchmark vs Packed
- Shape-aware selection: M>=8, N>=8 触发 SME 候选

### SVE Kernel 支持 ✅

| 文件 | 状态 |
|------|------|
| gemm_ukernel_fp32_sve.cpp | ✅ switch-case unrolling |
| gemm_ukernel_bf16_sve.cpp | ✅ |
| gemm_ukernel_int8_sve.cpp | ✅ |

### Autotune 功能

| 功能 | 状态 |
|------|------|
| Kernel selection | ✅ (含 SME) |
| Blocking autotune | ✅ |
| Tile autotune | ✅ |
| Threshold autotune | ✅ |
| Warmup | ✅ `warmup_all_autotune()` |
| Persistence | ✅ `load/save_all_autotune_cache()` |

### Small-M SVE Kernel ✅

性能 (oneDNN弱点):
| Shape | GFLOPS |
|-------|--------|
| 4x127x1024 | 26.1 (不规则 N) |
| 4x63x1024 | 27.6 |

## Hardware Support

| CPU | SME | DNN-Opt |
|-----|-----|---------|
| Neoverse N2 | ❌ | Small-M SVE + Autotune (开发环境) |
| Neoverse V3 | ✅ | SME FP32/BF16/INT8 + Autotune |

## Version History

### v2.2 (2026-04-20)
- SME 编译验证完成 (Clang 15)
- SME Autotune 集成 (kSME kernel selection)
- NEON-SVE bridge memory fallback
- SVE kernel switch-case unrolling

### v2.1 (2026-04-20)
- Small-M SVE kernel
- Tile autotune 恢复

### v2.0 (2026-04-20)
- Autotune 三方向 (Blocking/Tile/Threshold)

## 编译验证

```bash
cd build_sme
cmake .. -DDNNOPT_ENABLE_SME=ON -DCMAKE_C_COMPILER=clang-15 -DCMAKE_CXX_COMPILER=clang++-15
cmake --build . -j$(nproc)
ctest  # 100% passed (8/8)
```