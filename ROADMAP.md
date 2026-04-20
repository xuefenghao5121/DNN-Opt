# DNN-Opt Roadmap

## 设计定位

**核心原则**: DNN-Opt 是 oneDNN supplementary patch，不是 replacement。

| oneDNN | DNN-Opt |
|--------|---------|
| 大矩阵 (M>=64) 很好 | 补充优化 (Autotune 可帮助) |
| **M<8 很慢** | **核心目标** |
| **不规则 N 很慢** | **核心目标** |

## Current Version: v2.2

**Status**: Autotune warmup + 持久化 + SME 编译框架完成

### Autotune 功能

| 功能 | 状态 |
|------|------|
| Kernel selection | ✅ |
| Blocking autotune | ✅ |
| Tile autotune | ✅ |
| Threshold autotune | ✅ |
| Warmup | ✅ `warmup_all_autotune()` |
| Persistence | ✅ `load/save_all_autotune_cache()` |

### SME Kernel 支持 ✅

cmake 选项: `-DDNNOPT_ENABLE_SME=ON`

| 文件 | 指令 | Priority |
|------|------|----------|
| gemm_ukernel_fp32_sme.cpp | FMOPA | 300 |
| gemm_ukernel_bf16_sme.cpp | BFMOPA | 300 |
| gemm_ukernel_int8_sme.cpp | SMOPA | 300 |

SME kernel 在 SME-capable 硬件 (V3+) 上自动激活。

### Small-M SVE Kernel ✅

性能 (oneDNN弱点):
| Shape | GFLOPS |
|-------|--------|
| 4x127x1024 | 26.1 (不规则 N) |
| 4x63x1024 | 27.6 |

## Hardware Support

| CPU | SME | DNN-Opt |
|-----|-----|---------|
| Neoverse N2 | ❌ | Small-M SVE + Autotune |
| Neoverse V3 | ✅ | SME FP32/BF16/INT8 |

## Version History

### v2.2 (2026-04-20)
- Autotune warmup + 持久化
- SME 编译框架

### v2.1 (2026-04-20)
- Small-M SVE kernel
- Tile autotune 恢复

### v2.0 (2026-04-20)
- Autotune 三方向 (Blocking/Tile/Threshold)