# DNN-Opt Roadmap

## 设计定位

**核心原则**: DNN-Opt 是 oneDNN supplementary patch，不是 replacement。

| oneDNN | DNN-Opt |
|--------|---------|
| 大矩阵 (M>=64) 很好 | 补充优化 (Autotune 可帮助) |
| **M<8 很慢** | **核心目标** |
| **不规则 N 很慢** | **核心目标** |

## Current Version: v2.2

**Status**: SME + Autotune 完全集成

### SME Kernel 支持 ✅

cmake 选项: `-DDNNOPT_ENABLE_SME=ON -DCMAKE_C_COMPILER=clang-15`

| 文件 | 指令 | Priority |
|------|------|----------|
| gemm_ukernel_fp32_sme.cpp | FMOPA | 300 |
| gemm_ukernel_bf16_sme.cpp | BFMOPA | 300 |
| gemm_ukernel_int8_sme.cpp | SMOPA | 300 |

**编译 Flags:** `-march=armv9-a+sme+bf16+dotprod+fp16+i8mm`

### SME Autotune 完全集成 ✅

| 层级 | 状态 | 集成点 |
|------|------|--------|
| Kernel selection | ✅ | select_gemm_kernel() → kSME |
| SME dispatch | ✅ | gemm.cpp switch case kSME |
| Tile autotune | ✅ | dispatch_via_registry() |
| Threshold autotune | ✅ | gemm.cpp dispatch |

**Autotune 流程 (DNNOPT_AUTOTUNE=1):**

```
Shape M×N×K:
  ├─ select_gemm_kernel() → kernel ID
  ├─ select_tile_params() → Mr, Nr
  ├─ get_autotuned_blocking_params() → Mc, Nc, Kc
  └─ get_current_thresholds() → dispatch bounds
```

### SVE Kernel 支持 ✅

| 文件 | 状态 |
|------|------|
| gemm_ukernel_fp32_sve.cpp | ✅ switch-case unrolling |
| gemm_ukernel_bf16_sve.cpp | ✅ |
| gemm_ukernel_int8_sve.cpp | ✅ |

### Autotune 功能 (完整)

| 功能 | 状态 | API |
|------|------|-----|
| Kernel selection | ✅ | select_gemm_kernel() |
| Blocking autotune | ✅ | get_autotuned_blocking_params() |
| Tile autotune | ✅ | select_tile_params() |
| Threshold autotune | ✅ | get_current_thresholds() |
| Warmup | ✅ | warmup_all_autotune() |
| Persistence | ✅ | load/save_all_autotune_cache() |

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
- SME Autotune 完全集成 (kSME + dispatch + tile + threshold)
- Tile autotune 集成到 dispatch_via_registry
- Threshold autotune 集成到 gemm.cpp dispatch
- gemm_sme.cpp skeleton 清理

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

## 使用 Autotune

```bash
# 启用 autotune
export DNNOPT_AUTOTUNE=1

# Warmup (预填充缓存)
./bench_gemm  # 自动 warmup 常用 shapes

# 持久化缓存
# load_all_autotune_cache("/path/to/cache.bin")
# save_all_autotune_cache("/path/to/cache.bin")
```