# DNN-Opt TODO 列表

**日期**: 2026-04-25
**版本**: v2.6

---

## P0 (紧急 - 必须修复)

### 1. 修复 brgemm stride bug ⚠️

**文件**: `oneDNN v3.7/src/cpu/aarch64/matmul/brgemm_matmul.cpp`

**问题**: `execute_dnnopt()` stride 计算错误导致 tall-skinny shapes 崩溃

**修复方案**:
```cpp
// 行 1470-1472: 修改 stride 计算
// 当前代码 (错误):
const dim_t lda = bgmmc.A_strides[0] / bgmmc.a_dt_sz;
const dim_t ldb = bgmmc.B_strides[0] / bgmmc.b_dt_sz;
const dim_t ldc = bgmmc.C_strides[0] / bgmmc.c_dt_sz;

// 修复后 (正确):
const dim_t lda = bgmmc.A_strides[1] / bgmmc.a_dt_sz;  // row stride
const dim_t ldb = bgmmc.B_strides[1] / bgmmc.b_dt_sz;
const dim_t ldc = bgmmc.C_strides[1] / bgmmc.c_dt_sz;
```

**验证步骤**:
1. 修改代码后重新编译 oneDNN v3.7
2. 运行 benchdnn 测试所有 shapes
3. 确认 M=2,4,8,16,32,64,128,256,512 不再崩溃

**预期结果**: 所有 19 shapes 成功，无崩溃

---

## P1 (重要 - 性能优化)

### 2. GEMV K-blocking 优化

**目标**: 针对大 K GEMV shapes (N=1, K>=256) 提升性能

**当前问题**: GEMV shapes 使用 Standard blocking，但 K 很大时可以优化

**实现方案**:
- 在 `gemm_tiny_dispatch.cpp` 添加 K-blocking
- 为 N=1, K>=256 使用 `Maximum` blocking preset
- Block K 到 cache-friendly chunks (Kc = 256 或 512)

**预期收益**: +20-25% (M=1, K=800)

**文件**: `src/gemm/gemm_tiny_dispatch.cpp`

---

### 3. 极小 shape inline path

**目标**: 极小 shapes (M*N*K < 1000) 避免 kernel dispatch overhead

**当前问题**: M=39,N=1,K=5 只有 5.46 GFLOPS，dispatch overhead 占主导

**实现方案**:
- 在 `gemm_fp32()` 入口添加 threshold check
- 极小 shape 直接 inline scalar loop
- 避免 autotune cache lookup overhead

**预期收益**: +200%

**文件**: `src/gemm/gemm.cpp`

---

## P2 (可选 - 体验优化)

### 4. Shape cache 预加载

**目标**: 模型加载时预加载已知 shapes

**实现方案**:
- 添加 `warmup_gemm_shapes()` API
- 在 inference engine 初始化时调用
- 预加载 embedding layer shapes (35,39,46)

**预期收益**: 减少首次推理延迟

**文件**: `src/autotune/shape_cache.cpp`

---

### 5. Autotune 结果持久化

**目标**: Autotune 结果保存到文件，下次加载直接使用

**实现方案**:
- 扩展 `save_all_autotune_cache()` API
- 支持 blocking/tile/threshold cache 持久化
- 添加 `load_all_autotune_cache()` API

**预期收益**: 减少冷启动 autotune overhead

---

## 已完成项目

| 项目 | 完成日期 | 状态 |
|------|----------|------|
| Blocking autotune | 2026-04-23 | ✅ |
| Tile size autotune | 2026-04-23 | ✅ |
| Threshold autotune | 2026-04-23 | ✅ |
| M=35,39,46 tile heuristic | 2026-04-25 | ✅ |
| benchdnn comparison script | 2026-04-25 | ✅ |
| Performance analysis docs | 2026-04-25 | ✅ |

---

## 优先级排序

1. **P0**: brgemm stride bug → 紧急，阻塞后续测试
2. **P1**: GEMV K-blocking → 高收益，中等难度
3. **P1**: 极小 shape inline → 高收益，低难度
4. **P2**: Shape cache 预加载 → 体验优化
5. **P2**: Autotune 持久化 → 体验优化