# GEMM 微内核设计文档

## 1. 概述

GEMM（General Matrix Multiply）是深度学习推理中最核心的计算原语。
Conv2D（通过 im2col）、MatMul、InnerProduct（FC 层）最终都归结为 GEMM。
微内核的性能直接决定了整个推理引擎的性能上限。

## 2. ARM NEON FP32 微内核设计

### 2.1 输出 Tile 尺寸选择

AArch64 有 32 个 128-bit NEON 寄存器（v0-v31），每个可存放 4 个 FP32。

**选择 8×12 tile**:
- 累加器: 8×12/4 = 24 个寄存器 (v8-v31)
- A 输入: 8/4 = 2 个寄存器 (v0-v1)
- B 输入: 12/4 = 3 个寄存器 (v2-v4)
- 预取缓冲: 3 个寄存器 (v5-v7)
- 总计: 24 + 2 + 3 + 3 = 32 寄存器，完全利用

**计算访存比**: 8×12 = 96 FMA / (8+12) = 4.8 FMA/加载元素 → 优秀

### 2.2 汇编伪代码

```asm
// 8x12 FP32 GEMM 微内核
// 输入: A (8×K), B (K×12), C (8×12 累加器)
// 寄存器分配:
//   v0-v1:   A 列 (8 个 FP32)
//   v2-v4:   B 行 (12 个 FP32)
//   v5-v7:   下一迭代 B 预取
//   v8-v31:  8×12/4 = 24 个累加器

    // 初始化累加器为 0 或加载已有 C
    movi v8.4s, #0
    ...
    movi v31.4s, #0

.Lloop_k:
    // 预取下一迭代数据
    prfm pldl1keep, [x_a, #256]
    prfm pldl1keep, [x_b, #384]

    // 加载 A 的 8 个元素 (2 个 v 寄存器)
    ld1 {v0.4s}, [x_a], #16
    ld1 {v1.4s}, [x_a], #16

    // 加载 B 的 12 个元素 (3 个 v 寄存器)
    ld1 {v2.4s}, [x_b], #16
    ld1 {v3.4s}, [x_b], #16
    ld1 {v4.4s}, [x_b], #16

    // 外积计算: 8 行 × 12 列
    // A[0:3] × B[0:3]
    fmla v8.4s,  v2.4s, v0.s[0]    // C[0,0:3] += A[0] * B[0:3]
    fmla v9.4s,  v3.4s, v0.s[0]    // C[0,4:7] += A[0] * B[4:7]
    fmla v10.4s, v4.4s, v0.s[0]    // C[0,8:11]+= A[0] * B[8:11]
    fmla v11.4s, v2.4s, v0.s[1]    // C[1,0:3] += A[1] * B[0:3]
    fmla v12.4s, v3.4s, v0.s[1]
    fmla v13.4s, v4.4s, v0.s[1]
    fmla v14.4s, v2.4s, v0.s[2]    // C[2,0:3] += A[2] * B[0:3]
    fmla v15.4s, v3.4s, v0.s[2]
    fmla v16.4s, v4.4s, v0.s[2]
    fmla v17.4s, v2.4s, v0.s[3]    // C[3,0:3] += A[3] * B[0:3]
    fmla v18.4s, v3.4s, v0.s[3]
    fmla v19.4s, v4.4s, v0.s[3]

    // A[4:7] × B[0:11]
    fmla v20.4s, v2.4s, v1.s[0]    // C[4,0:3] += A[4] * B[0:3]
    fmla v21.4s, v3.4s, v1.s[0]
    fmla v22.4s, v4.4s, v1.s[0]
    fmla v23.4s, v2.4s, v1.s[1]
    fmla v24.4s, v3.4s, v1.s[1]
    fmla v25.4s, v4.4s, v1.s[1]
    fmla v26.4s, v2.4s, v1.s[2]
    fmla v27.4s, v3.4s, v1.s[2]
    fmla v28.4s, v4.4s, v1.s[2]
    fmla v29.4s, v2.4s, v1.s[3]
    fmla v30.4s, v3.4s, v1.s[3]
    fmla v31.4s, v4.4s, v1.s[3]

    subs x_k, x_k, #1
    b.ne .Lloop_k

    // 存储结果
    st1 {v8.4s},  [x_c], #16
    ...
```

### 2.3 软件流水线优化

```
迭代 N:                    迭代 N+1:
  FMLA (计算)      ←→      LD1 (加载)
  FMLA (计算)      ←→      PRFM (预取 N+2)
  FMLA (计算)      ←→      LD1 (加载)
  ...
```

关键：将 LD1 和 FMLA 交错排列，利用乱序执行隐藏访存延迟。

## 3. BF16 微内核设计

### 3.1 BFMMLA 指令利用

```
BFMMLA Vd.4S, Vn.8H, Vm.8H
```
- 输入: 两个 128-bit 寄存器，各含 8 个 BF16 元素
- Vn 视为 2×4 矩阵，Vm 视为 4×2 矩阵
- 输出: 2×2 FP32 累加到 Vd

**吞吐**: 相比 FP32 FMLA，每条指令计算 16 次乘加（vs FMLA 的 4 次），理论 4x 操作密度。
考虑到执行延迟，实际吞吐约 **2x FP32**。

### 3.2 数据打包格式

权重预打包为 BF16:
```
原始 FP32 权重 [K, N]
  → BFCVTN 转换为 BF16
  → 重排为 [K/4, N/2, 4, 2] 匹配 BFMMLA 输入格式
```

## 4. INT8 微内核设计

### 4.1 SDOT 路径 (ARMv8.2+)

```
SDOT Vd.4S, Vn.16B, Vm.16B
```
- 4 组并行: 每组 4 个 int8 相乘求和 → 1 个 int32
- 每条指令: 16 次乘法 + 12 次加法

### 4.2 SMMLA 路径 (ARMv8.6+)

```
SMMLA Vd.4S, Vn.16B, Vm.16B
```
- 2×8 × 8×2 → 2×2 int32
- 比 SDOT 吞吐翻倍
- 需要将数据重排为 2×8 / 8×2 块格式

## 5. Cache Blocking 参数

| 目标核心 | L1D | L2 | Mc | Nc | Kc |
|---|---|---|---|---|---|
| Neoverse N1 | 64KB | 1MB | 128 | 2048 | 512 |
| Neoverse V1 | 64KB | 1MB | 128 | 2048 | 384 |
| Neoverse V2 | 64KB | 1MB | 128 | 2048 | 384 |
| Cortex-X2 | 64KB | 512KB | 64 | 1024 | 256 |

参数需根据实际 Benchmark 进一步微调。
