# BF16 与 SME 优化设计文档

## 1. BF16 推理优化

### 1.1 BF16 数据格式

```
FP32:  [1 sign][8 exponent][23 mantissa]  = 32 bit
BF16:  [1 sign][8 exponent][ 7 mantissa]  = 16 bit

相同的指数范围 → 相同的动态范围 (±3.4×10^38)
精度降低: ~3 位十进制 vs ~7 位
```

对 DNN 推理: 权重和激活的精度需求通常远低于 FP32 提供的精度，BF16 足够。

### 1.2 BFMMLA 微内核

```asm
// BFMMLA: 每条指令完成 2x4 × 4x2 → 2x2 FP32 矩阵乘
// 输入: Vn.8H (2行×4列 BF16), Vm.8H (4行×2列 BF16)
// 输出: Vd.4S (2×2 FP32 累加)

// 8×8 BF16 GEMM tile 需要 4×4=16 条 BFMMLA 指令
// 累加器: 16 个寄存器 (v16-v31)
// A 输入: 4 个寄存器 (v0-v3, 每个含 2×4 BF16)
// B 输入: 4 个寄存器 (v4-v7, 每个含 4×2 BF16)

.Lbf16_loop:
    // 加载 A: 8 行 × 4 列 BF16 (4 个 128-bit 寄存器)
    ld1 {v0.8h - v3.8h}, [x_a], #64

    // 加载 B: 4 行 × 8 列 BF16 (4 个 128-bit 寄存器)
    ld1 {v4.8h - v7.8h}, [x_b], #64

    // 计算 8×8 输出:
    // C[0:1, 0:1] += A[0:1, 0:3] × B[0:3, 0:1]
    bfmmla v16.4s, v0.8h, v4.8h
    // C[0:1, 2:3] += A[0:1, 0:3] × B[0:3, 2:3]
    bfmmla v17.4s, v0.8h, v5.8h
    // C[0:1, 4:5]
    bfmmla v18.4s, v0.8h, v6.8h
    // C[0:1, 6:7]
    bfmmla v19.4s, v0.8h, v7.8h

    // C[2:3, 0:1] += A[2:3, 0:3] × B[0:3, 0:1]
    bfmmla v20.4s, v1.8h, v4.8h
    bfmmla v21.4s, v1.8h, v5.8h
    bfmmla v22.4s, v1.8h, v6.8h
    bfmmla v23.4s, v1.8h, v7.8h

    // C[4:5, :]
    bfmmla v24.4s, v2.8h, v4.8h
    bfmmla v25.4s, v2.8h, v5.8h
    bfmmla v26.4s, v2.8h, v6.8h
    bfmmla v27.4s, v2.8h, v7.8h

    // C[6:7, :]
    bfmmla v28.4s, v3.8h, v4.8h
    bfmmla v29.4s, v3.8h, v5.8h
    bfmmla v30.4s, v3.8h, v6.8h
    bfmmla v31.4s, v3.8h, v7.8h

    subs x_k, x_k, #4
    b.ne .Lbf16_loop
```

### 1.3 FP32 → BF16 转换优化

```asm
// BFCVTN: 将 4 个 FP32 narrowing 转换为 4 个 BF16 (放入低 64-bit)
// BFCVTN2: 将 4 个 FP32 转换为 4 个 BF16 (放入高 64-bit)

    // 8 个 FP32 → 8 个 BF16 (一个完整 128-bit 寄存器)
    bfcvtn  v4.4h, v0.4s     // 低 4 个 BF16
    bfcvtn2 v4.8h, v1.4s     // 高 4 个 BF16

    // 批量权重转换 (离线预处理)
    ld1 {v0.4s - v3.4s}, [x_fp32], #64    // 加载 16 个 FP32
    bfcvtn  v4.4h, v0.4s                   // 转换
    bfcvtn2 v4.8h, v1.4s
    bfcvtn  v5.4h, v2.4s
    bfcvtn2 v5.8h, v3.4s
    st1 {v4.8h - v5.8h}, [x_bf16], #32    // 存储 16 个 BF16 (32 字节)
```

## 2. SME (Scalable Matrix Extension)

### 2.1 SME 编程模型

```
┌──────────────────────────────┐
│         Streaming Mode       │
│  ┌────────────────────────┐  │
│  │     ZA Tile Storage    │  │
│  │  ┌──────────────────┐  │  │
│  │  │  ZA0.S  ZA1.S    │  │  │
│  │  │  ZA2.S  ZA3.S    │  │  │
│  │  │  (SVL×SVL bits)  │  │  │
│  │  └──────────────────┘  │  │
│  └────────────────────────┘  │
│  + Streaming SVE (Z0-Z31)   │
│  + Predicate (P0-P15)       │
└──────────────────────────────┘
```

进入/退出 Streaming Mode:
```asm
    smstart sm     // 进入 streaming SVE mode + 启用 ZA
    // ... SME 计算 ...
    smstop sm      // 退出 streaming mode + 保存 ZA
```

**注意**: 进入/退出 streaming mode 有固定开销 (~100 cycles)，
因此只适合大规模矩阵运算，不适合小规模 elementwise 操作。

### 2.2 SME GEMM: FMOPA 外积累加

```asm
// FMOPA: FP32 外积累加到 ZA tile
// Za.S, Pn/M, Pm/M, Zn.S, Zm.S
// Za[i,j] += (Pn[i] && Pm[j]) ? Zn[i] * Zm[j] : 0

    smstart sm                          // 进入 streaming mode

    zero {za}                           // 清零 ZA tile
    ptrue p0.s                          // 全谓词

.Lsme_k_loop:
    ld1w {z0.s}, p0/z, [x_a]           // 加载 A 的一列 (SVL 个元素)
    ld1w {z1.s}, p0/z, [x_b]           // 加载 B 的一行 (SVL 个元素)

    fmopa za0.s, p0/m, p0/m, z0.s, z1.s  // ZA += A_col × B_row (外积!)
    // 对 SVL=512, 这是 16×16 = 256 次 FMA 一条指令!

    add x_a, x_a, x_lda
    add x_b, x_b, x_ldb
    subs x_k, x_k, #1
    b.ne .Lsme_k_loop

    // 从 ZA 读出结果
    mov w_i, #0
.Lstore_loop:
    mov p1.s, p0.s
    st1w {za0h.s[w_i, #0]}, p1, [x_c]  // 存储 ZA 的第 i 行
    add x_c, x_c, x_ldc
    add w_i, w_i, #1
    cmp w_i, x_svl_words
    b.lt .Lstore_loop

    smstop sm                           // 退出 streaming mode
```

### 2.3 SME INT8 外积: SMOPA

```asm
// SMOPA: 有符号 INT8 外积，展宽到 INT32
// Za.S, Pn/M, Pm/M, Zn.B, Zm.B
// 每对 (i,j): Za[i,j] += sum(Zn.B[i*4+k] * Zm.B[j*4+k]) for k=0..3

    smstart sm
    zero {za}
    ptrue p0.b                          // byte 粒度谓词

.Lsme_int8_loop:
    ld1b {z0.b}, p0/z, [x_a]           // A: SVL 个 int8
    ld1b {z1.b}, p0/z, [x_b]           // B: SVL 个 int8

    smopa za0.s, p0/m, p0/m, z0.b, z1.b  // INT8 外积 → INT32 累加
    // 对 SVL=512: 64×64 int8 外积，含 4-元素点积展宽

    ...
    smstop sm
```

### 2.4 SME2 多向量操作

SME2 引入 multi-vector 指令:
```asm
    // 2-vector outer product: 同时处理两组外积
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z3.s
    // SME2 可将上述两条融合为一条 multi-vector 指令

    // 4-vector dot product
    sdot za.s[w_i, 0:3], {z0.b-z3.b}, {z4.b-z7.b}
    // 一条指令处理 4 组向量的点积
```

## 3. 性能预期

| 扩展 | 硬件 | FP32 GEMM 峰值 | 相对 NEON |
|---|---|---|---|
| NEON 128-bit | Neoverse N1 | 8 FLOP/cycle | 1x |
| SVE 256-bit | Neoverse V1 | 16 FLOP/cycle | 2x |
| SVE2 4×128-bit | Neoverse V2 | 32 FLOP/cycle | 4x |
| SME (SVL=512) | Future | 256 FMA/cycle | ~32x |
| BFMMLA (NEON) | N2/V2 | 16 BF16 FMA/cycle | 2x (vs FP32) |
| SMMLA (NEON) | N2/V2 | 32 INT8 MAC/cycle | 4x (vs FP32) |

## 4. 适配策略

```
运行时检测硬件能力
    │
    ├─ 有 SME → SME GEMM (大矩阵) + SVE (小算子)
    │
    ├─ 有 SVE2 + BF16 + I8MM
    │   ├─ INT8 模型 → SMMLA 路径
    │   ├─ BF16 模式 → BFMMLA 路径
    │   └─ FP32 模式 → SVE2 FMLA 路径
    │
    ├─ 有 SVE (无 SVE2)
    │   ├─ SVE 256-bit → SVE FMLA + BFMMLA (if BF16)
    │   └─ SVE 128-bit → SVE FMLA (类似 NEON 但有谓词优势)
    │
    └─ 仅 NEON
        ├─ 有 DotProd → SDOT INT8 路径
        └─ 基线 → NEON FMLA FP32 路径
```
