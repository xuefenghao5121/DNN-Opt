# 量化推理优化设计文档

## 1. 量化方案

### 1.1 量化公式

```
对称量化:   x_q = round(x_f / scale)
            x_f = x_q * scale

非对称量化: x_q = round(x_f / scale) + zero_point
            x_f = (x_q - zero_point) * scale
```

### 1.2 量化粒度

| 粒度 | 描述 | 精度 | 性能 |
|---|---|---|---|
| Per-tensor | 整个 tensor 一个 scale | 低 | 最快 |
| Per-channel | 每个输出通道一个 scale | 高 | 稍慢（反量化开销） |
| Per-group | 每组元素一个 scale | 最高 | 较慢 |

推荐: 权重 per-channel + 激活 per-tensor（精度/性能平衡最优）

## 2. INT8 GEMM 实现

### 2.1 SDOT 路径

```
// C[i,j] += sum_k(A[i,k] * B[k,j]) 其中 A,B 为 int8, C 为 int32
//
// SDOT Vd.4S, Vn.16B, Vm.16B
// 将 16B 视为 4 组 x 4 个 int8
// 每组: d[i] += n[4i]*m[4i] + n[4i+1]*m[4i+1] + n[4i+2]*m[4i+2] + n[4i+3]*m[4i+3]

// 数据打包: A 按行打包为 int8, 每 4 个 K 元素连续
// B 按列打包为 int8, 每 4 个 K 元素连续

.Lint8_loop:
    ld1 {v0.16b}, [x_a], #16     // A: 4 行 × 4 个 K 元素
    ld1 {v1.16b}, [x_b], #16     // B: 4 列 × 4 个 K 元素

    sdot v8.4s,  v0.16b, v1.4b[0]  // C[0:3, 0] += A[0:3, k:k+3] · B[k:k+3, 0]
    sdot v9.4s,  v0.16b, v1.4b[1]  // C[0:3, 1]
    sdot v10.4s, v0.16b, v1.4b[2]  // C[0:3, 2]
    sdot v11.4s, v0.16b, v1.4b[3]  // C[0:3, 3]

    subs x_k, x_k, #4
    b.ne .Lint8_loop
```

### 2.2 SMMLA 路径 (ARMv8.6+ / I8MM)

```
// SMMLA Vd.4S, Vn.16B, Vm.16B
// Vn: 2×8 int8 矩阵 (2 行, 每行 8 元素)
// Vm: 8×2 int8 矩阵 (8 行, 每行 2 元素, 列主序)
// Vd: 2×2 int32 累加

// 数据打包要求:
// A: [M/2, K/8, 2, 8] → 每 2 行 8 列为一个块
// B: [N/2, K/8, 8, 2] → 每 8 行 2 列为一个块

.Lint8mm_loop:
    ld1 {v0.16b}, [x_a], #16     // A: 2×8
    ld1 {v1.16b}, [x_b], #16     // B: 8×2

    smmla v8.4s, v0.16b, v1.16b  // C[0:1, 0:1] += A[0:1, k:k+7] × B[k:k+7, 0:1]

    subs x_k, x_k, #8
    b.ne .Lint8mm_loop
```

### 2.3 性能对比

| 指令 | 每条指令乘加次数 | 相对 FP32 FMLA |
|---|---|---|
| FP32 FMLA | 4 | 1x |
| SDOT | 16 | 4x |
| SMMLA | 32 | 8x |
| BFMMLA | 16 | 4x (FP32 累加) |

## 3. 量化反量化流��线

### 3.1 推理链路

```
INT8 Input → INT8 GEMM (SDOT/SMMLA) → INT32 Accumulator
    → + INT32 Bias
    → × FP32 Output Scale (per-channel)
    → FP32 Result
    → Activation (ReLU/GELU)
    → Requantize to INT8 (if next layer is INT8)
```

### 3.2 反量化向量化

```asm
    // INT32 累加器在 v8-v11 (4×4 int32)
    // Per-channel scale 在 v0 (4 个 FP32)
    // Bias 在 v1 (4 个 int32)

    // 加 bias
    add v8.4s, v8.4s, v1.4s

    // INT32 → FP32
    scvtf v8.4s, v8.4s

    // 乘 scale
    fmul v8.4s, v8.4s, v0.4s

    // ReLU (如果融合)
    movi v2.4s, #0
    fmax v8.4s, v8.4s, v2.4s

    // 如需重量化到 INT8:
    // FP32 → INT32 (四舍五入)
    fcvtns v8.4s, v8.4s
    // INT32 → INT16
    sqxtn v8.4h, v8.4s
    // INT16 → INT8
    sqxtn v8.8b, v8.8h
```

## 4. 量化感知的 Cache Blocking

INT8 数据量为 FP32 的 1/4，意味着:
- 相同 cache 容量可装入 4x 数据
- 可使用更大的 Mc/Nc 分块，提高计算访存比
- L1 热数据区大幅增加

| 参数 | FP32 | INT8 |
|---|---|---|
| Mc | 128 | 256-512 |
| Nc | 2048 | 4096-8192 |
| Kc | 512 | 1024-2048 |

## 5. 混合精度策略

### 5.1 BF16 透明降精度
```
FP32 输入 → BFCVTN 转 BF16 → BFMMLA 计算 → FP32 累加 → FP32 输出
```
- 通过 `ONEDNN_DEFAULT_FPMATH_MODE=BF16` 启用
- 精度损失通常 < 0.5%，对推理可接受

### 5.2 动态精度选择
```python
if model.is_quantized:
    use INT8 path (SDOT/SMMLA)
elif hardware.has_bf16 and config.allow_bf16:
    use BF16 path (BFMMLA)
else:
    use FP32 path (FMLA)
```
