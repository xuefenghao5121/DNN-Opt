# 卷积算子优化设计文档

## 1. 卷积实现策略选择

| 方法 | 适用场景 | 优势 | 劣势 |
|---|---|---|---|
| im2col + GEMM | 通用卷积 | 复用高度优化的 GEMM | im2col 内存开销 |
| Winograd | 3×3, stride=1 | 减少乘法量 2.25-4x | 变换开销，数值误差 |
| Direct Conv | 1×1, depthwise | 无额外内存 | 需要特化内核 |

## 2. im2col + GEMM 优化

### 2.1 NHWC 优先布局

```
NCHW: tensor[batch][channel][height][width]    ← 通道不连续
NHWC: tensor[batch][height][width][channel]    ← 通道连续，利于向量化
```

NHWC 下 im2col 的内存访问模式更友好:
- 每个输出像素对应的输入 patch 在通道维上连续
- LD1 可直接加载完整通道向量
- 减少 cache miss

### 2.2 Lazy im2col / Fused im2col

传统方式: 先完整 im2col，再 GEMM → 大量临时内存
优化方式: 按 GEMM 分块的需要，逐块展开 im2col

```
for each Mc block of output:
    im2col for this Mc block only → small buffer in L2
    GEMM micro-kernel on this block
```

**好处**: im2col 缓冲区从 O(C×K×K×H×W) 降至 O(Mc×Kc)

## 3. Winograd 卷积

### 3.1 F(2×2, 3×3) 实现

变换矩阵:
```
BT = [[1, 0, -1, 0],    G = [[1,    0,   0  ],    AT = [[1, 1, 1, 0],
      [0, 1,  1, 0],         [1/2,  1/2, 1/2],          [0, 1,-1, 1]]
      [0,-1,  1, 0],         [1/2, -1/2, 1/2],
      [0, 1,  0,-1]]        [0,    0,   1  ]]
```

计算流程:
1. 输入变换: d = BT × input_tile × B    (4×4 → 4×4)
2. 滤波器变换: g = G × filter × GT       (3×3 → 4×4，离线完成)
3. 逐元素乘法: m = d ⊙ g                 (4×4 element-wise)
4. 输出变换: output = AT × m × A          (4×4 → 2×2)

### 3.2 ARM 特化优化

#### 变换向量化
- 输入/输出变换矩阵的系数为简单整数 (0, ±1, ±1/2)
- 变换可以分解为 FADD/FSUB/FMUL 序列，避免通用矩阵乘
- 使用 NEON 4-lane 并行处理 4 个通道

#### Winograd 域 GEMM
- 对 16 个 Winograd 域点，分别执行 batched GEMM
- 每个 GEMM: (C_out, C_in) × (C_in, num_tiles)
- 使用 Phase 2 中优化的 GEMM 微内核

#### 融合输出变换
```
Winograd GEMM 输出 (in registers)
    → 立即执行 AT × m × A 变换（仍在寄存器中）
    → 直接写入输出 tensor
```
避免中间结果回写内存。

### 3.3 数值稳定性

F(2×2, 3×3) 精度损失 < 0.01%，可安全使用。
F(4×4, 3×3) 精度损失约 0.1-1%，需逐模型验证:
- 对 FP32 推理一般可接受
- 对量化模型不建议使用

## 4. Direct Convolution

### 4.1 1×1 卷积
1×1 Conv 等价于逐像素的矩阵乘:
```
output[h,w,:] = input[h,w,:] × weight[:,:]
```
- NHWC 下直接映射为 GEMM: (H×W, C_in) × (C_in, C_out)
- 无需 im2col，零额外内存

### 4.2 Depthwise Convolution
每个通道独立卷积:
```
for c in range(channels):
    output[:,:,c] = conv2d(input[:,:,c], filter[:,:,c])
```
- 计算量小，受访存瓶颈限制
- NEON 向量化跨通道并行（一次处理 4/8 个通道）
- 关键: 数据预取和 cache 友好的遍历顺序

## 5. Post-ops Fusion

### 5.1 融合模式

```
Conv → BiasAdd → ReLU      → 单一 kernel，省 2 次内存回写
Conv → BN → ReLU           → BN 折叠到 weight/bias，免费
Conv → Sum → ReLU          → 残差加法融合，省 1 次回写
Conv → BiasAdd → GELU      → 需要 GELU 近似的向量化实现
```

### 5.2 实现方式

在 GEMM 微内核 epilogue 中插入后处理:
```asm
    // GEMM 累加完成后，结果在 v8-v31
    // BiasAdd
    ld1 {v0.4s}, [x_bias], #16
    fadd v8.4s, v8.4s, v0.4s

    // ReLU
    movi v1.4s, #0
    fmax v8.4s, v8.4s, v1.4s

    // 写入输出
    st1 {v8.4s}, [x_out], #16
```
