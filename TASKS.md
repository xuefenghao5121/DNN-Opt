# Task Breakdown — oneDNN ARM Optimization

## Phase 1: 基础设施搭建

### 1.1 构建环境与基准测试框架
- 搭建 ARM 交叉编译 / 原生编译环境（GCC 12+ / LLVM 16+）
- 编译 oneDNN（开启 `DNNL_AARCH64_USE_ACL=ON`）
- 建立性能基准测试套件（覆盖 Conv2D, MatMul, Pooling, Eltwise 等主要算子）
- 集成 perf / LIKWID / ARM SPE 等性能剖析工具
- 建立回归测试框架，确保优化不破坏正确性

### 1.2 性能 Profiling 与瓶颈分析
- 使用典型推理模型（ResNet-50, BERT-base, GPT-2）进行端到端 Profiling
- 识别热点算子与瓶颈（计算/访存/分支预测）
- 生成 roofline model，量化各算子的计算密度与带宽利用率

---

## Phase 2: GEMM 微内核优化（核心，影响 MatMul / Conv / FC）

### 2.1 NEON FP32 GEMM 微内核
- **目标**: 手写 AArch64 汇编微内核，达到 NEON 理论峰值 90%+
- **方法**:
  - 设计 8x12 或 12x8 输出 tile（充分利用 32 个 V 寄存器）
  - 使用 `FMLA Vd.4S, Vn.4S, Vm.S[lane]` by-element 指令实现外积
  - 软件流水线：将第 N+1 次迭代的 LD1 与第 N 次的 FMLA 交错
  - `PRFM PLDL1KEEP` 预取距离调优（8-12 迭代，匹配目标核心延迟）
  - 循环展开 K 维度 4-8 次，摊薄循环控制开销
- **寄存器分配策略**:
  - v0-v3: A 矩阵输入行
  - v4-v7: B 矩阵输入列
  - v8-v23: 累加器（4x4 = 16 个 FP32 tile）
  - v24-v31: 下一迭代预取缓冲

### 2.2 SVE/SVE2 GEMM 微内核
- **目标**: 编写向量长度无关（VLA）的 SVE GEMM 内核
- **方法**:
  - 使用 `FMLA (SVE)` 替代 NEON FMLA，支持 128-2048bit 自适应
  - 利用谓词寄存器（P0-P15）处理尾部元素，消除标量尾循环
  - `WHILELT` 循环控制，`PTRUE` 全谓词初始化
  - Gather/Scatter load 优化稀疏访问模式
  - 针对 Neoverse V1 (256-bit SVE) 和 V2 (4x128-bit SVE2) 分别调优

### 2.3 BF16 GEMM 微内核
- **目标**: 利用 BFMMLA/BFDOT 指令实现接近 2x FP32 的吞吐
- **方法**:
  - `BFMMLA`: 每条指令完成 2x4 × 4x2 → 2x2 FP32 累加
  - `BFDOT`: BF16 点积，适用于不同 tile 形状
  - `BFCVT/BFCVTN`: FP32 ↔ BF16 转换，最小化精度损失
  - 权重离线打包为 BF16 格式，推理时直接使用
  - 适配 `ONEDNN_DEFAULT_FPMATH_MODE=BF16` 配置路径

### 2.4 INT8 GEMM 微内核
- **目标**: 利用 SDOT/UDOT 和 I8MM 指令实现 ~4x FP32 的吞吐
- **方法**:
  - `SDOT/UDOT Vd.4S, Vn.16B, Vm.16B`: 4 组 int8 点积 → int32 累加
  - `SMMLA/UMMLA/USMMLA` (ARMv8.6+): 2x8 × 8x2 → 2x2 int32，吞吐翻倍
  - 逐通道量化（per-channel quantization）支持，避免精度退化
  - 对称/非对称量化路径均实现优化

### 2.5 BLIS 风格 Cache Blocking
- **目标**: 设计匹配 ARM 缓存层级的分块策略
- **方法**:
  - L1 (32-64KB): 微内核输出 tile + B 面板热数据
  - L2 (256KB-1MB): A 矩阵 Mc×Kc 块打包
  - L3 (共享多 MB): B 矩阵 Kc×Nc 面板
  - 数据打包（packing）: 将 A/B 重排为微内核友好的连续内存布局
  - 根据不同 ARM 核心的缓存大小动态调整 Mc/Nc/Kc

---

## Phase 3: 卷积算子优化

### 3.1 im2col + GEMM 卷积优化
- 优化 im2col 内存布局，减少数据拷贝开销
- NHWC 数据格式优先，通道维连续便于向量化
- im2col 与 GEMM 流水线重叠（边展开边计算）

### 3.2 Winograd 卷积
- **F(2×2, 3×3)**: 乘法量减少 2.25x，数值稳定
- **F(4×4, 3×3)**: 乘法量减少 4x，需注意数值精度
- ARM 特化优化:
  - Winograd 域 GEMM 使用 NEON/SVE 微内核
  - 输出变换与 GEMM 融合（数据在寄存器/缓存中直接变换）
  - 自定义内存布局 `[L][K_blk][T_blk][eta][alpha][theta]` 消除非连续访问

### 3.3 Direct Convolution
- 对 1×1 和深度可分离卷积，直接卷积可能优于 im2col
- 利用 NEON/SVE 在通道维上向量化
- 针对常见 kernel size (1×1, 3×3, 5×5) 编写特化内核

### 3.4 卷积后融合（Post-ops Fusion）
- Conv + BiasAdd + ReLU/GELU 融合为单一内核
- Conv + BatchNorm 折叠
- Conv + Sum（残差连接）融合
- 减少中间 tensor 的内存回写

---

## Phase 4: 内存子系统优化

### 4.1 数据布局优化
- 推广 NHWC 作为默认布局（通道维连续，利于向量化）
- blocked layout（如 nChw16c）适配 SVE 向量宽度
- Reorder 算子的向量化加速

### 4.2 预取策略
- `PRFM PLDL1KEEP/PLDL2KEEP` 软件预取
- 根据目标核心内存延迟调整预取距离
- 区分 temporal (KEEP) 与 streaming (STRM) 访问模式
- 在 GEMM 外层循环插入 L2/L3 级预取

### 4.3 内存池与分配优化
- 对齐分配（64 字节对齐，匹配缓存行）
- Scratchpad 内存复用，减少 malloc/free 开销
- 大页（Huge Pages）支持，降低 TLB miss

---

## Phase 5: 量化推理全链路优化

### 5.1 量化框架支持
- 支持 TensorFlow Lite / ONNX Runtime / PyTorch 量化模型格式
- 实现 per-tensor 和 per-channel 量化
- 支持对称量化（zero_point=0）和非对称量化

### 5.2 量化算子优化
- INT8 Conv2D: im2col + INT8 GEMM (SDOT/SMMLA)
- INT8 MatMul: 直接 INT8 GEMM
- INT8 Pooling: 整数算术池化
- 量化/反量化算子向量化

### 5.3 混合精度推理
- FP32 输入 → BF16 计算 → FP32 输出（透明降精度）
- INT8 计算 → FP32 反量化 → 激活函数 → 重量化
- 动态选择最优精度路径

---

## Phase 6: SME（Scalable Matrix Extension）前瞻优化

### 6.1 SME GEMM 内核
- `FMOPA`: FP32 外积累加到 ZA tile
- `SMOPA/UMOPA`: INT8 外积累加（int8→int32 展宽）
- `BFMOPA`: BF16 外积累加
- Streaming Mode (`SMSTART/SMSTOP`) 管理

### 6.2 SME2 多向量操作
- 2-vector / 4-vector 批量操作，进一步提升吞吐
- 适配未来 Neoverse V3 / Cortex-X5 等核心

---

## Phase 7: 算子级专项优化

### 7.1 Pooling 优化
- Max Pooling: NEON `FMAX` 向量化
- Avg Pooling: NEON `FADD` + 标量除法（或乘以 1/count）
- Global Pooling: 树形归约 (`FADDV` on SVE)

### 7.2 Elementwise / Activation 优化
- ReLU: `FMAX(x, 0)` 单指令实现
- GELU / Sigmoid / Tanh: 多项式近似 + NEON 向量化
- SiLU (x * sigmoid(x)): 融合计算
- Softmax: 安全 max 减法 + 向量化 exp 近似 + 归约

### 7.3 LayerNorm / BatchNorm
- 两遍扫描（mean + variance）→ 单遍 Welford 算法
- NEON/SVE 向量化均值和方差计算
- 与后续线性变换融合

### 7.4 Attention / Transformer 特化
- Multi-Head Attention: Q×K^T 和 Attn×V 的 GEMM 优化
- Flash Attention 风格 tiling: 在线 softmax + 分块 GEMM
- KV-Cache 优化: 内存布局与预取

---

## Phase 8: 多线程与系统级优化

### 8.1 线程并行策略
- GEMM 外层循环 M/N 维度并行（OpenMP）
- 避免 false sharing（填充到缓存行边界）
- 大小核调度感知（big.LITTLE / DynamIQ）
- NUMA-aware 内存分配（多 socket ARM 服务器）

### 8.2 编译器优化
- GCC vs Clang 性能对比与编译选项调优
- `-march=armv8.6-a+sve2+bf16+i8mm` 特性启用
- LTO（Link Time Optimization）启用
- PGO（Profile-Guided Optimization）应用

---

## Priority & Impact Matrix

| 优化项 | 影响范围 | 预期加速比 | 优先级 |
|---|---|---|---|
| GEMM 微内核 (FP32/BF16/INT8) | MatMul, Conv, FC | 2-4x | **P0 - 最高** |
| Cache Blocking | 所有计算密集算子 | 1.5-2x | **P0 - 最高** |
| Winograd 3×3 卷积 | Conv2D 3×3 | 2-3x | **P1 - 高** |
| INT8 量化全链路 | 所有量化算子 | 3-4x vs FP32 | **P1 - 高** |
| 预取优化 | 所有算子 | 1.1-1.3x | **P1 - 高** |
| Post-ops Fusion | Conv + Act | 1.2-1.5x | **P2 - 中** |
| SVE/SVE2 向量化 | 所有算子 | 1.5-2x (V1/V2) | **P2 - 中** |
| SME 支持 | GEMM 类算子 | 2x+ (未来硬件) | **P3 - 前瞻** |
| Attention 特化 | Transformer 模型 | 1.5-2x | **P2 - 中** |
| 多线程优化 | 端到端推理 | 近线性扩展 | **P2 - 中** |
