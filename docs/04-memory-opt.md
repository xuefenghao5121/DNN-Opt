# 内存子系统优化设计文档

## 1. 缓存层级与优化策略

### 1.1 典型 ARM 服务器缓存参数

| 核心 | L1D | L1D 延迟 | L2 | L2 延迟 | L3 | 缓存行 |
|---|---|---|---|---|---|---|
| Neoverse N1 | 64KB 4-way | 4 cycle | 1MB 8-way | 11 cycle | 共享 32MB | 64B |
| Neoverse V1 | 64KB 4-way | 4 cycle | 1MB 8-way | 11 cycle | 共享 32MB | 64B |
| Neoverse V2 | 64KB 4-way | 4 cycle | 1MB 8-way | ~11 cycle | 共享 32MB | 64B |

### 1.2 BLIS 分块与缓存映射

```
┌─────────────────────────────────────────────┐
│  L3 / 共享缓存                                │
│  ┌────────────────────┐                      │
│  │ B 面板: Kc × Nc    │  ← 整个 B 面板驻留 L3  │
│  └────────────────────┘                      │
│                                              │
│  L2 / 每核私有                                │
│  ┌──────────┐                                │
│  │ A 块:     │  ← Mc × Kc 的 A 块打包后驻留 L2 │
│  │ Mc × Kc  │                                │
│  └──────────┘                                │
│                                              │
│  L1 / 最快访问                                │
│  ┌────┐ ┌────┐                               │
│  │ A微 │ │ B微│  ← 微内核操作的小块驻留 L1      │
│  │面板│ │面板│  Mr×Kc 和 Kc×Nr                 │
│  └────┘ └────┘                               │
│                                              │
│  寄存器                                       │
│  ┌───────────┐                               │
│  │ C tile:   │  ← Mr × Nr 输出始终在寄存器中    │
│  │ Mr × Nr   │                                │
│  └───────────┘                               │
└─────────────────────────────────────────────┘
```

## 2. 数据打包 (Packing)

### 2.1 A 矩阵打包

将 A 的 Mc×Kc 子块重排为微内核友好格式:

```
原始 A (行主序):         打包后 A:
a00 a01 a02 a03 ...     a00 a10 a20 a30 a40 a50 a60 a70  ← 第一个微面板
a10 a11 a12 a13 ...     a01 a11 a21 a31 a41 a51 a61 a71     (Mr=8 行的第 0 列)
a20 a21 a22 a23 ...     a02 a12 a22 a32 a42 a52 a62 a72     (第 1 列)
...                     ...
```

打包确保微内核的 LD1 是完全连续的内存访问。

### 2.2 打包向量化

```asm
// 使用 LD4 进行 4×4 转置打包
ld1 {v0.4s}, [x_row0], x_stride   // 行 0
ld1 {v1.4s}, [x_row1], x_stride   // 行 1
ld1 {v2.4s}, [x_row2], x_stride   // 行 2
ld1 {v3.4s}, [x_row3], x_stride   // 行 3

// 4x4 转置
trn1 v4.4s, v0.4s, v1.4s
trn2 v5.4s, v0.4s, v1.4s
trn1 v6.4s, v2.4s, v3.4s
trn2 v7.4s, v2.4s, v3.4s

trn1 v0.2d, v4.2d, v6.2d
trn2 v2.2d, v4.2d, v6.2d
trn1 v1.2d, v5.2d, v7.2d
trn2 v3.2d, v5.2d, v7.2d

// 连续写入打包缓冲区
st1 {v0.4s-v3.4s}, [x_pack], #64
```

## 3. 预取策略

### 3.1 PRFM 指令变体

```asm
prfm pldl1keep, [addr]    // 预取到 L1, 临时数据（会重用）
prfm pldl2keep, [addr]    // 预取到 L2, 临时数据
prfm pldl1strm, [addr]    // 预取到 L1, 流式数据（用一次）
prfm pstl1keep, [addr]    // 预取用于写入, L1
```

### 3.2 预取距离调优

```
预取距离 = 内存延迟 / 每次迭代时间

Neoverse N1 L2 延迟 ~11 cycle, 每次 FMLA 迭代 ~1 cycle (流水线):
→ 预取距离 ≈ 11-16 次迭代

Neoverse V1 L2 延迟类似但 FMLA 吞吐更高:
→ 预取距离 ≈ 8-12 次迭代
```

### 3.3 分层预取模式

```asm
// GEMM 外层循环: L2 预取
prfm pldl2keep, [x_b_next_panel]    // 预取下一个 B 面板到 L2

// GEMM 内层循环: L1 预取
.Linner:
    prfm pldl1keep, [x_a, #256]     // 预取 A 的后续数据到 L1
    prfm pldl1keep, [x_b, #384]     // 预取 B 的后续数据到 L1
    // ... FMLA 计算 ...
```

## 4. 内存对齐与大页

### 4.1 缓存行对齐

```c
// 所有打包缓冲区对齐到 64 字节（缓存行大小）
void* pack_buf = aligned_alloc(64, pack_size);

// 微内核输出地址也应对齐
// NHWC 布局下，确保 C 维度 padding 到 16 的倍数（4 个 FP32 = 16B）
```

### 4.2 Huge Pages

```c
// 对大型权重矩阵使用 2MB 大页，降低 TLB miss
#include <sys/mman.h>

void* weight_buf = mmap(NULL, size,
    PROT_READ | PROT_WRITE,
    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
    -1, 0);
```

TLB miss 对大矩阵影响显著:
- 4KB 页: 1GB 数据需要 262144 个 TLB 条目
- 2MB 页: 1GB 数据只需 512 个 TLB 条目

### 4.3 Scratchpad 复用

```c
// oneDNN 的 scratchpad 机制: 预分配工作缓冲区
// 避免每次 primitive 执行时 malloc/free

// 全局 scratchpad 模式 (推荐用于推理):
dnnl_primitive_attr_set_scratchpad_mode(attr, dnnl_scratchpad_mode_user);

// 用户管理一个大的 scratchpad buffer, 所有 primitive 共享
```

## 5. 数据布局优化

### 5.1 NHWC vs blocked layout

```
NHWC:     [..., C]           → 通道连续，通用性好
nChw16c:  [..., C/16, ..., 16] → 每 16 通道一块，匹配 NEON 4*4
nChw32c:  [..., C/32, ..., 32] → 匹配 SVE 256-bit (8*4)
```

选择策略:
- 输入/输出: NHWC（与框架兼容）
- 内部计算: 根据向量宽度选择 blocked layout
- Reorder 算子: 使用 NEON/SVE 向量化转换

### 5.2 权重布局预优化

```
推理时权重不变 → 可离线重排为最优布局

FP32 权重: OIhw → OIhw8i8o (匹配 8x8 微内核 tile)
INT8 权重: OIhw → OIhw4i16o4i (匹配 SDOT 4-元素分组)
BF16 权重: OIhw → OIhw2i8o2i (匹配 BFMMLA 2x4 输入)
```
