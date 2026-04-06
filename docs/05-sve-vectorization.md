# SVE/SVE2 向量化优化设计文档

## 1. SVE 编程模型

### 1.1 向量长度无关 (VLA) 编程

SVE 的核心优势: 同一份代码在不同向量宽度硬件上自动适配。

```asm
// VLA 风格的向量加法
    ptrue p0.s                    // 所有 lane 激活
    whilelt p1.s, x_i, x_n       // p1 = (i < n) ? true : false，逐 lane

.Lloop:
    ld1w {z0.s}, p1/z, [x_a, x_i, lsl #2]   // 谓词加载 A
    ld1w {z1.s}, p1/z, [x_b, x_i, lsl #2]   // 谓词加载 B
    fadd z2.s, z0.s, z1.s                      // 向量加法
    st1w {z2.s}, p1, [x_c, x_i, lsl #2]      // 谓词存储 C

    incw x_i                                   // i += VL/32 (自动适配向量宽度)
    whilelt p1.s, x_i, x_n                    // 更新谓词
    b.first .Lloop                             // 还有活跃 lane 则继续
```

**关键**: 无需硬编码向量宽度，`incw` 自动增加当前硬件的向量长度。

### 1.2 谓词 (Predication) 消除尾循环

传统 NEON 处理 N 不是 4 倍数的情况:
```
main_loop:  处理 N/4 次 (4 元素/次)
tail_loop:  标量处理剩余 N%4 个元素   ← 额外代码，性能损失
```

SVE 谓词方式:
```
loop:  WHILELT 生成谓词，自动处理所有元素，包括尾部
       无需额外的尾循环代码
```

## 2. SVE GEMM 微内核

### 2.1 VLA GEMM 设计

```asm
// SVE FP32 GEMM 微内核
// Mr = 8 (固定), Nr = VL (自适应)
// 在 SVE-256 上 Nr=8, 在 SVE-128 上 Nr=4

    ptrue p0.s                         // 全谓词

    // 初始化累加器
    fmov z8.s, #0
    fmov z9.s, #0
    ... // Mr 个累加器

.Lk_loop:
    // 加载 B 的一行 (Nr 个 FP32, 自适应宽度)
    ld1w {z0.s}, p0/z, [x_b]
    add x_b, x_b, x_ldb

    // 加载 A 的一列 (8 个 FP32)
    ld1rw {z1.s}, p0/z, [x_a]          // 广播 A[0,k]
    fmla z8.s, p0/m, z0.s, z1.s        // C[0,:] += A[0,k] * B[k,:]

    ld1rw {z1.s}, p0/z, [x_a, #4]     // 广播 A[1,k]
    fmla z9.s, p0/m, z0.s, z1.s        // C[1,:] += A[1,k] * B[k,:]

    ... // 重复 8 次 (Mr=8)

    add x_a, x_a, x_lda
    subs x_k, x_k, #1
    b.ne .Lk_loop
```

### 2.2 SVE 256-bit (Neoverse V1) 特化

V1 有 2×256-bit SVE 管线，可同时发射 2 条 SVE 指令:
```
策略: 展开 Mr=2, 每次循环发射 2 条 FMLA 到不同管线
    fmla z8.s, p0/m, z0.s, z1.s    // 管线 0
    fmla z9.s, p0/m, z0.s, z2.s    // 管线 1 (同时执行)
```

### 2.3 SVE2 (Neoverse V2/N2) 特化

V2 有 4×128-bit SVE2 管线:
```
策略: 展开 4 条独立 FMLA，填满 4 个管线
    fmla z8.s,  p0/m, z0.s, z4.s   // 管线 0
    fmla z9.s,  p0/m, z1.s, z4.s   // 管线 1
    fmla z10.s, p0/m, z2.s, z4.s   // 管线 2
    fmla z11.s, p0/m, z3.s, z4.s   // 管线 3
```

## 3. SVE 优化 Elementwise 算子

### 3.1 ReLU

```asm
    fmov z_zero.s, #0
    ptrue p0.s
    whilelt p1.s, x_i, x_n

.Lrelu_loop:
    ld1w {z0.s}, p1/z, [x_in, x_i, lsl #2]
    fmax z0.s, p1/m, z0.s, z_zero.s
    st1w {z0.s}, p1, [x_out, x_i, lsl #2]
    incw x_i
    whilelt p1.s, x_i, x_n
    b.first .Lrelu_loop
```

### 3.2 Softmax (SVE 归约)

```asm
    // Step 1: 求最大值 (tree reduction)
    fmov z_max.s, #-inf
.Lmax_loop:
    ld1w {z0.s}, p1/z, [x_in, x_i, lsl #2]
    fmax z_max.s, p1/m, z_max.s, z0.s
    ...
    fmaxv s_max, p0, z_max.s              // 水平归约求全局 max

    // Step 2: exp(x - max) 和 sum
    fmov z_sum.s, #0
.Lexp_loop:
    ld1w {z0.s}, p1/z, [x_in, x_i, lsl #2]
    fsub z0.s, p1/m, z0.s, z_max.s        // x - max
    // ... 多项式近似 exp() ...
    fadd z_sum.s, p1/m, z_sum.s, z0.s
    st1w {z0.s}, p1, [x_tmp, x_i, lsl #2]
    ...
    faddv s_sum, p0, z_sum.s               // 水平求和

    // Step 3: 除以 sum
    fmov z_inv.s, #1.0
    fdiv z_inv.s, p0/m, z_inv.s, z_sum.s  // 1/sum (广播)
.Ldiv_loop:
    ld1w {z0.s}, p1/z, [x_tmp, x_i, lsl #2]
    fmul z0.s, p1/m, z0.s, z_inv.s
    st1w {z0.s}, p1, [x_out, x_i, lsl #2]
    ...
```

## 4. Gather/Scatter 优化

### 4.1 Embedding Lookup

传统 NEON 需要标量 gather:
```c
for (int i = 0; i < n; i++)
    output[i] = table[indices[i]];  // 逐个标量访问
```

SVE gather:
```asm
    ld1w {z_idx.s}, p0/z, [x_indices]          // 加载索引向量
    ld1w {z_out.s}, p0/z, [x_table, z_idx.s, lsl #2]  // 向量 gather!
    st1w {z_out.s}, p0, [x_output]
```

一条指令完成多个不规则内存访问，对 Embedding / Attention 的间接访问非常有价值。

## 5. 运行时向量宽度检测

```c
#include <sys/auxv.h>

// 检测 SVE 支持与向量宽度
bool has_sve = getauxval(AT_HWCAP) & HWCAP_SVE;
bool has_sve2 = getauxval(AT_HWCAP2) & HWCAP2_SVE2;
bool has_bf16 = getauxval(AT_HWCAP2) & HWCAP2_SVEBF16;
bool has_i8mm = getauxval(AT_HWCAP2) & HWCAP2_SVEI8MM;

// 获取 SVE 向量宽度
uint64_t sve_vl;
asm("rdvl %0, #1" : "=r"(sve_vl));  // 返回字节数
int sve_bits = sve_vl * 8;           // 128, 256, 512, ...

// 根据能力选择内核
if (has_i8mm && dtype == INT8)
    dispatch_smmla_kernel();
else if (has_sve && sve_bits >= 256)
    dispatch_sve256_kernel();
else if (has_sve)
    dispatch_sve128_kernel();
else
    dispatch_neon_kernel();
```
