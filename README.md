# oneDNN ARM Platform Optimization Project

## Project Goal

基于 ARM 平台特性，针对 oneDNN 库进行深度优化，充分利用 ARM 指令集（NEON/SVE/SVE2/SME）和微架构特征，
在 ARM CPU 环境下实现极致的深度学习推理性能。

## Target Platforms

| CPU Core | 架构 | 向量扩展 | 关键 ML 特性 |
|---|---|---|---|
| Neoverse N1 | ARMv8.2 | 128-bit NEON | DotProd (SDOT/UDOT) |
| Neoverse V1 | ARMv8.4+ | 2x256-bit SVE | BF16, SVE 256-bit |
| Neoverse N2 | ARMv9.0 | 128-bit SVE2 | BF16, I8MM, SVE2 |
| Neoverse V2 | ARMv9.0 | 4x128-bit SVE2 | BF16, I8MM, SVE2 |
| Cortex-X2/X3 | ARMv9.0 | 128-bit SVE2 | BF16, I8MM, SVE2 |

## Project Structure

```
onednn-arm-opt/
├── README.md                  # 项目说明
├── docs/                      # 详细设计文档
│   ├── 01-gemm-microkernel.md
│   ├── 02-convolution-opt.md
│   ├── 03-quantization.md
│   ├── 04-memory-opt.md
│   ├── 05-sve-vectorization.md
│   └── 06-bf16-sme.md
├── benchmarks/                # 性能测试脚本与结果
├── patches/                   # 按模块组织的优化补丁
│   ├── convolution/
│   ├── matmul/
│   ├── pooling/
│   ├── elementwise/
│   ├── quantization/
│   └── memory/
├── scripts/                   # 构建、测试、性能分析脚本
└── tests/                     # 正确性验证测试
```

## Optimization Roadmap

详见 docs/ 下各专题文档及 TASKS.md 任务分解。
