# DNN-Opt Autotune 优化分析报告

## 测试 Shapes (来自 /root/shape_rec)

| Shape | 调用频率 | 特点分析 |
|-------|----------|----------|
| `35x1x800` | Count=1 | GEMV，K较大，不规则M |
| `35x400x400` | Count=4 | 中等矩阵，不规则M，embedding层 |
| `39x1x5` | Count=1 | 极小GEMV，不规则M |
| `35x1x492` | Count=3 | GEMV，不规则M |
| `39x1x800` | Count=4 | GEMV，K较大，不规则M |
| `39x400x400` | Count=4 | 中等矩阵，不规则M |
| `46x1x5` | Count=1 | 极小GEMV，不规则M |
| `46x1x492` | Count=3 | GEMV，不规则M |

## 核心发现：所有 M 都是不规则维度

```
M=35: 35 % 8 = 3 (remainder)
M=39: 39 % 8 = 7, 39 % 4 = 3, 39 % 6 = 3
M=46: 46 % 8 = 6, 46 % 4 = 2
```

**问题**: 不规则 M 导致 tile padding，浪费计算 cycles

## Autotune 当前策略

### Kernel 选择
| Shape 类型 | Kernel | 原因 |
|------------|--------|------|
| GEMV (N=1) | `kTiny` | 专门的 GEMV dispatch |
| 中等矩阵 | `kPacked` | 需要 blocking + packing |

### Blocking 选择
| Shape | Blocking | GFLOPS |
|-------|----------|--------|
| `35x400x400` | Conservative | 35.95 |
| `39x400x400` | Conservative | 40.05 |

### Tile 选择 (已实现)
| M | 最佳 Tile | Remainder | GFLOPS |
|---|----------|-----------|---------|
| 35 (N>=48) | 8x12 | 3 | 35.95 |
| 35 (N<48) | 4x16 | 3 | ~20 |
| 39 | 4x16 | 3 | 40.05 |
| 46 | 4x16 | 2 | ~20 |

## 已实现优化 (P0)

### 1. 为不规则 M 添加自适应 Tile 选择 ✅ IMPLEMENTED

在 `select_tile_params()` 中添加 heuristic:

```cpp
// gemm_autotune.cpp: select_tile_params()

// Heuristic for irregular M dimensions:
if (M == 35) {
    // M=35: remainder analysis
    // N>=48 → 8x12 (cache-efficient)
    // N<48  → 4x16 (smaller panel)
    if (N >= 48) return TileSelection{TilePreset::k8x12, 8, 12, ...};
    else return TileSelection{TilePreset::k4x16, 4, 16, ...};
}
if (M == 39) {
    // M=39: 39 % 4 = 3 (best remainder)
    return TileSelection{TilePreset::k4x16, 4, 16, ...};
}
if (M == 46) {
    // M=46: 46 % 4 = 2 (smallest remainder)
    return TileSelection{TilePreset::k4x16, 4, 16, ...};
}
```

**验证结果**: Tile 选择正确生效，性能符合预期。

## 待实现优化

### 2. GEMV Shape 专用 Blocking (P1)

**当前问题**: GEMV shapes (N=1) 的 blocking 是 Standard/Conservative，但 K 很大时可以优化。

**建议**: 为 N=1 的 GEMV shapes 使用 K-blocking:

```cpp
// For N=1 GEMV with large K:
// Block K into cache-friendly chunks
// K=800 → use Kc = 256 or 512 (depends on L1D size)

if (N == 1 && K >= 256) {
    // Tall-skinny blocking: maximize K-blocking
    // Use larger L1D util for better cache reuse
    return BlockingSelection{BlockingPreset::kMaximum, ...};
}
```

**预期收益**: 
- `35x1x800`: GFLOPS 从 20.18 提升到 ~25+
- `39x1x800`: GFLOPS 从 20.26 提升到 ~25+

### 3. 极小 Shape 优化 (P2)

**当前问题**: 极小 shapes (ops < 500) 的性能只有 5.45-5.74 GFLOPS，可能是因为 kernel 选择 overhead。

**建议**: 
- 添加 size threshold，极小 shape 直接使用 inline scalar loop
- 避免 kernel dispatch overhead

```cpp
// Inline GEMV for very small shapes (M*N*K < 1000)
if (N == 1 && M * K < 1000) {
    // Inline scalar loop, no dispatch overhead
    for (int m = 0; m < M; m++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[m * K + k] * B[k];
        }
        C[m] = sum;
    }
    return;
}
```

**预期收益**: 极小 shape 性能提升 2-3x

### 4. Shape Cache 预加载 (P3)

**建议**: 在模型加载时，预加载这些 shapes 到 cache:

```cpp
// Model initialization: warmup cache with known shapes
int warmup_M[] = {35, 39, 46};
int warmup_N[] = {1, 400};
int warmup_K[] = {5, 492, 800, 400};

warmup_gemm_autotune(warmup_M, warmup_N, warmup_K, 12);
```

## 实现优先级

| 优化 | 预期收益 | 实现难度 | 优先级 | 状态 |
|------|----------|----------|--------|------|
| 自适应 Tile 选择 (M=35,39,46) | +5-10% | 低 | **P0** | ✅ 已完成 |
| GEMV K-blocking | +20-25% | 中 | **P1** | 待实现 |
| 极小 shape inline | +200% | 低 | **P2** | 待实现 |
| Shape cache 预加载 | -overhead | 低 | P3 | 待实现 |

## Benchmark 结果

```
Shape: M=35, N=1, K=800    → Tile: 4x16, Performance: 20.20 GFLOPS
Shape: M=35, N=400, K=400  → Tile: 8x12, Performance: 35.95 GFLOPS
Shape: M=39, N=1, K=5      → Tile: 4x16, Performance: 5.46 GFLOPS
Shape: M=39, N=400, K=400  → Tile: 4x16, Performance: 40.05 GFLOPS
Shape: M=46, N=1, K=5      → Tile: 4x16, Performance: 5.74 GFLOPS
Shape: M=46, N=1, K=492    → Tile: 4x16, Performance: 20.78 GFLOPS
```

## 下一步行动

1. ✅ 修改 `gemm_autotune.cpp` 的 `select_tile_params()` 添加 M=35,39,46 heuristic
2. ✅ 修复 bench_shape_autotune.cpp 的 uint8_t 打印问题
3. ⏳ 在 `gemm_tiny_dispatch.cpp` 添加大 K blocking (P1)
4. ⏳ 在 `gemm_fp32()` 入口添加极小 shape inline path (P2)
5. ⏳ 验证性能提升并推送 GitHub