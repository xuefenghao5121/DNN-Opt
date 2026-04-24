# oneDNN Integration

DNN-Opt integrates as a **supplementary patch** for oneDNN via the `dnnl_sgemm` interface.

## DNN-Opt Patches

| Patch File | Description |
|------------|-------------|
| `autotune_irregular_m_tile.patch` | Tile selection optimization for irregular M (35, 39, 46) |
| `onednn_v3.7_dnnopt.patch` | oneDNN v3.7 basic integration |
| `onednn_v3.7_brgemm_integration.patch` | oneDNN v3.7 brgemm hybrid dispatch |

## Supported oneDNN Versions

| Version | Patch File | Status |
|---------|------------|--------|
| **v3.7 + brgemm** | `onednn_v3.7_brgemm_integration.patch` | ✅ Tested (SVE_512/256) |
| **v3.7** | `onednn_v3.7_dnnopt.patch` | ✅ Tested |
| **v3.4** | `onednn_v3.4_dnnopt.patch` | ✅ Tested |
| v3.x (upstream) | `0001-dnnopt-integration.patch` | ✅ Tested |

## v3.7 Integration

oneDNN v3.7 requires special handling due to parameter swap in `dnnl_sgemm`:

```c
// dnnl_sgemm internally swaps parameters:
extended_sgemm(&transb, &transa, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)
```

This causes stride mismatch for non-square matrices (M ≠ K). Our patch handles this via **implicit transpose duality**.

### v3.7 + brgemm_matmul Integration (SVE_512/256 servers)

**重要**: 对于支持 SVE_512 或 SVE_256 的服务器（如 Neoverse V1/V2），oneDNN v3.7 会选择 `brgemm_matmul_t` 而非 `gemm_f32_matmul_t`，导致 DNN-Opt 不生效。

此 patch 在 `brgemm_matmul_t::execute_body()` 中添加 **hybrid dispatch**：
- 小矩阵/不规则矩阵：自动调用 `dnnopt::gemm_fp32()`
- 大矩阵：继续使用 brgemm kernel

#### Shape 分类规则

DNN-Opt 自动生效的 shapes：
```cpp
// should_use_dnnopt_shape(M, N, K) 返回 true 的条件:
// - M <= 16: tiny/small_m kernels
// - M <= 32 && N <= 128 && K <= 128: small square-ish
// - M*N*K < 32768: very small total ops
// - M <= 64 && !is_pow2(M): irregular M dimensions
```

#### Build for v3.7 with brgemm integration

```bash
# Step 1: Build DNN-Opt with Clang-15
cd /path/to/onednn-arm-opt
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
make dnnopt_core -j$(nproc)

# Step 2: Apply BOTH patches to oneDNN v3.7
cd /path/to/onednn_v3.7
git apply /path/to/onednn-arm-opt/integration/onednn/onednn_v3.7_dnnopt.patch
git apply /path/to/onednn-arm-opt/integration/onednn/onednn_v3.7_brgemm_integration.patch

# Step 3: Build oneDNN with DNN-Opt (must use Clang-15)
mkdir build_patched && cd build_patched
cmake .. \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DDNNOPT_ROOT=/path/to/onednn-arm-opt \
    -DDNNL_BUILD_TESTS=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make benchdnn -j$(nproc)
```

#### Verify brgemm integration

```bash
# 使用 verbose 模式确认 ISA 和 kernel
DNNL_VERBOSE=1 LD_LIBRARY_PATH=src ./tests/benchdnn/benchdnn \
    --matmul --mode=P --dt=f32 16x32:32x16 2>&1 | head -15

# 输出解读:
# - "isa:AArch64 SVE (xxx bits)" → 显示服务器 ISA
# - "matmul,gemm:jit:f32" → gemm_f32_matmul → DNN-Opt生效
# - "matmul,brg:sve_512" → brgemm_matmul → 小矩阵自动使用 DNN-Opt

# 检查 dnnopt symbols
nm src/libdnnl.so.3.7 | grep dnnopt | head -10
```

#### Files Modified in brgemm integration

| File | Modification |
|------|--------------|
| `src/cpu/aarch64/matmul/brgemm_matmul_utils.hpp` | Add `should_use_dnnopt_shape()` and `use_dnnopt` flag |
| `src/cpu/aarch64/matmul/brgemm_matmul.hpp` | Add `should_use_dnnopt()` and `execute_dnnopt()` methods |
| `src/cpu/aarch64/matmul/brgemm_matmul.cpp` | Implement DNN-Opt dispatch in `execute_body()` |
| `cmake/dnnopt.cmake` | Fix library linking configuration |

### Build for v3.7 (basic, without brgemm integration)

**重要**: DNN-Opt 和 oneDNN 必须使用相同的编译器 (Clang-15) 以保证 ABI 兼容。

```bash
# Step 1: Build DNN-Opt with Clang-15
cd /path/to/onednn-arm-opt
mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Step 2: Apply patch to oneDNN v3.7
cd /path/to/onednn_v3.7
git apply /path/to/onednn-arm-opt/integration/onednn/onednn_v3.7_dnnopt.patch

# Step 3: Build oneDNN with DNN-Opt (must use Clang-15)
mkdir build && cd build
cmake .. \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DDNNOPT_ROOT=/path/to/onednn-arm-opt \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) dnnl
```

### Performance (Neoverse N2)

**测试日期**: 2026-04-23

| Shape | Baseline v3.7 | +DNN-Opt | Status |
|-------|---------------|----------|--------|
| M=1, N=1024, K=4096 | FAILED | 16.88 GFLOPS | **FIXED** |
| M=2 | FAILED | 14.74 GFLOPS | **FIXED** |
| M=4 | FAILED | 31.79 GFLOPS | **FIXED** |
| M=8 | FAILED | 17.76 GFLOPS | **FIXED** |
| M=16 | FAILED | 22.33 GFLOPS | **FIXED** |
| M=32 | FAILED | 32.78 GFLOPS | **FIXED** |
| M=64 | FAILED | 36.08 GFLOPS | **FIXED** |
| M=128 | FAILED | 41.25 GFLOPS | **FIXED** |
| M=256 | FAILED | 42.64 GFLOPS | **FIXED** |
| M=512 | FAILED | 44.63 GFLOPS | **FIXED** |
| M=1024, N=1024, K=1024 | 41.87 GFLOPS | 44.97 GFLOPS | +7% |
| M=3 (prime) | FAILED | 16.92 GFLOPS | **FIXED** |
| M=5 (prime) | FAILED | 13.98 GFLOPS | **FIXED** |
| M=7 (prime) | FAILED | 13.87 GFLOPS | **FIXED** |
| M=11 (prime) | FAILED | 24.23 GFLOPS | **FIXED** |

**Baseline v3.7**: 14/15 shapes FAILED (STATUS=2 stride mismatch)
**+DNN-Opt**: 15/15 shapes OK

### Benchmark Scripts

测试脚本位于 oneDNN v3.7 的 `scripts/` 目录：

```bash
cd /path/to/onednn_v3.7/scripts
./bench_compare_direct.sh    # 直接 dnnl_sgemm 对比 (推荐)
./bench_compare_benchdnn.sh  # benchdnn matmul 测试
```

## Using benchdnn with DNN-Opt

benchdnn 是 oneDNN 内置的基准测试工具，可用于验证 DNN-Opt 性能。

### 1. Build benchdnn

```bash
cd /path/to/onednn/build_patched
make benchdnn -j$(nproc)

# 验证 benchdnn 已编译
ls -l tests/benchdnn/benchdnn
```

### 2. Set Environment

```bash
# 指定 libdnnl.so 路径
export LD_LIBRARY_PATH=/path/to/onednn/build_patched/src:$LD_LIBRARY_PATH
```

### 3. Run MatMul/GEMM Tests

```bash
# 进入 build 目录
cd /path/to/onednn/build_patched

# 单个 shape 测试
./tests/benchdnn/benchdnn matmul --batch 1,1024,4096:1024,4096

# 批量测试
./tests/benchdnn/benchdnn matmul --batch 1,2,4,8,16,32:1024:4096

# Square matrix
./tests/benchdnn/benchdnn matmul --batch 1024,1024,1024:1024,1024

# 性能模式 (只测性能，跳过 correctness)
./tests/benchdnn/benchdnn --mode=P matmul --batch 1,1024,4096:1024,4096

# 详细输出
./tests/benchdnn/benchdnn -v matmul --batch 1,1024,4096:1024,4096
```

### 4. Compare Baseline vs DNN-Opt

```bash
# Baseline (无 DNN-Opt)
export LD_LIBRARY_PATH=/path/to/onednn/build_baseline/src:$LD_LIBRARY_PATH
./tests/benchdnn/benchdnn --mode=P matmul --batch 1,1024,4096:1024,4096

# DNN-Opt
export LD_LIBRARY_PATH=/path/to/onednn/build_patched/src:$LD_LIBRARY_PATH
./tests/benchdnn/benchdnn --mode=P matmul --batch 1,1024,4096:1024,4096
```

### 5. Verify DNN-Opt Integration

```bash
# 检查 libdnnl.so 是否包含 dnnopt symbols
nm -D /path/to/onednn/build_patched/src/libdnnl.so.3 | grep dnnopt

# 或者检查是否定义了 DNNL_USE_DNNOPT
nm -D /path/to/onednn/build_patched/src/libdnnl.so.3 | grep DNNL_USE_DNNOPT
```

### 6. Common benchdnn Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--batch` | Shape specification | `--batch M,N,K:N,K` |
| `--mode=P` | Performance mode only | Skip correctness check |
| `--mode=C` | Correctness mode only | Only verify results |
| `-v` | Verbose output | Show detailed info |
| `--mb` | Mini-batch size | `--mb 1,4,8` |
| `--tag` | Memory format | `--tag ab` |

### 7. Expected Results

**Baseline oneDNN (无 DNN-Opt)**:
- 非 square matrix 会报 STATUS=2 错误
- Square matrix (M=N=K) 性能正常

**oneDNN + DNN-Opt**:
- 所有 shape 正常执行
- 小矩阵 (M<32) 性能显著提升
- 大矩阵性能与 baseline 相当或略优

## v3.4 Integration

oneDNN v3.4 has the same parameter swap pattern as v3.7:

```c
// In src/common/gemm.cpp: same parameter swap
extended_sgemm(&transb, &transa, &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc)
```

### Build for v3.4

```bash
# Step 1: Apply patch to oneDNN v3.4
cd /path/to/onednn_v3.4
git apply /path/to/onednn-arm-opt/integration/onednn/onednn_v3.4_dnnopt.patch

# Step 2: Build oneDNN with DNN-Opt (must use Clang-15)
mkdir build_patched && cd build_patched
cmake .. \
    -DCMAKE_CXX_COMPILER=clang++-15 \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DDNNOPT_ROOT=/path/to/onednn-arm-opt \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc) dnnl
```

### Performance (Neoverse N2)

**测试日期**: 2026-04-23

| Shape | Baseline v3.4 | +DNN-Opt | Status |
|-------|---------------|----------|--------|
| M=1, N=1024, K=4096 | FAILED | 15.03 GFLOPS | **FIXED** |
| M=2 | FAILED | 13.33 GFLOPS | **FIXED** |
| M=4 | FAILED | 31.50 GFLOPS | **FIXED** |
| M=8 | FAILED | 16.69 GFLOPS | **FIXED** |
| M=16 | FAILED | 19.89 GFLOPS | **FIXED** |
| M=32 | FAILED | 24.76 GFLOPS | **FIXED** |
| M=64 | FAILED | 33.27 GFLOPS | **FIXED** |
| M=128 | FAILED | 41.44 GFLOPS | **FIXED** |
| M=256 | FAILED | 42.52 GFLOPS | **FIXED** |
| M=512 | FAILED | 44.64 GFLOPS | **FIXED** |
| M=1024, N=1024, K=1024 | 16.59 GFLOPS | 45.08 GFLOPS | **+171%** |
| M=3 (prime) | FAILED | 16.93 GFLOPS | **FIXED** |
| M=5 (prime) | FAILED | 13.95 GFLOPS | **FIXED** |
| M=7 (prime) | FAILED | 13.28 GFLOPS | **FIXED** |
| M=11 (prime) | FAILED | 23.90 GFLOPS | **FIXED** |

**Baseline v3.4**: 14/15 shapes FAILED (STATUS=2 stride mismatch)
**+DNN-Opt**: 15/15 shapes OK

## Design Rationale

DNN-Opt does **not** replace oneDNN. It accelerates the shapes where oneDNN underperforms:

- M=1 GEMV (inference batch=1)
- M=2--7 small matrices (small-batch inference)
- Irregular/prime N (N=17, 37, 53...)
- Tall-skinny matrices (M=128, N=2--7)

For large regular matrices (M>=32), dnnopt falls back to OpenBLAS (cblas_sgemm), matching upstream oneDNN's behavior.

## Dispatch Strategy

```
dnnl_sgemm() call
       |
       v
 Shape Classification (M, N, K, transa, transb)
       |
       +-- Small-M (M < 32) or irregular (N/K < 16) --> dnnopt kernels
       |
       +-- Large regular (M >= 32) --> OpenBLAS cblas_sgemm
       |
       +-- Final fallback --> ref_gemm
```

## Build Instructions (v3.x upstream)

### Prerequisites

- **dnnopt**: Clang-15 (`/usr/bin/clang++-15`)
- **OpenBLAS**: `yum install openblas-devel` (for large matrix fallback)
- **oneDNN source**: `git clone https://github.com/oneapi-src/oneDNN`

### Step 1: Build dnnopt

```bash
cd onednn-arm-opt && mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Step 2: Apply Patch to oneDNN

```bash
cd /path/to/onednn
git apply /path/to/onednn-arm-opt/integration/onednn/0001-dnnopt-integration.patch
```

### Step 3: Build oneDNN with dnnopt + OpenBLAS

```bash
cd /path/to/onednn && mkdir build && cd build
cmake .. \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DDNNOPT_ROOT=/path/to/onednn-arm-opt \
    -DDNNL_BLAS_VENDOR=OPENBLAS \
    -DBLAS_INCLUDE_DIR=/usr/include/openblas \
    -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_CPU_RUNTIME=OMP
cmake --build . -j$(nproc)
```

**Note**: OpenBLAS is required for large matrix fallback. Without it, large matrices would fall back to slow `ref_gemm`.

### Quick Build Script

```bash
./scripts/build_onednn_with_dnnopt.sh [/path/to/onednn]
```

### Verify

```bash
# Check library built
ls -lh build/src/libdnnl.so.*

# Run benchmark
LD_LIBRARY_PATH=build/src ./build/tests/bench_onednn_sgemm
```

## Performance (Neoverse N2, 2 cores @ 3GHz)

### dnnopt + OpenBLAS vs upstream oneDNN (v3.x)

| Shape | dnnopt+OpenBLAS | upstream | Speedup |
|-------|-----------------|----------|---------|
| CVR embedding b1 | 11.91 GF | 4.36 GF | **2.7x** |
| CVR embedding b4 | 28.63 GF | 4.69 GF | **6.1x** |
| LLM qkv b1 | 19.71 GF | 10.54 GF | **1.87x** |
| LLM qkv b4 | 38.70 GF | 29.53 GF | **1.31x** |
| BERT qkv b128 | 43.15 GF | 62.94 GF | 0.69x* |
| **Average** | **31.18 GF** | ~25 GF | **1.24x** |

*Large-M shapes use OpenBLAS fallback, which matches upstream behavior.

### End-to-End Inference

| Model | oneDNN | +dnnopt | Speedup |
|-------|--------|---------|---------|
| CVR batch=1 | 691 us | 52 us | **13.3x** |
| CVR batch=4 | 1861 us | 85 us | **21.9x** |
| BERT-small batch=1 | 7698 us | 1145 us | **6.7x** |
| BERT-small batch=4 | 27215 us | 2365 us | **11.5x** |
| LLM batch=1 | 260927 us | 44016 us | **5.9x** |
| LLM batch=4 | 531784 us | 148018 us | **3.6x** |

## Files Modified in oneDNN

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Include dnnopt.cmake |
| `cmake/dnnopt.cmake` | Find and link dnnopt library |
| `cmake/options.cmake` | Add DNNL_AARCH64_USE_DNNOPT option |
| `src/cpu/aarch64/dnnopt_gemm_wrapper.hpp` | sgemm dispatch adapter |
| `src/cpu/gemm/gemm.cpp` | Shape-based dispatch logic |

## Key Design Decisions

### Why OpenBLAS Fallback?

oneDNN aarch64 lacks `gemm_driver` (only x64/ppc64 have it). Without OpenBLAS, large matrices would fall back to slow `ref_gemm`. Using OpenBLAS matches upstream oneDNN's behavior.

### Why Not ACL?

TensorFlow's bazel cache contains ACL v31.0.1, but oneDNN requires ACL v52.4+ for `CpuActivation.h` API. Version mismatch prevents integration.

### Why Not BRGEMM?

oneDNN's BRGEMM has JIT compilation overhead per call, making it unsuitable for single-batch GEMM calls.

## Troubleshooting

### `DNNOPT_ROOT not found`

```bash
# Set DNNOPT_ROOT in cmake command
cmake .. -DDNNOPT_ROOT=/path/to/onednn-arm-opt ...

# Or set environment variable
export DNNOPT_ROOT=/path/to/onednn-arm-opt
cmake .. -DDNNL_AARCH64_USE_DNNOPT=ON ...
```

### `OpenBLAS not found`

```bash
# Install OpenBLAS
yum install openblas-devel  # CentOS/RHEL
apt-get install libopenblas-dev  # Ubuntu/Debian

# Specify include path
cmake .. -DBLAS_INCLUDE_DIR=/usr/include/openblas ...
```

### Patch conflicts

```bash
# Check oneDNN version
cd /path/to/onednn
git log --oneline -1

# For v3.7, use onednn_v3.7_dnnopt.patch
git apply /path/to/onednn-arm-opt/integration/onednn/onednn_v3.7_dnnopt.patch

# For v3.x upstream, use 0001-dnnopt-integration.patch
git apply /path/to/onednn-arm-opt/integration/onednn/0001-dnnopt-integration.patch
```

## References

- [oneDNN GitHub](https://github.com/oneapi-src/oneDNN)
- [OpenBLAS GitHub](https://github.com/OpenMathLib/OpenBLAS)
- [autoGEMM Paper (SC'24)](https://github.com/wudu98/autoGEMM)