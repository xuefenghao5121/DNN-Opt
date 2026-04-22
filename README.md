# DNN-Opt: oneDNN Supplementary Patch for ARM Inference

**Version 0.9.28** | ARM GEMM optimization for oneDNN weak spots.

Dnnopt accelerates shapes where oneDNN underperforms — small M, irregular N, tall-skinny — while falling back to oneDNN for large regular shapes.

## Performance Summary

| Category | Speedup vs oneDNN |
|----------|------------------|
| M=1 (batch-1) | **4-89x** |
| M=2-7 (small batch) | **3-190x** |
| Tall-skinny | **2-4x** |
| Irregular N (prime) | **1.5-2.6x** |

**Peak performance**: 45 GFLOPS (94% of theoretical on Neoverse N2)

## Quick Start

```bash
# Build (requires Clang-15 on AArch64)
cmake -B build -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run tests
cd build && ctest --output-on-failure
```

## Integration with oneDNN

```bash
# Apply patch to oneDNN
git clone https://github.com/oneapi-src/oneDNN
cd oneDNN
git apply DNN-Opt/integration/onednn/0001-dnnopt-integration.patch

# Build oneDNN with dnnopt
cmake -B build \
  -DDNNL_AARCH64_USE_DNNOPT=ON \
  -DDNNOPT_ROOT=/path/to/DNN-Opt \
  -DDNNL_BLAS_VENDOR=OPENBLAS
cmake --build build
```

## API

```cpp
#include <dnnopt/gemm/gemm.h>

// Direct call
dnnopt::gemm_fp32(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

// Autotune (optional)
setenv("DNNOPT_AUTOTUNE", "1", 1);
```

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `DNNOPT_BUILD_TESTS` | OFF | Build test suite |
| `DNNOPT_BUILD_BLAS` | ON | Build BLAS wrapper (libdnnopt_blas.so) |
| `DNNOPT_ENABLE_SME` | OFF | Enable SME kernels (requires SME hardware) |

## Project Structure

```
include/dnnopt/     # Public headers
  gemm/             # GEMM API
  conv/             # Convolution API
  autotune/         # Autotune framework
src/                # Implementation
integration/        # oneDNN/TensorFlow patches
```

## License

Apache 2.0