# oneDNN Integration

Dnnopt integrates as a **supplementary patch** for oneDNN via the `dnnl_sgemm` interface.

## Design Rationale

Dnnopt does **not** replace oneDNN. It accelerates the shapes where oneDNN underperforms:

- M=1 GEMV (inference batch=1)
- M=2--7 small matrices (small-batch inference)
- Irregular/prime N (N=17, 37, 53...)
- Tall-skinny matrices (M=128, N=2--7)

oneDNN already achieves near-peak performance on large regular matrices (M>=64, aligned N/K). Dnnopt does not interfere with those shapes.

## How It Works

Dnnopt's `dnnl_sgemm` wrapper sits in oneDNN's GEMM dispatch path:

1. Detects shape parameters (M, N, K)
2. For weakness shapes (small M, irregular N): routes to dnnopt optimized kernels
3. For strong shapes (large regular matrices): falls back to oneDNN native implementation

## Build Instructions

### Prerequisites

- Clang-15 (`/usr/bin/clang++-15`)
- oneDNN source: `git clone https://github.com/oneapi-src/oneDNN`

### Step 1: Build dnnopt

```bash
cd onednn-arm-opt && mkdir build && cd build
cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
```

### Step 2: Build oneDNN with dnnopt

```bash
cd /path/to/onednn && mkdir build && cd build
cmake .. -DDNNL_AARCH64_USE_DNNOPT=ON \
         -DCMAKE_PREFIX_PATH=/path/to/dnnopt/build \
         -DCMAKE_BUILD_TYPE=Release \
         -DDNNL_CPU_RUNTIME=OMP \
         -DDNNL_BUILD_EXAMPLES=OFF \
         -DDNNL_BUILD_TESTS=ON
cmake --build . -j$(nproc)
```

### Quick Build Script

```bash
./scripts/build_onednn_with_dnnopt.sh [/path/to/onednn]
```

### Verify

```bash
LD_LIBRARY_PATH=/path/to/onednn/build/src ./build/tests/bench_onednn_sgemm
```

## Files Modified in oneDNN

- `cmake/dnnopt.cmake` -- CMake find module for dnnopt
- `src/cpu/aarch64/dnnopt_gemm_wrapper.hpp` -- sgemm dispatch adapter
- `src/cpu/gemm/gemm.cpp` -- route to dnnopt in extended_sgemm

## Performance (Neoverse N2, 2 cores @ 3GHz)

### 55 shapes: 54 wins / 1 loss vs oneDNN-native

| Shape Category | Example | Speedup |
|---------------|---------|---------|
| M=1 GEMV | Inference batch=1 | 4--89x |
| M=2--7 | Small-batch inference | 3--190x |
| Prime N | N=17, 37, 53 | 1.5--2.6x |
| Tall-skinny | M=128, N=2--7 | 2.1--4.2x |
| Irregular M+N | M=16, N=23, 47 | 1.7--1.9x |

### End-to-End Inference

| Model | oneDNN | +dnnopt | Speedup |
|-------|--------|---------|---------|
| CVR batch=1 | 691 us | 52 us | **13.3x** |
| CVR batch=4 | 1861 us | 85 us | **21.9x** |
| BERT-small batch=1 | 7698 us | 1145 us | **6.7x** |
| BERT-small batch=4 | 27215 us | 2365 us | **11.5x** |
| LLM batch=1 | 260927 us | 44016 us | **5.9x** |
| LLM batch=4 | 531784 us | 148018 us | **3.6x** |

### Measurement Methodology

- Each shape: 100 iterations, median GFLOPS reported
- Warmup: 10 iterations before measurement
- Thread pinning: `OMP_PROC_BIND=true`, `GOMP_CPU_AFFINITY=0-1`
- Frequency: fixed at 3 GHz (performance governor)
- oneDNN version: v3.7 (upstream), built with same compiler flags
