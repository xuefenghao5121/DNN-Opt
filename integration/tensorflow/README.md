# TensorFlow + oneDNN + DNN-Opt Integration

This directory contains guidance for integrating DNN-Opt into TensorFlow via oneDNN backend.

## Status

**Tested**: TensorFlow 2.16.1, Neoverse N2 (2 cores @ 3GHz), Ubuntu 22.04

## Integration Approach

DNN-Opt integrates into TensorFlow through oneDNN:

```
TensorFlow XLA → oneDNN (dnnl_sgemm) → DNN-Opt dispatch
```

1. Build TensorFlow with oneDNN backend (`--config=mkl_aarch64_threadpool`)
2. Apply oneDNN patch from `../onednn/0001-dnnopt-integration.patch`
3. DNN-Opt handles small-M GEMM, OpenBLAS handles large-M

## Prerequisites

- TensorFlow 2.16+ source tree
- Bazel 6.5.0+
- Clang 15.0.7+
- **LLD linker** (`yum install lld`) - required for ARM64 TLSLE
- OpenBLAS (`yum install openblas-devel`)
- Neoverse N2 or similar ARMv8.5-a platform

## Build Steps

### 1. Clone and Configure TensorFlow

```bash
git clone https://github.com/tensorflow/tensorflow.git tf-build
cd tf-build
./configure  # Accept defaults, no GPU/cloud services
```

### 2. Build with oneDNN Threadpool (CRITICAL)

**IMPORTANT**: Use `mkl_aarch64_threadpool`, NOT `mkl_aarch64`.

| Config | `OneDnnThreadPool` inheritance | Result |
|--------|-------------------------------|--------|
| `mkl_aarch64` | NO (stub only) | ❌ XLA compile fails |
| `mkl_aarch64_threadpool` | YES (proper) | ✅ Works |

```bash
bazel build --config=mkl_aarch64_threadpool \
  --config=noaws --config=nogcp --config=nohdfs --config=nonccl \
  --jobs=2 --distinct_host_configuration=false \
  --linkopt=-fuse-ld=lld \
  //tensorflow:libtensorflow.so //tensorflow:libtensorflow_framework.so
```

### 3. Apply DNN-Opt Patch (Optional)

For additional small-M acceleration:

```bash
# After baseline oneDNN build succeeds, patch TF's internal oneDNN
cd third_party/mkl_dnn
git apply /path/to/DNN-Opt/integration/onednn/0001-dnnopt-integration.patch

# Add dnnopt.BUILD for bazel
cp /path/to/DNN-Opt/integration/tensorflow/dnnopt.BUILD .

# Rebuild (uses bazel cache)
bazel build --config=mkl_aarch64_threadpool --linkopt=-fuse-ld=lld \
  //tensorflow:libtensorflow.so
```

## Known Issues & Solutions

### 1. XLA MatMul vs Threadpool Interface Mismatch

When using OpenMP config (`--config=mkl_aarch64`), XLA fails:

```
error: no matching function for call to 'MakeOneDnnStream'
```

**Solution**: Use `--config=mkl_aarch64_threadpool` instead.

### 2. ARM64 TLSLE Linker Error

Gold linker fails on large binaries:

```
ld.gold: error: unsupported reloc 549/551 in non-static TLSLE mode
```

**Solution**: Install LLD and use `--linkopt=-fuse-ld=lld`.

### 3. tfcompile Binary Fails

`tfcompile` requires PIC which LLD doesn't support for this case. Use shared library targets instead.

## Performance Results

### TensorFlow + oneDNN (Baseline)

| Configuration | GFLOPS | Notes |
|---------------|--------|-------|
| Pre-built tensorflow-aarch64 | 13.24 | Eigen backend |
| Compiled TF + oneDNN | 20.29 | 1.53x faster |

### TensorFlow + oneDNN + DNN-Opt

| Shape | dnnopt+OpenBLAS | upstream | Speedup |
|-------|-----------------|----------|---------|
| CVR b1 | 11.91 | 4.36 | **2.7x** |
| CVR b4 | 28.63 | 4.69 | **6.1x** |
| LLM b1 | 19.71 | 10.54 | **1.87x** |

## Verification

```bash
export LD_LIBRARY_PATH=$PWD/bazel-out/aarch64-opt/bin/tensorflow:$LD_LIBRARY_PATH
python3 -c "import tensorflow as tf; print('TF:', tf.__version__)"
```

## Files

- `dnnopt.BUILD` - Bazel BUILD for dnnopt library (if needed)
- `acl_sve_fix.patch` - ACL SVE compilation fix (legacy)

## References

- [TensorFlow Configure](https://www.tensorflow.org/install/source)
- oneDNN in TF: `third_party/mkl_dnn/`
- XLA oneDNN matmul: `third_party/xla/xla/service/cpu/onednn_matmul.cc`
- Threadpool interface: `third_party/tsl/tsl/util/onednn_threadpool.h`