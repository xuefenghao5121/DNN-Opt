# TensorFlow + oneDNN + DNN-Opt Integration

This directory contains files for integrating DNN-Opt into TensorFlow's oneDNN backend.

## Files

- `acl_sve_fix.patch` - Fix for ACL (Compute Library) SVE compilation issue
- `onednn_dnnopt.patch` - oneDNN patch to add DNN-Opt dispatch
- `apply_dnnopt.sh` - Automation script to apply patches
- `dnnopt.BUILD` - Bazel BUILD file for dnnopt library

## Prerequisites

- TensorFlow 2.16+ source tree
- Bazel 6.5.0+
- Clang 15.0.7+
- Neoverse N2 or similar ARMv8.5-a platform

## Integration Steps

1. Copy integration files to TensorFlow source tree:
   ```bash
   cp acl_sve_fix.patch <tf-source>/third_party/compute_library/
   cp onednn_dnnopt.patch <tf-source>/third_party/mkl_dnn/
   cp apply_dnnopt.sh <tf-source>/
   cp dnnopt.BUILD <tf-source>/
   ```

2. Run integration script:
   ```bash
   cd <tf-source>
   ./apply_dnnopt.sh
   ```

3. Build TensorFlow with oneDNN+dnnopt:
   ```bash
   bazel build --config=mkl_aarch64 --jobs=2 \
       //tensorflow/tools/pip_package:build_pip_package
   ```

## ACL SVE Fix Details

### Problem
ACL's `arm_compute` core library defines `ARM_COMPUTE_ENABLE_SVE` in `local_defines`
but compiles with `-march=armv8.2-a+fp16` (without SVE), causing compilation errors.

### Solution
The patch removes:
1. `ARM_COMPUTE_ENABLE_SVEF32MM` from `common_defines`
2. `ARM_COMPUTE_ENABLE_SVE` from `arm_compute` core library's `local_defines`

### Verified Results
- ACL compilation: 1268 processes, 596s ✓
- oneDNN + ACL + dnnopt: 428 processes, 664s ✓
- Generated library: `libmkl_dnn_acl.so` (31MB)

## Build Configuration

- Compiler: Clang 15.0.7
- Architecture: ARMv8.2-a+fp16 (ACL), ARMv8.5-a+bf16+dotprod+fp16+i8mm (dnnopt)
- Threading: OpenMP (libgomp)
- Bazel jobs: 2 (Neoverse N2 dual-core)

## Performance Notes

DNN-Opt shows 54 wins / 1 loss vs oneDNN standalone on inference workloads:
- CVR: 13.3x (batch=1), 21.9x (batch=4)
- BERT-small: 6.7x (batch=1), 11.5x (batch=4)
- LLM: 5.9x (batch=1), 3.6x (batch=4)

## Next Steps

- Run end-to-end TensorFlow benchmarks with oneDNN+dnnopt
- Compare against baseline TensorFlow (Eigen/XNNPACK)
- Validate GEMM dispatch correctness
