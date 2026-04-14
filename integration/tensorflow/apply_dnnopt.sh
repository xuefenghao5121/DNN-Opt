#!/bin/bash
# apply_dnnopt.sh - Apply DNN-Opt integration to TF build
# Run AFTER the base TF build completes successfully.
#
# This script modifies:
# 1. WORKSPACE - adds dnnopt as new_local_repository
# 2. workspace2.bzl - adds dnnopt patch to mkl_dnn_acl_compatible
# 3. third_party/mkl_dnn/mkldnn_acl.BUILD - adds DNNL_USE_DNNOPT and dnnopt dep
#
# Then rebuilds with: bazel build //tensorflow/tools/pip_package:build_pip_package

set -e
cd /root/tf-build

echo "=== Applying DNN-Opt integration ==="

# 1. Add dnnopt to WORKSPACE (before the workspace() function)
if ! grep -q "dnnopt" WORKSPACE; then
    cat >> WORKSPACE << 'EOF'

# DNN-Opt GEMM library for oneDNN integration
new_local_repository(
    name = "dnnopt",
    path = "/root/onednn-arm-opt",
    build_file = "//:dnnopt.BUILD",
)
EOF
    echo "  [OK] Added dnnopt to WORKSPACE"
else
    echo "  [SKIP] dnnopt already in WORKSPACE"
fi

# 2. Add patch to workspace2.bzl mkl_dnn_acl_compatible entry
if ! grep -q "onednn_dnnopt" tensorflow/workspace2.bzl; then
    sed -i '/onednn_acl_indirect_conv.patch",$/a\
            "//third_party/mkl_dnn:onednn_dnnopt.patch",' \
        tensorflow/workspace2.bzl
    echo "  [OK] Added dnnopt patch to workspace2.bzl"
else
    echo "  [SKIP] dnnopt patch already in workspace2.bzl"
fi

# 3. Create dnnopt.BUILD in root (if not exists)
if [ ! -f dnnopt.BUILD ]; then
    cp /root/onednn-arm-opt/BUILD.bazel dnnopt.BUILD
    echo "  [OK] Created dnnopt.BUILD"
else
    echo "  [SKIP] dnnopt.BUILD already exists"
fi

# 4. Modify mkldnn_acl.BUILD to add DNNL_USE_DNNOPT and dnnopt dep
if ! grep -q "DNNL_USE_DNNOPT" third_party/mkl_dnn/mkldnn_acl.BUILD; then
    # Add DNNL_USE_DNNOPT to defines
    sed -i 's/defines = \["DNNL_AARCH64_USE_ACL=1"\]/defines = ["DNNL_AARCH64_USE_ACL=1", "DNNL_USE_DNNOPT=1"]/' \
        third_party/mkl_dnn/mkldnn_acl.BUILD

    # Add dnnopt to deps
    sed -i 's/deps = \[@compute_library\/\/:arm_compute"\]/deps = ["@compute_library\/\/:arm_compute", "@dnnopt\/\/:dnnopt"]/' \
        third_party/mkl_dnn/mkldnn_acl.BUILD

    echo "  [OK] Modified mkldnn_acl.BUILD"
else
    echo "  [SKIP] DNNL_USE_DNNOPT already in mkldnn_acl.BUILD"
fi

echo ""
echo "=== Integration complete ==="
echo "To rebuild with dnnopt:"
echo "  bazel build --config=mkl_aarch64 --config=noaws --config=nogcp --config=nohdfs --config=nonccl --jobs=2 --distinct_host_configuration=false //tensorflow/tools/pip_package:build_pip_package"
