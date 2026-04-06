#!/bin/bash
# oneDNN ARM 优化版本构建脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ONEDNN_SRC="${PROJECT_DIR}/third_party/oneDNN"
BUILD_DIR="${PROJECT_DIR}/build"
INSTALL_DIR="${PROJECT_DIR}/install"

# ============================================================
# 配置选项
# ============================================================
BUILD_TYPE="${BUILD_TYPE:-Release}"
USE_ACL="${USE_ACL:-ON}"
ACL_ROOT="${ACL_ROOT:-/opt/arm/acl}"
AARCH64_TARGET="${AARCH64_TARGET:-native}"  # native / armv8.2-a / armv8.6-a+sve2+bf16+i8mm

# 编译器选择
CC="${CC:-gcc}"
CXX="${CXX:-g++}"

echo "============================================"
echo " oneDNN ARM Optimized Build"
echo "============================================"
echo " Build type:   ${BUILD_TYPE}"
echo " ACL support:  ${USE_ACL}"
echo " Target arch:  ${AARCH64_TARGET}"
echo " Compiler:     ${CXX}"
echo "============================================"

# ============================================================
# Step 1: 获取 oneDNN 源码
# ============================================================
if [ ! -d "${ONEDNN_SRC}" ]; then
    echo "[1/4] Cloning oneDNN..."
    mkdir -p "${PROJECT_DIR}/third_party"
    git clone --depth 1 https://github.com/uxlfoundation/oneDNN.git "${ONEDNN_SRC}"
else
    echo "[1/4] oneDNN source found at ${ONEDNN_SRC}"
fi

# ============================================================
# Step 2: 获取 ARM Compute Library (可选)
# ============================================================
if [ "${USE_ACL}" = "ON" ] && [ ! -d "${ACL_ROOT}" ]; then
    echo "[2/4] ACL not found at ${ACL_ROOT}. Building ACL..."
    ACL_SRC="${PROJECT_DIR}/third_party/ComputeLibrary"
    if [ ! -d "${ACL_SRC}" ]; then
        git clone --depth 1 https://github.com/ARM-software/ComputeLibrary.git "${ACL_SRC}"
    fi
    cd "${ACL_SRC}"
    scons arch=armv8.2-a neon=1 opencl=0 os=linux \
          build=native extra_cxx_flags="-march=${AARCH64_TARGET}" \
          Werror=0 -j$(nproc)
    ACL_ROOT="${ACL_SRC}/build"
    cd "${PROJECT_DIR}"
else
    echo "[2/4] ACL configuration: USE_ACL=${USE_ACL}"
fi

# ============================================================
# Step 3: 构建 oneDNN
# ============================================================
echo "[3/4] Building oneDNN..."
mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"
    -DCMAKE_C_COMPILER="${CC}"
    -DCMAKE_CXX_COMPILER="${CXX}"
    # ARM 特定选项
    -DDNNL_AARCH64_USE_ACL="${USE_ACL}"
    # 性能关键编译选项
    -DCMAKE_CXX_FLAGS="-O3 -march=${AARCH64_TARGET} -mtune=native -ffast-math -fno-math-errno"
    # 仅构建推理常用算子 (减少编译时间和二进制大小)
    # -DONEDNN_ENABLE_PRIMITIVE="CONVOLUTION;MATMUL;INNER_PRODUCT;POOLING;ELTWISE;BATCH_NORMALIZATION;LAYER_NORMALIZATION;SOFTMAX;REORDER;BINARY;REDUCTION"
    # 构建 benchmark 工具
    -DDNNL_BUILD_TESTS=ON
    -DDNNL_BUILD_EXAMPLES=ON
)

if [ "${USE_ACL}" = "ON" ]; then
    CMAKE_ARGS+=(-DACL_ROOT_DIR="${ACL_ROOT}")
fi

cmake "${ONEDNN_SRC}" "${CMAKE_ARGS[@]}"
make -j$(nproc)

# ============================================================
# Step 4: 安装
# ============================================================
echo "[4/4] Installing..."
make install

echo ""
echo "============================================"
echo " Build Complete!"
echo " Install dir: ${INSTALL_DIR}"
echo "============================================"
echo ""
echo "To run benchmarks:"
echo "  cd ${BUILD_DIR}"
echo "  ./tests/benchdnn/benchdnn --conv --batch=inputs/conv/shapes_resnet_50"
echo "  ./tests/benchdnn/benchdnn --matmul --batch=inputs/matmul/shapes_transformer"
