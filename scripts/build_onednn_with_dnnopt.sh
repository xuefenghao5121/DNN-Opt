#!/bin/bash
# build_onednn_with_dnnopt.sh
# Build oneDNN with dnnopt as ARM GEMM supplementary patch.
#
# dnnopt patches oneDNN's sgemm to accelerate small/irregular matrix shapes.
# oneDNN handles large regular shapes natively — dnnopt only activates on weakness shapes.
#
# Prerequisites:
#   - Clang-15 installed (/usr/bin/clang++-15)
#   - oneDNN source (git clone https://github.com/oneapi-src/oneDNN)
#
# Usage:
#   ./scripts/build_onednn_with_dnnopt.sh [ONEDNN_SRC_DIR]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DNNOPT_ROOT="${DNNOPT_ROOT:-$(dirname "$SCRIPT_DIR")}"
ONEDNN_DIR="${1:-${ONEDNN_DIR:-/root/onednn}}"

echo "=== dnnopt + oneDNN Integration Build ==="
echo "  DNNOPT_ROOT: $DNNOPT_ROOT"
echo "  ONEDNN_DIR:  $ONEDNN_DIR"
echo ""

# Step 1: Build dnnopt
if [ ! -f "$DNNOPT_ROOT/build/src/libdnnopt_blas.so" ]; then
    echo "[1/3] Building dnnopt (Clang-15, Release)..."
    mkdir -p "$DNNOPT_ROOT/build"
    cd "$DNNOPT_ROOT/build"
    cmake .. -DCMAKE_CXX_COMPILER=clang++-15 -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)
else
    echo "[1/3] dnnopt already built"
fi

# Step 2: Build oneDNN with dnnopt patch
echo "[2/3] Building oneDNN with DNNL_AARCH64_USE_DNNOPT=ON..."
mkdir -p "$ONEDNN_DIR/build"
cd "$ONEDNN_DIR/build"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_AARCH64_USE_DNNOPT=ON \
    -DCMAKE_PREFIX_PATH="$DNNOPT_ROOT/build" \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_BUILD_EXAMPLES=OFF \
    -DDNNL_BUILD_TESTS=ON
cmake --build . -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "  oneDNN+dnnopt: $ONEDNN_DIR/build/src/libdnnl.so"
echo ""
echo "To benchmark vs upstream oneDNN:"
echo "  # Build upstream oneDNN separately (without dnnopt flags)"
echo "  # Then compare with bench_onednn_sgemm test"
echo "  LD_LIBRARY_PATH=$ONEDNN_DIR/build/src ./build/tests/bench_onednn_sgemm"
