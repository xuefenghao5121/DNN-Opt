#!/bin/bash
# build_onednn_with_dnnopt.sh
# Build oneDNN with DNN-Opt as BLAS backend for ARM platforms.
#
# Prerequisites:
#   - DNN-Opt built (cmake --build . in dnnopt/build)
#   - oneDNN source cloned (git clone https://github.com/oneapi-src/oneDNN)
#
# Usage:
#   ./scripts/build_onednn_with_dnnopt.sh [ONEDNN_SRC_DIR]
#
# Environment:
#   DNNOPT_ROOT  — path to dnnopt repo root (default: script's parent dir)
#   ONEDNN_DIR   — path to oneDNN source (default: first arg or /tmp/onednn)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DNNOPT_ROOT="${DNNOPT_ROOT:-$(dirname "$SCRIPT_DIR")}"
ONEDNN_DIR="${1:-${ONEDNN_DIR:-/tmp/onednn}}"

echo "=== DNN-Opt oneDNN Integration Build ==="
echo "  DNNOPT_ROOT: $DNNOPT_ROOT"
echo "  ONEDNN_DIR:  $ONEDNN_DIR"
echo ""

# Step 1: Build dnnopt if not already built
if [ ! -f "$DNNOPT_ROOT/build/src/libdnnopt_blas.so" ]; then
    echo "[1/3] Building dnnopt..."
    mkdir -p "$DNNOPT_ROOT/build"
    cd "$DNNOPT_ROOT/build"
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j$(nproc)
else
    echo "[1/3] dnnopt already built: $DNNOPT_ROOT/build/src/libdnnopt_blas.so"
fi

# Step 2: Apply dnnopt patches to oneDNN (if not already applied)
if [ ! -f "$ONEDNN_DIR/cmake/FindDNNOPT.cmake" ]; then
    echo "[2/3] Applying dnnopt patches to oneDNN..."
    # Copy FindDNNOPT.cmake
    if [ -f "$DNNOPT_ROOT/patches/FindDNNOPT.cmake" ]; then
        cp "$DNNOPT_ROOT/patches/FindDNNOPT.cmake" "$ONEDNN_DIR/cmake/"
    else
        echo "  WARNING: patches not found, assuming oneDNN already patched"
    fi
else
    echo "[2/3] oneDNN already patched"
fi

# Step 3: Build oneDNN with dnnopt
echo "[3/3] Building oneDNN with DNNL_BLAS_VENDOR=DNNOPT..."
mkdir -p "$ONEDNN_DIR/build"
cd "$ONEDNN_DIR/build"
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_BLAS_VENDOR=DNNOPT \
    -DDNNOPT_ROOT="$DNNOPT_ROOT" \
    -DDNNL_CPU_RUNTIME=OMP \
    -DDNNL_BUILD_EXAMPLES=OFF \
    -DDNNL_BUILD_TESTS=ON
cmake --build . -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "  oneDNN library: $ONEDNN_DIR/build/src/libdnnl.so"
echo ""
echo "To verify:"
echo "  export LD_LIBRARY_PATH=$DNNOPT_ROOT/build/src:\$LD_LIBRARY_PATH"
echo "  DNNL_VERBOSE=1 $ONEDNN_DIR/build/tests/benchdnn/benchdnn --matmul --dt=f32 2m3n4k"
echo "  # Look for 'dnnopt' or 'blas' in verbose output"
