#!/bin/bash
# oneDNN ARM 性能基准测试脚本
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
BENCHDNN="${BUILD_DIR}/tests/benchdnn/benchdnn"
RESULT_DIR="${PROJECT_DIR}/benchmarks/results/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULT_DIR}"

# ============================================================
# 环境信息收集
# ============================================================
echo "Collecting system info..."
{
    echo "=== Date ==="
    date
    echo ""
    echo "=== CPU Info ==="
    lscpu 2>/dev/null || cat /proc/cpuinfo
    echo ""
    echo "=== Memory ==="
    free -h
    echo ""
    echo "=== Kernel ==="
    uname -a
    echo ""
    echo "=== Compiler ==="
    g++ --version 2>/dev/null || true
    echo ""
    echo "=== SVE Vector Length ==="
    # 如果有 SVE 支持
    if grep -q 'sve' /proc/cpuinfo 2>/dev/null; then
        echo "SVE supported"
    else
        echo "SVE not detected in cpuinfo"
    fi
} > "${RESULT_DIR}/system_info.txt"

# ============================================================
# 设置运行时环境
# ============================================================
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
export GOMP_CPU_AFFINITY="0-$((OMP_NUM_THREADS-1))"
export DNNL_VERBOSE=0

echo "============================================"
echo " oneDNN ARM Benchmark Suite"
echo " Threads: ${OMP_NUM_THREADS}"
echo " Results: ${RESULT_DIR}"
echo "============================================"

# ============================================================
# Benchmark 1: Convolution (ResNet-50 shapes)
# ============================================================
echo ""
echo "[1/6] Convolution - ResNet-50 shapes..."

# FP32
${BENCHDNN} --conv --mode=p \
    --dt=f32 --dir=FWD_I --tag=nhwc:nhwc \
    mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3 \
    mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1 \
    mb1_ic64oc128_ih56oh28kh3sh2dh0ph1_iw56ow28kw3sw2dw0pw1 \
    mb1_ic128oc128_ih28oh28kh3sh1dh0ph1_iw28ow28kw3sw1dw0pw1 \
    mb1_ic128oc256_ih28oh14kh3sh2dh0ph1_iw28ow14kw3sw2dw0pw1 \
    mb1_ic256oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1 \
    mb1_ic256oc512_ih14oh7kh3sh2dh0ph1_iw14ow7kw3sw2dw0pw1 \
    mb1_ic512oc512_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1 \
    2>&1 | tee "${RESULT_DIR}/conv_resnet50_fp32.txt"

# INT8 (如果支持)
${BENCHDNN} --conv --mode=p \
    --dt=u8:s8:u8 --dir=FWD_I --tag=nhwc:nhwc \
    mb1_ic3oc64_ih224oh112kh7sh2dh0ph3_iw224ow112kw7sw2dw0pw3 \
    mb1_ic64oc64_ih56oh56kh3sh1dh0ph1_iw56ow56kw3sw1dw0pw1 \
    mb1_ic128oc128_ih28oh28kh3sh1dh0ph1_iw28ow28kw3sw1dw0pw1 \
    mb1_ic256oc256_ih14oh14kh3sh1dh0ph1_iw14ow14kw3sw1dw0pw1 \
    mb1_ic512oc512_ih7oh7kh3sh1dh0ph1_iw7ow7kw3sw1dw0pw1 \
    2>&1 | tee "${RESULT_DIR}/conv_resnet50_int8.txt"

# ============================================================
# Benchmark 2: MatMul (Transformer shapes)
# ============================================================
echo ""
echo "[2/6] MatMul - Transformer shapes..."

# BERT-base shapes (batch=1, seq_len=128, hidden=768, heads=12)
${BENCHDNN} --matmul --mode=p \
    --dt=f32 \
    128x768:768x768 \
    128x768:768x3072 \
    128x3072:3072x768 \
    12x128x64:12x64x128 \
    12x128x128:12x128x64 \
    2>&1 | tee "${RESULT_DIR}/matmul_bert_fp32.txt"

# BF16 MatMul (如果支持)
${BENCHDNN} --matmul --mode=p \
    --dt=bf16:bf16:f32 \
    128x768:768x768 \
    128x768:768x3072 \
    128x3072:3072x768 \
    2>&1 | tee "${RESULT_DIR}/matmul_bert_bf16.txt" || echo "BF16 not supported, skipping"

# ============================================================
# Benchmark 3: Inner Product (FC layers)
# ============================================================
echo ""
echo "[3/6] Inner Product..."

${BENCHDNN} --ip --mode=p \
    --dt=f32 --dir=FWD_I \
    mb1ic768oc768 \
    mb1ic768oc3072 \
    mb1ic3072oc768 \
    mb1ic1024oc1024 \
    mb1ic1024oc4096 \
    2>&1 | tee "${RESULT_DIR}/ip_fp32.txt"

# ============================================================
# Benchmark 4: Pooling
# ============================================================
echo ""
echo "[4/6] Pooling..."

${BENCHDNN} --pool --mode=p \
    --dt=f32 --dir=FWD_I --tag=nhwc \
    mb1ic64_ih112oh56kh3sh2ph0 \
    mb1ic256_ih56oh28kh3sh2ph0 \
    mb1ic512_ih28oh14kh3sh2ph0 \
    mb1ic2048_ih7oh1kh7sh1ph0 \
    2>&1 | tee "${RESULT_DIR}/pool_fp32.txt"

# ============================================================
# Benchmark 5: Elementwise / Activation
# ============================================================
echo ""
echo "[5/6] Elementwise (ReLU, GELU, Sigmoid)..."

${BENCHDNN} --eltwise --mode=p \
    --dt=f32 --dir=FWD_D --tag=nhwc \
    --alg=relu 1x64x112x112 1x256x56x56 1x512x28x28 1x2048x7x7 \
    2>&1 | tee "${RESULT_DIR}/eltwise_relu_fp32.txt"

${BENCHDNN} --eltwise --mode=p \
    --dt=f32 --dir=FWD_D \
    --alg=gelu_tanh 1x768x128 1x3072x128 \
    2>&1 | tee "${RESULT_DIR}/eltwise_gelu_fp32.txt"

# ============================================================
# Benchmark 6: Layer Normalization
# ============================================================
echo ""
echo "[6/6] Layer Normalization..."

${BENCHDNN} --lnorm --mode=p \
    --dt=f32 --dir=FWD_I --tag=ab \
    128x768 128x1024 128x4096 \
    2>&1 | tee "${RESULT_DIR}/lnorm_fp32.txt"

# ============================================================
# 汇总结果
# ============================================================
echo ""
echo "============================================"
echo " Benchmark Complete!"
echo " Results saved to: ${RESULT_DIR}"
echo "============================================"
echo ""
echo "Result files:"
ls -la "${RESULT_DIR}/"
