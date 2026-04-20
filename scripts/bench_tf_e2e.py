#!/usr/bin/env python3
"""
TensorFlow End-to-End Inference Benchmark
Compare: TensorFlow default BLAS vs DNN-Opt BLAS wrapper

Test models:
- BERT-like transformer (batch=1,4,8)
- ResNet-like CNN (batch=1,4,8)
- LLM-like MLP (batch=1,4)

oneDNN/DNN-Opt focuses on small-batch GEMM optimization.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================
# Model definitions
# ============================================================

def create_bert_like_model(seq_len=512, hidden_dim=768, num_layers=4):
    """BERT-like transformer for attention/GEMM benchmark."""
    inputs = tf.keras.Input(shape=(seq_len, hidden_dim))

    x = inputs
    for _ in range(num_layers):
        # Self-attention: QKV projection (small-M GEMM)
        # M=1-8, N=hidden_dim, K=hidden_dim
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=12, key_dim=64
        )
        x = attn(x, x)

        # FFN: two dense layers (oneDNN optimized for M>=64)
        # FFN1: hidden_dim → 3072 (4x expansion)
        # FFN2: 3072 → hidden_dim
        x = tf.keras.layers.Dense(3072, activation='gelu')(x)
        x = tf.keras.layers.Dense(hidden_dim)(x)

    outputs = tf.keras.layers.Dense(hidden_dim)(x)
    return tf.keras.Model(inputs, outputs)


def create_resnet_like_model(input_shape=(224, 224, 3), num_classes=1000):
    """ResNet-like CNN for conv/GEMM benchmark."""
    inputs = tf.keras.Input(shape=input_shape)

    # Conv layers
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # ResNet blocks
    for filters in [64, 128, 256, 512]:
        for _ in range(2):
            x = tf.keras.layers.Conv2D(filters, 3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs, outputs)


def create_llm_mlp_model(input_dim=4096, hidden_dim=4096, num_layers=4):
    """LLM-like MLP for large GEMM benchmark."""
    inputs = tf.keras.Input(shape=(input_dim,))

    x = inputs
    for _ in range(num_layers):
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)

    outputs = tf.keras.layers.Dense(input_dim)(x)
    return tf.keras.Model(inputs, outputs)


# ============================================================
# Benchmark functions
# ============================================================

def benchmark_model(model, input_data, num_runs=50, warmup=5):
    """Benchmark model inference."""
    # Warmup
    for _ in range(warmup):
        _ = model(input_data, training=False)

    # Timed runs
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        end = time.perf_counter()
        times.append(end - start)

    # Statistics
    median_time = np.median(times)
    mean_time = np.mean(times)
    std_time = np.std(times)

    return {
        'median_ms': median_time * 1000,
        'mean_ms': mean_time * 1000,
        'std_ms': std_time * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def get_gemm_flops(M, N, K):
    """Calculate GEMM FLOPs: 2*M*N*K (multiply + add)."""
    return 2 * M * N * K


def estimate_gflops(model_name, batch_size, result):
    """Estimate GFLOPS based on model type and timing."""
    # Approximate FLOPs per inference
    flops_estimates = {
        'BERT': 2e9 * batch_size,  # ~2 GFLOPs per token * seq_len
        'ResNet': 4e9 * batch_size,  # ~4 GFLOPs per image
        'LLM-MLP': 8e9 * batch_size,  # ~8 GFLOPs for MLP layers
    }

    flops = flops_estimates.get(model_name, 1e9)
    gflops = flops / (result['median_ms'] * 1e6)  # FLOPs / time

    return gflops


# ============================================================
# Main benchmark
# ============================================================

def main():
    print("=" * 70)
    print("  TensorFlow End-to-End Inference Benchmark")
    print("  DNN-Opt BLAS Performance Comparison")
    print("=" * 70)
    print()

    # Check if DNN-Opt BLAS is loaded
    blas_lib = os.environ.get('LD_PRELOAD', 'default')
    autotune = os.environ.get('DNNOPT_AUTOTUNE', '0')

    print(f"BLAS library: {blas_lib if blas_lib else 'TensorFlow default'}")
    print(f"DNN-Opt Autotune: {autotune}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()

    # Test configurations (store function references, not model instances)
    configs = [
        ('BERT', create_bert_like_model, [(1, 512, 768), (4, 512, 768), (8, 512, 768)]),
        ('ResNet', create_resnet_like_model, [(1, 224, 224, 3), (4, 224, 224, 3), (8, 224, 224, 3)]),
        ('LLM-MLP', create_llm_mlp_model, [(1, 4096), (4, 4096), (8, 4096)]),
    ]

    print("=" * 70)
    print("  Benchmark Results")
    print("=" * 70)
    print()
    print(f"{'Model':<12} {'Batch':<8} {'Input Shape':<20} {'Median(ms)':<12} {'GFLOPS':<10}")
    print("-" * 70)

    all_results = []

    for model_name, model_fn, shapes in configs:
        print(f"\n{model_name}:")

        for shape in shapes:
            batch_size = shape[0]

            # Create input data
            if model_name == 'BERT':
                input_data = tf.random.normal(shape)
            elif model_name == 'ResNet':
                input_data = tf.random.normal(shape)
            else:  # LLM-MLP
                input_data = tf.random.normal(shape)

            # Build and compile model
            model = model_fn()
            model.compile()

            # Run benchmark
            result = benchmark_model(model, input_data, num_runs=30, warmup=3)
            gflops = estimate_gflops(model_name, batch_size, result)

            shape_str = str(shape)
            print(f"  {model_name:<10} {batch_size:<6} {shape_str:<18} {result['median_ms']:>8.2f}    {gflops:>6.2f}")

            all_results.append({
                'model': model_name,
                'batch': batch_size,
                'shape': shape_str,
                'median_ms': result['median_ms'],
                'gflops': gflops,
            })

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()

    # Group by model
    for model_name in ['BERT', 'ResNet', 'LLM-MLP']:
        model_results = [r for r in all_results if r['model'] == model_name]
        if model_results:
            avg_gflops = np.mean([r['gflops'] for r in model_results])
            batch1_gflops = [r['gflops'] for r in model_results if r['batch'] == 1][0]
            batch4_gflops = [r['gflops'] for r in model_results if r['batch'] == 4][0]

            print(f"{model_name}: Avg={avg_gflops:.2f} GFLOPS")
            print(f"  batch-1: {batch1_gflops:.2f} GFLOPS (oneDNN weakness)")
            print(f"  batch-4: {batch4_gflops:.2f} GFLOPS")

    print()
    print("Note: batch-1 shapes are oneDNN's weakness domain.")
    print("DNN-Opt specializes in optimizing small-M GEMM (M<8).")
    print()


if __name__ == '__main__':
    main()