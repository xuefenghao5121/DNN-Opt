#!/usr/bin/env python3
"""
MLP benchmark with small hidden_dim (DNN-Opt target domain)
Tests N<=128 to trigger DNN-Opt optimization
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def create_mlp_small(input_dim, hidden_dim, num_layers=4):
    """MLP with configurable dimensions."""
    inputs = tf.keras.Input(shape=(input_dim,))
    
    x = inputs
    for i in range(num_layers):
        x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    
    outputs = tf.keras.layers.Dense(input_dim)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile()
    return model

def benchmark(model, input_data, runs=50, warmup=10):
    """Benchmark model inference."""
    for _ in range(warmup):
        _ = model(input_data, training=False)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.median(times), np.mean(times)

def main():
    backend = 'Eigen' if os.environ.get('TF_DISABLE_ONEDNN') == '1' else 'oneDNN'
    
    print("=" * 70)
    print(f"  MLP Benchmark with Small Hidden_dim (DNN-Opt Target)")
    print(f"  Backend: {backend}")
    print("=" * 70)
    print()
    
    # Test configurations: small N to trigger DNN-Opt optimization
    configs = [
        # (batch, input_dim, hidden_dim, num_layers) - GEMM shape: [batch, hidden_dim, input_dim]
        (1, 32, 32, 4),    # [1,32,32] - tiny GEMV
        (1, 64, 64, 4),    # [1,64,64] - small GEMV
        (1, 128, 128, 4),  # [1,128,128] - small GEMV
        (1, 64, 128, 4),   # [1,64,128] - embedding-like
        (4, 32, 32, 4),    # [4,32,32]
        (4, 64, 64, 4),    # [4,64,64]
        (4, 128, 128, 4),  # [4,128,128]
        (4, 64, 128, 4),   # [4,64,128]
        (8, 32, 32, 4),    # [8,32,32]
        (8, 64, 64, 4),    # [8,64,64]
        (8, 128, 128, 4),  # [8,128,128]
        
        # Prime N values (irregular)
        (1, 13, 64, 4),    # prime N=13
        (1, 17, 64, 4),    # prime N=17
        (1, 37, 64, 4),    # prime N=37
        (4, 13, 64, 4),
        (4, 17, 64, 4),
        (4, 37, 64, 4),
        
        # Larger for comparison
        (1, 256, 256, 4),  # [1,256,256] - medium
        (1, 512, 512, 4),  # [1,512,512] - medium
        (4, 256, 256, 4),  # [4,256,256]
        
        # Very large (DNN-Opt weak)
        (1, 4096, 4096, 4), # [1,4096,4096] - large N
    ]
    
    print(f"{'Config':<20} {'Layers':<8} {'GEMM Shape':<15} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 70)
    
    results = []
    for batch, input_dim, hidden_dim, num_layers in configs:
        # Create model
        model = create_mlp_small(input_dim, hidden_dim, num_layers)
        input_data = tf.random.normal([batch, input_dim])
        
        # Benchmark
        median_ms, mean_ms = benchmark(model, input_data, runs=30, warmup=5)
        
        # Calculate GFLOPS
        # Each Dense layer: 2 * batch * hidden_dim * input_dim FLOPs
        flops_per_layer = 2 * batch * hidden_dim * input_dim
        total_flops = flops_per_layer * num_layers * 2  # forward + internal
        gflops = total_flops / (mean_ms * 1e6)
        
        config_str = f"[{batch},{input_dim},{hidden_dim}]"
        shape_str = f"[{batch},{hidden_dim},{input_dim}]"
        print(f"{config_str:<20} {num_layers:<8} {shape_str:<15} {mean_ms:<12.3f} {gflops:<10.2f}")
        
        results.append({
            'config': config_str,
            'batch': batch,
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'time_ms': mean_ms,
            'gflops': gflops,
        })
    
    print()
    print("=" * 70)
    print("  Analysis")
    print("=" * 70)
    print()
    
    # Group by N size
    small_n = [r for r in results if r['hidden_dim'] <= 128]
    medium_n = [r for r in results if 128 < r['hidden_dim'] <= 512]
    large_n = [r for r in results if r['hidden_dim'] > 512]
    
    if small_n:
        avg_gf = np.mean([r['gflops'] for r in small_n])
        print(f"Small N (<=128): Avg {avg_gf:.2f} GFLOPS")
        print("  DNN-Opt target domain: expected +10-50x improvement")
    
    if medium_n:
        avg_gf = np.mean([r['gflops'] for r in medium_n])
        print(f"Medium N (128-512): Avg {avg_gf:.2f} GFLOPS")
    
    if large_n:
        avg_gf = np.mean([r['gflops'] for r in large_n])
        print(f"Large N (>512): Avg {avg_gf:.2f} GFLOPS")
        print("  DNN-Opt weak domain: may be slower than baseline")
    
    print()
    print("Note: TensorFlow framework overhead still present.")
    print("Expected DNN-Opt standalone improvement:")
    print("  - N<=128, M<=8: +10-50x vs TF baseline")
    print("  - Need patched TensorFlow build for E2E test")

if __name__ == '__main__':
    main()
