#!/usr/bin/env python3
"""
Full Performance Comparison for TensorFlow

Three configurations:
1. TensorFlow Eigen (TF_DISABLE_ONEDNN=1)
2. TensorFlow oneDNN (default)
3. oneDNN + DNN-Opt (via separate binary)

Shapes tested:
- M=1 (GEMV)
- M=2-7 (small batch)
- Irregular N/K (prime numbers)
- Non-power-of-2 boundaries
- Tiny shapes (N<=32)
- Tall-skinny / Short-wide
- BERT/LLM-like shapes
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def bench_matmul(M, N, K, runs=30, warmup=5):
    A = tf.random.normal([M, K])
    B = tf.random.normal([K, N])
    
    for _ in range(warmup):
        C = tf.matmul(A, B)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = tf.matmul(A, B)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.median(times) * 1000  # ms

def print_section(title):
    print(f"\n=== {title} ===")
    print(f"{'Shape':<15} {'M':<6} {'N':<6} {'K':<6} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 60)

def test_shape(M, N, K, label=None):
    try:
        ms = bench_matmul(M, N, K)
        gflops = 2 * M * N * K / (ms * 1e6)
        shape_str = label if label else f"[{M},{N},{K}]"
        print(f"{shape_str:<15} {M:<6} {N:<6} {K:<6} {ms:>10.4f}  {gflops:>8.2f}")
        return gflops
    except Exception as e:
        print(f"[{M},{N},{K}]".ljust(15), f"ERROR: {e}")
        return 0

def main():
    disable_onednn = os.environ.get('TF_DISABLE_ONEDNN', '0')
    backend = "Eigen" if disable_onednn == '1' else "oneDNN"
    
    print("=" * 70)
    print(f"  TensorFlow {backend}: Full Shape Performance Benchmark")
    print("=" * 70)
    
    # ================================================================
    # M=1: GEMV
    # ================================================================
    print_section("M=1: GEMV (batch-1 inference)")
    
    test_shape(1, 8, 8)
    test_shape(1, 16, 16)
    test_shape(1, 32, 32)
    test_shape(1, 64, 64)
    test_shape(1, 128, 128)
    test_shape(1, 256, 256)
    test_shape(1, 512, 512)
    test_shape(1, 768, 768)
    test_shape(1, 1024, 1024)
    test_shape(1, 2048, 2048)
    test_shape(1, 4096, 4096)
    test_shape(1, 256, 1024)
    test_shape(1, 512, 1024)
    test_shape(1, 1024, 4096)
    
    # ================================================================
    # M=2-7: Small batch
    # ================================================================
    print_section("M=2-7: Small batch inference")
    
    for M in [2, 3, 4, 5, 6, 7]:
        for N in [32, 64, 128, 256]:
            test_shape(M, N, N)
    
    test_shape(4, 512, 512)
    test_shape(4, 768, 768)
    test_shape(4, 1024, 1024)
    test_shape(6, 512, 512)
    test_shape(6, 768, 768)
    
    # ================================================================
    # Irregular N (prime numbers)
    # ================================================================
    print_section("Irregular N (prime numbers)")
    
    primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107]
    for p in primes[:12]:
        K = 64 if p < 53 else 128
        test_shape(4, p, K, f"prime_N{p}")
    
    # ================================================================
    # Irregular K (prime numbers)
    # ================================================================
    print_section("Irregular K (prime numbers)")
    
    for p in [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
        test_shape(4, 64, p, f"prime_K{p}")
    
    # ================================================================
    # Non-power-of-2 boundaries
    # ================================================================
    print_section("Non-power-of-2 boundaries")
    
    test_shape(4, 31, 64)
    test_shape(4, 32, 64)
    test_shape(4, 33, 64)
    test_shape(4, 63, 64)
    test_shape(4, 64, 64)
    test_shape(4, 65, 64)
    test_shape(4, 127, 128)
    test_shape(4, 128, 128)
    test_shape(4, 129, 128)
    
    # ================================================================
    # Tiny shapes
    # ================================================================
    print_section("Tiny shapes (N<=32)")
    
    for M in [1, 4, 8]:
        for N in [4, 8, 12, 16, 24, 32]:
            test_shape(M, N, N)
    
    # ================================================================
    # Tall-skinny
    # ================================================================
    print_section("Tall-skinny (N small, M large)")
    
    for M in [64, 128, 256]:
        for N in [1, 2, 4, 8, 16, 32]:
            K = M
            test_shape(M, N, K)
    
    # ================================================================
    # Short-wide
    # ================================================================
    print_section("Short-wide (M small, N large)")
    
    for M in [2, 4, 6]:
        for N in [128, 256, 512, 1024]:
            test_shape(M, N, 128)
    
    # ================================================================
    # BERT/LLM-like
    # ================================================================
    print_section("BERT/LLM-like shapes")
    
    test_shape(1, 768, 768, "bert_M1")
    test_shape(4, 768, 768, "bert_M4")
    test_shape(4, 3072, 768, "bert_ffn1")
    test_shape(4, 768, 3072, "bert_ffn2")
    
    test_shape(1, 4096, 4096, "llm_M1")
    test_shape(4, 4096, 4096, "llm_M4")
    test_shape(8, 512, 512, "llm_qkv")
    
    # ================================================================
    # Large M (control)
    # ================================================================
    print_section("Large M (oneDNN strength)")
    
    test_shape(16, 256, 256)
    test_shape(32, 256, 256)
    test_shape(32, 512, 512)
    test_shape(64, 512, 512)
    test_shape(128, 256, 256)
    
    print("\n" + "=" * 70)
    print("  End of benchmark")
    print("=" * 70)

if __name__ == '__main__':
    main()
