#!/usr/bin/env python3
"""TensorFlow matmul benchmark - oneDNN weakness domain (M=1-7)"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
    
    return np.median(times) * 1000

# Same shapes as bench_dnnopt_weakness.cpp
shapes = [
    (1, 128, 128), (1, 256, 512), (1, 1024, 1024), (1, 4096, 4096), (1, 1024, 4096),
    (2, 64, 64), (2, 128, 128),
    (3, 64, 64), (3, 128, 128),
    (4, 64, 64), (4, 128, 128),
    (5, 64, 64), (5, 128, 128),
    (6, 64, 64), (6, 128, 128),
    (7, 64, 64), (7, 128, 128),
    (8, 17, 64), (8, 37, 64), (8, 53, 128), (8, 97, 128),
    (32, 512, 512), (64, 512, 512),
]

disable_onednn = os.environ.get('TF_DISABLE_ONEDNN', '0')
backend = "Eigen" if disable_onednn == '1' else "oneDNN"

print("=" * 70)
print(f"  TensorFlow {backend} - Weakness Domain Benchmark")
print("=" * 70)
print()
print(f"{'Shape':<15} {'M':<5} {'N':<5} {'K':<5} {'Time(ms)':<10} {'GFLOPS':<8}")
print("-" * 70)

small_m_total = 0
large_m_total = 0
small_m_count = 0
large_m_count = 0

for M, N, K in shapes:
    try:
        ms = bench_matmul(M, N, K)
        gflops = 2 * M * N * K / (ms * 1e6)
        print(f"[{M},{N},{K}]".ljust(15), f"{M:<5} {N:<5} {K:<5} {ms:>8.4f}  {gflops:>6.2f}")
        
        if M < 8:
            small_m_total += gflops
            small_m_count += 1
        else:
            large_m_total += gflops
            large_m_count += 1
    except Exception as e:
        print(f"[{M},{N},{K}]".ljust(15), f"ERROR: {e}")

print()
print("=" * 70)
print(f"Small-M (M<8): Avg {small_m_total/small_m_count:.2f} GFLOPS ({small_m_count} shapes)")
print(f"Large-M (M>=8): Avg {large_m_total/large_m_count:.2f} GFLOPS ({large_m_count} shapes)")
