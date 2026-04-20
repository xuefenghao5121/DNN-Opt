#!/usr/bin/env python3
"""
TensorFlow Full Coverage Benchmark: Small-M + Irregular N
Matches bench_full_coverage.cpp for direct comparison
"""

import os
import sys
import time
import math
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def is_prime(n):
    if n < 2: return False
    if n == 2: return True
    if n % 2 == 0: return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0: return False
    return True

def is_power_of_two(n):
    return n > 0 and (n & (n-1)) == 0

def bench_matmul(M, N, K, runs=20, warmup=3):
    try:
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
    except Exception as e:
        return None

def main():
    disable_onednn = os.environ.get('TF_DISABLE_ONEDNN', '0')
    backend = "Eigen" if disable_onednn == '1' else "oneDNN"
    
    print("=" * 80)
    print(f"  TensorFlow {backend}: Full Coverage Benchmark")
    print("=" * 80)
    print()
    
    # Part 1: M=1 GEMV with various N
    print("=== Part 1: M=1 (GEMV) Full Coverage ===\n")
    
    n_values = []
    # Prime numbers 11-127
    for n in range(11, 128):
        if is_prime(n): n_values.append(n)
    # Non-power-of-2
    for n in range(6, 251):
        if not is_power_of_two(n) and not is_prime(n):
            if n <= 127 or n % 16 == 0 or n % 32 == 1:
                n_values.append(n)
    # Power of 2
    n = 8
    while n <= 256:
        n_values.append(n)
        n *= 2
    
    n_values = sorted(set(n_values))
    k_values = [64, 128]
    
    print(f"Testing {len(n_values)} N values x {len(k_values)} K values\n")
    
    print(f"{'N':<8} {'K':<8} {'Type':<8} {'Time(ms)':<12} {'GFLOPS':<10} {'Prime?':<8}")
    print("-" * 60)
    
    results = {}
    for K in k_values:
        for N in n_values:
            if N > 4 * K: continue
            
            ms = bench_matmul(1, N, K)
            if ms is None: continue
            
            gflops = 2 * 1 * N * K / (ms * 1e6)
            type_str = "pow2" if is_power_of_two(N) else "prime" if is_prime(N) else "irreg"
            prime_mark = "YES" if is_prime(N) else ""
            
            print(f"{N:<8} {K:<8} {type_str:<8} {ms:>10.4f}  {gflops:>8.2f}  {prime_mark:<8}")
            results[(1, N, K)] = {'ms': ms, 'gflops': gflops, 'type': type_str}
        print()
    
    # Part 2: M=2-7
    print("\n=== Part 2: M=2-7 Full Coverage ===\n")
    
    for M in range(2, 8):
        print(f"M={M} Results:")
        print(f"{'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
        print("-" * 40)
        
        for N in range(8, 256, 8):
            for K in [64, 128]:
                if N > 4 * K: continue
                
                ms = bench_matmul(M, N, K)
                if ms is None: continue
                
                gflops = 2 * M * N * K / (ms * 1e6)
                print(f"{N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}")
                results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
        print()
        
        # Prime N
        print(f"Prime N for M={M}:")
        print(f"{'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
        print("-" * 40)
        
        primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
        for N in primes:
            for K in [64, 128]:
                ms = bench_matmul(M, N, K)
                if ms is None: continue
                
                gflops = 2 * M * N * K / (ms * 1e6)
                print(f"{N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}")
                results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
        print()
    
    # Part 3: M=8 boundary
    print("\n=== Part 3: M=8 (Boundary) ===\n")
    print(f"{'N':<8} {'K':<8} {'Type':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 50)
    
    for N in [8, 12, 16, 17, 24, 32, 37, 48, 64, 73, 128]:
        for K in [64, 128]:
            ms = bench_matmul(8, N, K)
            if ms is None: continue
            
            gflops = 2 * 8 * N * K / (ms * 1e6)
            type_str = "prime" if is_prime(N) else "pow2" if is_power_of_two(N) else "irreg"
            print(f"{N:<8} {K:<8} {type_str:<8} {ms:>10.4f}  {gflops:>8.2f}")
            results[(8, N, K)] = {'ms': ms, 'gflops': gflops}
    
    # Part 4: Prime K
    print("\n=== Part 4: Irregular K (Prime K) ===\n")
    print(f"{'M':<8} {'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 50)
    
    prime_k = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    for M in [4, 6, 8]:
        for N in [32, 64, 128]:
            for K in prime_k:
                if K > N * 2: continue
                
                ms = bench_matmul(M, N, K)
                if ms is None: continue
                
                gflops = 2 * M * N * K / (ms * 1e6)
                print(f"{M:<8} {N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}")
                results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
    
    # Part 5: Tiny shapes
    print("\n=== Part 5: Tiny Shapes (N<=32) ===\n")
    print(f"{'M':<8} {'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 50)
    
    for M in range(1, 9):
        for N in range(4, 36, 4):
            for K in range(N, min(65, N*2+1), N):
                ms = bench_matmul(M, N, K)
                if ms is None: continue
                
                gflops = 2 * M * N * K / (ms * 1e6)
                print(f"{M:<8} {N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}")
                results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
    
    # Part 6: Tall-skinny
    print("\n=== Part 6: Tall-Skinny ===\n")
    print(f"{'M':<8} {'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10}")
    print("-" * 50)
    
    for M in [32, 64, 128, 256]:
        for N in [1, 2, 4, 8, 12, 16, 24, 32]:
            K = M
            ms = bench_matmul(M, N, K)
            if ms is None: continue
            
            gflops = 2 * M * N * K / (ms * 1e6)
            print(f"{M:<8} {N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}")
            results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
    
    # Part 7: Power-of-2 boundaries
    print("\n=== Part 7: Power-of-2 Boundaries ===\n")
    print(f"{'M':<8} {'N':<8} {'K':<8} {'Time(ms)':<12} {'GFLOPS':<10} {'N_type':<10}")
    print("-" * 60)
    
    for M in [4, 6, 8]:
        for base in [32, 64, 128]:
            for offset in [-1, 0, 1]:
                N = base + offset
                K = 128
                ms = bench_matmul(M, N, K)
                if ms is None: continue
                
                gflops = 2 * M * N * K / (ms * 1e6)
                n_type = "pow2" if offset == 0 else ("pow2-1" if offset == -1 else "pow2+1")
                print(f"{M:<8} {N:<8} {K:<8} {ms:>10.4f}  {gflops:>8.2f}  {n_type:<10}")
                results[(M, N, K)] = {'ms': ms, 'gflops': gflops}
    
    print("\n" + "=" * 80)
    print("  Benchmark Complete")
    print("=" * 80)

if __name__ == '__main__':
    main()
