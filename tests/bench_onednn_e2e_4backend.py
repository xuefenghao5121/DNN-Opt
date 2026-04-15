#!/usr/bin/env python3
"""
Comprehensive 4-Backend End-to-End Benchmark:

  1. oneDNN upstream   — dnnl_sgemm via ctypes (upstream, no dnnopt)
  2. oneDNN + dnnopt   — dnnl_sgemm via ctypes (with dnnopt dispatch)
  3. NumPy / OpenBLAS  — np.matmul (multi-threaded)
  4. TensorFlow        — tf.matmul (Eigen/XNNPACK backend)

Models: CVR, BERT-small, LLM (batch=1 and batch=4).
"""

import argparse
import ctypes
import os
import sys
import time

import numpy as np


# ============================================================
# Backend: oneDNN sgemm via ctypes
# ============================================================

class BackendOneDNN:
    """Call dnnl_sgemm via ctypes — same API TensorFlow uses."""

    def __init__(self, lib_path, label):
        self.lib = ctypes.CDLL(lib_path)
        self.name = label

        # dnnl_sgemm(transa, transb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
        self.lib.dnnl_sgemm.argtypes = [
            ctypes.c_char,        # transa
            ctypes.c_char,        # transb
            ctypes.c_int,         # M
            ctypes.c_int,         # N
            ctypes.c_int,         # K
            ctypes.c_float,       # alpha
            ctypes.POINTER(ctypes.c_float),  # A
            ctypes.c_int,         # lda
            ctypes.POINTER(ctypes.c_float),  # B
            ctypes.c_int,         # ldb
            ctypes.c_float,       # beta
            ctypes.POINTER(ctypes.c_float),  # C
            ctypes.c_int,         # ldc
        ]
        self.lib.dnnl_sgemm.restype = ctypes.c_int

    def sgemm(self, M, N, K, A, B):
        """C = A @ B, where A is (M,K) and B is (K,N), both row-major.

        dnnl_sgemm uses row-major convention internally but calls col-major
        extended_sgemm with swapped parameters.

        For row-major: C(M,N) = A(M,K) @ B(K,N)
        dnnl_sgemm('T', 'T', M, N, K, A, K, B, N, C, N)
        - transa='T': A is row-major (K columns), needs transpose to MxK
        - transb='T': B is row-major (N columns), needs transpose to KxN
        - lda=K, ldb=N, ldc=N (row-major strides)
        """
        C = np.zeros((M, N), dtype=np.float32)

        self.lib.dnnl_sgemm(
            b'T', b'T',
            M, N, K,
            1.0,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            0.0,
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
        )
        return C

    def bench_layer(self, M, N, K, warmup=5, iters=30):
        np.random.seed(42)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32))
        C = np.zeros((M, N), dtype=np.float32)

        # For row-major matrices, stride is the number of columns
        # A(M,K): lda = K
        # B(K,N): ldb = N
        # C(M,N): ldc = N
        transa = b'T'  # A stored row-major, need transpose for col-major
        transb = b'T'  # B stored row-major, need transpose for col-major

        # Warmup
        for _ in range(warmup):
            self.lib.dnnl_sgemm(
                transa, transb, M, N, K, 1.0,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
                0.0,
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            )

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.lib.dnnl_sgemm(
                transa, transb, M, N, K, 1.0,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
                0.0,
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        times.sort()
        return times[iters // 2]


# ============================================================
# Backend: dnnopt cblas_sgemm via ctypes
# ============================================================

class BackendDnnopt:
    """Call dnnopt's cblas_sgemm directly via ctypes."""

    CblasRowMajor = 101
    CblasNoTrans = 111

    def __init__(self):
        lib_path = '/root/onednn-arm-opt/build/src/libdnnopt_blas.so'
        self.lib = ctypes.CDLL(lib_path)
        self.lib.cblas_sgemm.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ]
        self.lib.cblas_sgemm.restype = None
        self.name = 'dnnopt'

    def sgemm(self, M, N, K, A, B):
        C = np.zeros((M, N), dtype=np.float32)
        self.lib.cblas_sgemm(
            self.CblasRowMajor, self.CblasNoTrans, self.CblasNoTrans,
            M, N, K, 1.0,
            A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
            B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            0.0,
            C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
        )
        return C

    def bench_layer(self, M, N, K, warmup=5, iters=30):
        np.random.seed(42)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32))
        C = np.zeros((M, N), dtype=np.float32)

        for _ in range(warmup):
            self.lib.cblas_sgemm(
                self.CblasRowMajor, self.CblasNoTrans, self.CblasNoTrans,
                M, N, K, 1.0,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
                0.0,
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            )

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            self.lib.cblas_sgemm(
                self.CblasRowMajor, self.CblasNoTrans, self.CblasNoTrans,
                M, N, K, 1.0,
                A.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), K,
                B.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
                0.0,
                C.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), N,
            )
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        times.sort()
        return times[iters // 2]


# ============================================================
# Backend: NumPy / OpenBLAS
# ============================================================

class BackendNumpy:
    def __init__(self):
        self.name = 'NumPy'

    def sgemm(self, M, N, K, A, B):
        return np.matmul(A, B)

    def bench_layer(self, M, N, K, warmup=5, iters=30):
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        for _ in range(warmup):
            np.matmul(A, B)

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            np.matmul(A, B)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        times.sort()
        return times[iters // 2]


# ============================================================
# Backend: TensorFlow
# ============================================================

class BackendTF:
    def __init__(self):
        import tensorflow as tf
        self.tf = tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        self.name = 'TF'

    def sgemm(self, M, N, K, A_np, B_np):
        A = self.tf.constant(A_np)
        B = self.tf.constant(B_np)
        return self.tf.matmul(A, B).numpy()

    def bench_layer(self, M, N, K, warmup=5, iters=30):
        np.random.seed(42)
        A = self.tf.constant(np.random.randn(M, K).astype(np.float32))
        B = self.tf.constant(np.random.randn(K, N).astype(np.float32))

        for _ in range(warmup):
            _ = self.tf.matmul(A, B).numpy()

        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = self.tf.matmul(A, B).numpy()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1e6)

        times.sort()
        return times[iters // 2]


# ============================================================
# Model layer definitions
# ============================================================

def get_cvr_layers(batch):
    return [
        ('embedding',   batch, 256, 1024),
        ('fc1',         batch, 128, 256),
        ('fc2',         batch, 64,  128),
        ('classifier',  batch, 10,  64),
    ]

def get_bert_layers(batch):
    return [
        ('qkv_proj',  batch, 768,  768),
        ('out_proj',  batch, 768,  768),
        ('ffn1',      batch, 3072, 768),
        ('ffn2',      batch, 768,  3072),
    ]

def get_llm_layers(batch):
    return [
        ('q_proj',     batch, 4096,  4096),
        ('k_proj',     batch, 4096,  4096),
        ('v_proj',     batch, 4096,  4096),
        ('o_proj',     batch, 4096,  4096),
        ('gate_proj',  batch, 11008, 4096),
        ('up_proj',    batch, 4096,  4096),
        ('down_proj',  batch, 4096,  11008),
    ]

MODELS = [
    ('CVR b=1',  lambda: get_cvr_layers(1)),
    ('CVR b=4',  lambda: get_cvr_layers(4)),
    ('BERT b=1', lambda: get_bert_layers(1)),
    ('BERT b=4', lambda: get_bert_layers(4)),
    ('LLM b=1',  lambda: get_llm_layers(1)),
    ('LLM b=4',  lambda: get_llm_layers(4)),
]


# ============================================================
# Correctness verification
# ============================================================

def verify_correctness(backends, tol_factor=2e-5):
    test_shapes = [
        (1, 256, 1024),
        (4, 4096, 4096),
        (1, 11008, 4096),
        (4, 64, 128),
    ]
    errors = []
    for M, N, K in test_shapes:
        np.random.seed(12345)
        A = np.ascontiguousarray(np.random.randn(M, K).astype(np.float32))
        B = np.ascontiguousarray(np.random.randn(K, N).astype(np.float32))
        C_ref = np.matmul(A, B)
        for be in backends:
            C_be = be.sgemm(M, N, K, A, B)
            if C_be is None:
                continue
            err = np.max(np.abs(C_ref - C_be))
            tol = K * tol_factor
            if err > tol:
                errors.append(f"  {be.name} M={M} N={N} K={K}: max_err={err:.2e} > tol={tol:.2e}")
    return errors


# ============================================================
# Run & print
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='4-backend end-to-end benchmark')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--no-tf', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--iters', type=int, default=30)
    args = parser.parse_args()
    if args.quick:
        args.warmup, args.iters = 2, 10

    print("=" * 100)
    print("  4-Backend End-to-End Benchmark: oneDNN vs oneDNN+dnnopt vs NumPy vs TF")
    print("=" * 100)

    # Setup backends
    onednn_upstream = BackendOneDNN(
        '/root/onednn-upstream/build/src/libdnnl.so.3', 'oneDNN')
    onednn_dnnopt = BackendOneDNN(
        '/root/onednn-dnnopt/build/src/libdnnl.so.3', 'oneDNN+dnnopt')
    backends = [onednn_upstream, onednn_dnnopt, BackendDnnopt(), BackendNumpy()]

    tf_backend = None
    if not args.no_tf:
        try:
            tf_backend = BackendTF()
            backends.append(tf_backend)
            import tensorflow as tf
            print(f"TensorFlow {tf.__version__} (Eigen/XNNPACK, 2 threads)")
        except ImportError:
            print("TensorFlow not available, skipping TF backend")

    print(f"oneDNN upstream:  /root/onednn-upstream/build/src/libdnnl.so.3")
    print(f"oneDNN+dnnopt:    /root/onednn-dnnopt/build/src/libdnnl.so.3  (63 dnnopt symbols)")
    print(f"dnnopt standalone: /root/onednn-arm-opt/build/src/libdnnopt_blas.so")
    print(f"NumPy OpenBLAS:   multi-threaded")
    print(f"Warmup: {args.warmup}, Iters: {args.iters}")
    print()

    # Correctness
    print("--- Correctness Verification ---")
    errors = verify_correctness(backends)
    if errors:
        for e in errors:
            print(f"  FAIL: {e}")
    else:
        print(f"  All {len(backends)} backends match (tolerance: K * 2e-5)")

    # Benchmarks
    be_names = [be.name for be in backends]

    for model_name, get_layers in MODELS:
        layers = get_layers()
        n_be = len(backends)

        print(f"\n--- {model_name} ---")
        # Header
        print(f"  {'Layer':<14s} {'Shape':<18s}", end='')
        for name in be_names:
            print(f" {name + '(us)':>14s}", end='')
        for name in be_names:
            print(f" {name + '(GF)':>12s}", end='')
        # Ratio columns: dnnopt speedup over each backend
        print(f" {'DNN/Up':>8s} {'DNN/NP':>8s}", end='')
        if tf_backend:
            print(f" {'DNN/TF':>8s}", end='')
        print()
        print(f"  {'-' * (14 + 18 + 14*n_be + 12*n_be + 9 + (9 if tf_backend else 0))}")

        total_us = {name: 0.0 for name in be_names}
        total_gf = {name: 0.0 for name in be_names}
        total_flops = 0

        for name_l, M, N, K in layers:
            flops = 2.0 * M * N * K
            total_flops += flops
            shape_str = f"[{M},{N},{K}]"
            print(f"  {name_l:<14s} {shape_str:<18s}", end='')

            row_us = {}
            row_gf = {}
            for be in backends:
                us = be.bench_layer(M, N, K, args.warmup, args.iters)
                gf = flops / (us * 1e3)
                row_us[be.name] = us
                row_gf[be.name] = gf
                total_us[be.name] += us
                print(f" {us:>13.0f}", end='')

            for be in backends:
                print(f" {row_gf[be.name]:>11.2f}", end='')

            # Ratio: oneDNN+dnnopt vs others
            dn_us = row_us['oneDNN+dnnopt']
            up_us = row_us['oneDNN']
            np_us = row_us['NumPy']
            ratio_up = up_us / dn_us if dn_us > 0 else 0
            ratio_np = np_us / dn_us if dn_us > 0 else 0
            print(f" {ratio_up:>7.2f}x {ratio_np:>7.2f}x", end='')
            if tf_backend:
                tf_us = row_us['TF']
                ratio_tf = tf_us / dn_us if dn_us > 0 else 0
                print(f" {ratio_tf:>7.2f}x", end='')
            print()

        # Total row
        print(f"  {'TOTAL':<14s} {'':18s}", end='')
        for name in be_names:
            print(f" {total_us[name]:>13.0f}", end='')
        for name in be_names:
            avg_gf = total_flops / (total_us[name] * 1e3) if total_us[name] > 0 else 0
            print(f" {avg_gf:>11.2f}", end='')
        dn_total = total_us['oneDNN+dnnopt']
        up_total = total_us['oneDNN']
        np_total = total_us['NumPy']
        print(f" {up_total/dn_total:>7.2f}x {np_total/dn_total:>7.2f}x", end='')
        if tf_backend:
            tf_total = total_us['TF']
            print(f" {tf_total/dn_total:>7.2f}x", end='')
        print()

    # Summary
    print(f"\n{'=' * 100}")
    print(f"  Summary")
    print(f"{'=' * 100}")
    print(f"  Column 'DNN/Up'  = oneDNN+dnnopt speedup over oneDNN upstream")
    print(f"  Column 'DNN/NP'  = oneDNN+dnnopt speedup over NumPy/OpenBLAS")
    if tf_backend:
        print(f"  Column 'DNN/TF'  = oneDNN+dnnopt speedup over TensorFlow/Eigen")
    print()
    print(f"  oneDNN+dnnopt uses OpenMP (2 threads) — same as upstream oneDNN")
    print(f"  dnnopt standalone is single-threaded (for reference only)")
    print(f"{'=' * 100}")


if __name__ == '__main__':
    main()
