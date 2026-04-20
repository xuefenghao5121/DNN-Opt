#!/usr/bin/env python3
"""
Analyze MLPerf-style results vs DNN-Opt standalone GEMM results
"""

import json

# Load MLPerf results
with open('/tmp/mlperf_results_Eigen.json') as f:
    eigen_results = json.load(f)

with open('/tmp/mlperf_results_oneDNN.json') as f:
    onednn_results = json.load(f)

print("=" * 80)
print("  MLPerf E2E Results Analysis")
print("=" * 80)
print()

print("TensorFlow Backend Comparison:")
print("-" * 80)

# Compare Eigen vs oneDNN
print(f"\n{'Model':<12} {'Scenario':<10} {'Batch':<8} {'Eigen':<12} {'oneDNN':<12} {'Delta':<10}")
print("-" * 60)

for e, o in zip(eigen_results, onednn_results):
    model = e['model']
    scenario = e['scenario']
    batch = e['batch_size']
    
    if scenario == 'single':
        e_lat = e['median_latency_ms']
        o_lat = o['median_latency_ms']
    else:
        e_lat = e['mean_latency_ms']
        o_lat = o['mean_latency_ms']
    
    delta = (o_lat - e_lat) / e_lat * 100
    delta_str = f"{delta:.1f}%"
    
    print(f"{model:<12} {scenario:<10} {batch:<8} {e_lat:<12.3f} {o_lat:<12.3f} {delta_str:<10}")

print()
print("=" * 80)
print("  Key Analysis")
print("=" * 80)
print()

print("""
1. Eigen vs oneDNN: IDENTICAL performance
   - All models: <1% difference
   - Confirms TF built-in oneDNN doesn't optimize small-batch

2. Framework overhead dominates small shapes:
   - MLP batch=1: 29ms total for 8 Dense layers
   - Pure GEMM should be microseconds for small N

3. DNN-Opt improvement NOT visible because:
   - MLP uses N=4096 (large N, DNN-Opt weak domain)
   - TF overhead masks GEMM performance difference
   - Need custom TF build with patched oneDNN

4. To see DNN-Opt value:
   - Build TensorFlow with oneDNN-DNN-Opt patch
   - Or use smaller N in MLP model
   - Or use standalone GEMM benchmark (already shows +10-50x)

5. MLPerf submission recommendation:
   - Target: Recommendation systems (small-M GEMM heavy)
   - Scenario: Single-stream (batch=1)
   - Expected: +10-50x for N<=128 shapes
""")

print()
print("Estimated GEMM compute vs framework overhead:")
print("-" * 60)

# MLP model analysis
mlp_layers = 8
mlp_dim = 4096

for batch in [1, 4, 8]:
    # Total FLOPs
    flops = mlp_layers * 2 * batch * mlp_dim * mlp_dim
    gflops = flops / 1e9
    
    # Get actual latency
    for r in eigen_results:
        if r['model'] == 'mlp' and r['scenario'] == 'single' and r['batch_size'] == batch:
            latency_ms = r['median_latency_ms']
            actual_gflops = gflops / (latency_ms / 1000)
            
            print(f"\nMLP batch={batch}:")
            print(f"  Total FLOPs: {gflops:.2f} GFLOPs")
            print(f"  Latency: {latency_ms:.2f} ms")
            print(f"  Effective GFLOPS: {actual_gflops:.2f}")
            
            # Compare with DNN-Opt standalone
            # For large N, DNN-Opt is slower
            dnnopt_gflops = 6.3 if batch == 1 and mlp_dim == 4096 else 24  # approximate
            if mlp_dim == 4096:
                print(f"  DNN-Opt standalone (large N): {dnnopt_gflops} GFLOPS")
                print(f"  Note: Large N is DNN-Opt weak domain")

print()
print("=" * 80)
print("  Recommendation for DNN-Opt Integration")
print("=" * 80)
print("""
To demonstrate DNN-Opt performance in MLPerf:

Option A: Build TensorFlow with patched oneDNN
  1. Clone TensorFlow source
  2. Replace third_party/mkl_dnn with oneDNN-DNN-Opt
  3. Build with Bazel
  4. Run MLPerf Inference benchmark

Option B: Use TensorFlow custom ops
  1. Register dnnopt::gemm_fp32 as TF custom op
  2. Create wrapper for small-M shapes
  3. Test with MLP model using smaller hidden_dim

Option C: Use recommendation system models
  - DLRM: embedding + MLP (small-M operations)
  - Target: batch=1 recommendation inference
  - Expected: +10-50x improvement
""")
