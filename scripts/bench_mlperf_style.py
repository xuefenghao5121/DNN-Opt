#!/usr/bin/env python3
"""
MLPerf-style End-to-End Inference Benchmark

Test scenarios:
1. TensorFlow Eigen (TF_DISABLE_ONEDNN=1) - baseline
2. TensorFlow oneDNN (default) - baseline
3. TensorFlow + oneDNN + DNN-Opt (requires patched TF build)

Models tested:
- BERT-like (transformer, attention layers)
- ResNet50-like (CNN, conv layers)
- MobileNet-like (lightweight CNN)
- MLP-like (pure Dense layers - DNN-Opt target)

Benchmark metrics:
- Single-stream latency (batch=1)
- Multi-stream throughput (batch=4,8)
- Offline throughput (large batch)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================
# Model definitions (MLPerf-like workloads)
# ============================================================

def create_bert_model(seq_len=384, hidden_dim=768, num_layers=4):
    """BERT-like transformer model for NLP inference."""
    inputs = tf.keras.Input(shape=(seq_len, hidden_dim), name='input_ids')
    
    x = inputs
    for i in range(num_layers):
        # Multi-head attention (small-M GEMM in QKV projection)
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=12, key_dim=64, name=f'attention_{i}'
        )
        x = attn(x, x)
        
        # Feed-forward network (oneDNN optimized for M>=64)
        x = tf.keras.layers.Dense(3072, activation='gelu', name=f'ffn1_{i}')(x)
        x = tf.keras.layers.Dense(hidden_dim, name=f'ffn2_{i}')(x)
    
    outputs = tf.keras.layers.Dense(hidden_dim, name='output')(x)
    model = tf.keras.Model(inputs, outputs, name='bert_like')
    
    # Compile for inference optimization
    model.compile()
    return model

def create_resnet50_model(input_shape=(224, 224, 3), num_classes=1000):
    """ResNet50-like CNN for vision inference."""
    inputs = tf.keras.Input(shape=input_shape, name='images')
    
    # Initial conv
    x = tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', name='conv1')(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    
    # ResNet blocks (simplified)
    for stage, filters in enumerate([64, 128, 256, 512]):
        for block in range(2):
            x = tf.keras.layers.Conv2D(filters, 3, padding='same', 
                                       name=f'conv{stage}_{block}')(x)
            x = tf.keras.layers.BatchNormalization(name=f'bn{stage}_{block}')(x)
            x = tf.keras.layers.ReLU(name=f'relu{stage}_{block}')(x)
        x = tf.keras.layers.MaxPooling2D(2, name=f'pool{stage}')(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D(name='gap')(x)
    outputs = tf.keras.layers.Dense(num_classes, name='classifier')(x)
    
    model = tf.keras.Model(inputs, outputs, name='resnet50_like')
    model.compile()
    return model

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=1000):
    """MobileNet-like lightweight CNN."""
    inputs = tf.keras.Input(shape=input_shape, name='images')
    
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Depthwise separable convs
    for filters in [64, 128, 256, 512]:
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Conv2D(filters, 1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    
    model = tf.keras.Model(inputs, outputs, name='mobilenet_like')
    model.compile()
    return model

def create_mlp_model(input_dim=4096, hidden_dim=4096, num_layers=8):
    """MLP model for pure Dense layer testing (DNN-Opt target domain)."""
    inputs = tf.keras.Input(shape=(input_dim,), name='features')
    
    x = inputs
    for i in range(num_layers):
        # Dense layers: M=batch_size, N/K=hidden_dim
        # Small batch (M=1-8) triggers DNN-Opt optimization
        x = tf.keras.layers.Dense(hidden_dim, activation='relu', 
                                   name=f'dense_{i}')(x)
    
    outputs = tf.keras.layers.Dense(input_dim, name='output')(x)
    model = tf.keras.Model(inputs, outputs, name='mlp_like')
    model.compile()
    return model

# ============================================================
# MLPerf-style benchmark functions
# ============================================================

def load_model(model_name, **kwargs):
    """Load/create model."""
    models = {
        'bert': create_bert_model,
        'resnet': create_resnet50_model,
        'mobilenet': create_mobilenet_model,
        'mlp': create_mlp_model,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Creating {model_name} model...")
    model = models[model_name](**kwargs)
    return model

def generate_input(model_name, batch_size):
    """Generate random input for model."""
    if model_name == 'bert':
        return tf.random.normal([batch_size, 384, 768])
    elif model_name == 'resnet':
        return tf.random.normal([batch_size, 224, 224, 3])
    elif model_name == 'mobilenet':
        return tf.random.normal([batch_size, 224, 224, 3])
    elif model_name == 'mlp':
        return tf.random.normal([batch_size, 4096])
    else:
        raise ValueError(f"Unknown model: {model_name}")

def benchmark_single_stream(model, input_data, num_queries=1000, warmup=100):
    """Single-stream latency benchmark (MLPerf style)."""
    # Warmup
    for _ in range(warmup):
        _ = model(input_data, training=False)
    
    # Measure latency for each query
    latencies = []
    for _ in range(num_queries):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms
    
    # MLPerf metrics
    results = {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p90_latency_ms': np.percentile(latencies, 90),
        'p99_latency_ms': np.percentile(latencies, 99),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'throughput_qps': 1000.0 / np.mean(latencies),  # queries per second
        'num_queries': num_queries,
    }
    
    return results

def benchmark_multi_stream(model, input_data, batch_size, num_queries=500, warmup=50):
    """Multi-stream throughput benchmark."""
    # Warmup
    for _ in range(warmup):
        _ = model(input_data, training=False)
    
    # Measure throughput
    latencies = []
    for _ in range(num_queries):
        start = time.perf_counter()
        _ = model(input_data, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
    
    results = {
        'batch_size': batch_size,
        'mean_latency_ms': np.mean(latencies),
        'throughput_qps': batch_size * 1000.0 / np.mean(latencies),
        'throughput_ips': batch_size * num_queries / np.sum(latencies) * 1000,  # images/sec
        'num_queries': num_queries,
    }
    
    return results

def benchmark_offline(model, input_data, batch_size, num_queries=100, warmup=20):
    """Offline throughput benchmark (maximum throughput)."""
    # Warmup
    for _ in range(warmup):
        _ = model(input_data, training=False)
    
    # Measure total throughput
    total_samples = batch_size * num_queries
    start = time.perf_counter()
    for _ in range(num_queries):
        _ = model(input_data, training=False)
    end = time.perf_counter()
    
    total_time = end - start
    throughput = total_samples / total_time  # samples/sec
    
    results = {
        'batch_size': batch_size,
        'total_samples': total_samples,
        'total_time_s': total_time,
        'throughput_sps': throughput,
    }
    
    return results

# ============================================================
# Main benchmark runner
# ============================================================

def run_benchmark(model_name, scenario, batch_size, num_queries, warmup):
    """Run complete benchmark for a model/scenario."""
    
    # Load model
    model = load_model(model_name)
    
    # Generate input
    input_data = generate_input(model_name, batch_size)
    
    # Run benchmark based on scenario
    if scenario == 'single':
        results = benchmark_single_stream(model, input_data, num_queries, warmup)
    elif scenario == 'multi':
        results = benchmark_multi_stream(model, input_data, batch_size, num_queries, warmup)
    elif scenario == 'offline':
        results = benchmark_offline(model, input_data, batch_size, num_queries, warmup)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Add model info
    results['model'] = model_name
    results['scenario'] = scenario
    results['backend'] = 'Eigen' if os.environ.get('TF_DISABLE_ONEDNN') == '1' else 'oneDNN'
    results['batch_size'] = batch_size
    
    return results

def print_results(results):
    """Print results in MLPerf format."""
    print(f"\n{'='*70}")
    print(f"  MLPerf-style Results: {results['model']} ({results['scenario']})")
    print(f"{'='*70}")
    print(f"Backend: {results['backend']}")
    print(f"Batch size: {results['batch_size']}")
    print(f"")
    
    if results['scenario'] == 'single':
        print(f"Single-stream latency:")
        print(f"  Mean:   {results['mean_latency_ms']:.3f} ms")
        print(f"  Median: {results['median_latency_ms']:.3f} ms")
        print(f"  P90:    {results['p90_latency_ms']:.3f} ms")
        print(f"  P99:    {results['p99_latency_ms']:.3f} ms")
        print(f"  Min:    {results['min_latency_ms']:.3f} ms")
        print(f"  Max:    {results['max_latency_ms']:.3f} ms")
        print(f"")
        print(f"Throughput: {results['throughput_qps']:.2f} queries/sec")
    elif results['scenario'] == 'multi':
        print(f"Multi-stream throughput:")
        print(f"  Latency: {results['mean_latency_ms']:.3f} ms")
        print(f"  QPS:     {results['throughput_qps']:.2f} queries/sec")
        print(f"  IPS:     {results['throughput_ips']:.2f} images/sec")
    elif results['scenario'] == 'offline':
        print(f"Offline throughput:")
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Total time:    {results['total_time_s']:.3f} s")
        print(f"  Throughput:    {results['throughput_sps']:.2f} samples/sec")

def main():
    parser = argparse.ArgumentParser(description='MLPerf-style inference benchmark')
    parser.add_argument('--model', default='all', 
                       choices=['bert', 'resnet', 'mobilenet', 'mlp', 'all'],
                       help='Model to benchmark')
    parser.add_argument('--scenario', default='all',
                       choices=['single', 'multi', 'offline', 'all'],
                       help='Benchmark scenario')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--queries', type=int, default=100, help='Number of queries')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  MLPerf-style End-to-End Inference Benchmark")
    print("=" * 70)
    
    # Check backend
    backend = 'Eigen' if os.environ.get('TF_DISABLE_ONEDNN') == '1' else 'oneDNN'
    print(f"\nTensorFlow backend: {backend}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"")
    
    # Determine models to test
    models = ['bert', 'resnet', 'mobilenet', 'mlp'] if args.model == 'all' else [args.model]
    
    # Determine scenarios
    scenarios = ['single', 'multi'] if args.scenario == 'all' else [args.scenario]
    
    # Batch sizes for each scenario
    batch_sizes = {
        'single': [1],
        'multi': [1, 4, 8],
        'offline': [32, 64],
    }
    
    all_results = []
    
    for model in models:
        for scenario in scenarios:
            for batch_size in batch_sizes[scenario]:
                print(f"\n>>> Testing {model} ({scenario}, batch={batch_size})")
                
                try:
                    results = run_benchmark(
                        model, scenario, batch_size,
                        args.queries, args.warmup
                    )
                    print_results(results)
                    all_results.append(results)
                except Exception as e:
                    print(f"Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    
    print(f"\n{'Model':<12} {'Scenario':<10} {'Batch':<8} {'Latency(ms)':<12} {'Throughput':<15}")
    print("-" * 70)
    
    for r in all_results:
        model = r['model']
        scenario = r['scenario']
        batch = r['batch_size']
        
        if scenario == 'single':
            latency = r['median_latency_ms']
            throughput = f"{r['throughput_qps']:.2f} qps"
        elif scenario == 'multi':
            latency = r['mean_latency_ms']
            throughput = f"{r['throughput_ips']:.2f} ips"
        else:
            latency = r['total_time_s'] * 1000 / r['num_queries']
            throughput = f"{r['throughput_sps']:.2f} sps"
        
        print(f"{model:<12} {scenario:<10} {batch:<8} {latency:<12.3f} {throughput:<15}")
    
    # Save results
    output_file = f"/tmp/mlperf_results_{backend}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
