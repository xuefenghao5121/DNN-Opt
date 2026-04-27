#!/usr/bin/env python3
"""Parse benchdnn matmul output and compare baseline vs patched."""

import re
import sys

def parse_benchdnn_output(filename):
    """Extract shape and performance from benchdnn output."""
    results = {}

    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    current_shape = None
    current_m, current_n, current_k = 0, 0, 0

    for line in lines:
        # Extract shape from "--- Shape: MxK:KxN ---"
        if '--- Shape:' in line:
            parts = line.split(':')[1].strip().split()
            if parts:
                shape_str = parts[0]  # e.g., "1x4096:4096x1024"
                # Parse MxK:KxN format
                matrices = shape_str.split(':')
                if len(matrices) == 2:
                    a_dims = matrices[0].split('x')
                    b_dims = matrices[1].split('x')
                    if len(a_dims) == 2 and len(b_dims) == 2:
                        current_m = int(a_dims[0])
                        current_k = int(a_dims[1])
                        current_n = int(b_dims[1])
                        current_shape = (current_m, current_n, current_k)

        # Extract performance from "perf,cpu,impl,..." line
        # Format: perf,%engine%,%impl%,...,%Gops%,...,%-Gflops%,...
        if line.startswith('perf,cpu,'):
            try:
                parts = line.split(',')
                # Find index of %-Gflops% (index 8 typically)
                # Format: perf,cpu,impl,name,prb,Gops,ctime,time,Gflops,...
                for i, p in enumerate(parts):
                    if 'Gflops' in p.lower() or (i == 8 and p.replace('.','',1).isdigit()):
                        pass
                # GFLOPS is at position 8 (after time)
                if len(parts) >= 9:
                    gflops_str = parts[8]
                    if gflops_str and gflops_str.replace('.','',1).replace('-','',1).isdigit():
                        gflops = float(gflops_str)
                        if current_shape:
                            results[current_shape] = gflops
            except (ValueError, IndexError):
                pass

    return results

def compare_results(baseline_file, patched_file):
    """Compare baseline and patched results."""
    baseline = parse_benchdnn_output(baseline_file)
    patched = parse_benchdnn_output(patched_file)

    print("=" * 80)
    print("benchdnn MatMul Performance Comparison")
    print("=" * 80)
    print(f"{'Shape (M,N,K)':<20} {'Baseline':<12} {'Patched':<12} {'Speedup':<10} {'Status'}")
    print("-" * 80)

    # Sort by M dimension
    all_shapes = sorted(set(baseline.keys()) | set(patched.keys()), key=lambda x: x[0])

    total_base = 0
    total_patch = 0
    count = 0

    for shape in all_shapes:
        M, N, K = shape
        shape_str = f"{M},{N},{K}"

        base_gf = baseline.get(shape, 0)
        patch_gf = patched.get(shape, 0)

        if base_gf > 0 and patch_gf > 0:
            speedup = patch_gf / base_gf
            status = "OK"
            total_base += base_gf
            total_patch += patch_gf
            count += 1
        elif base_gf == 0 and patch_gf > 0:
            speedup = float('inf')
            status = "BASELINE_N/A"
        elif base_gf > 0 and patch_gf == 0:
            speedup = 0
            status = "CRASHED"
        else:
            speedup = 0
            status = "N/A"

        if speedup == float('inf'):
            speedup_str = "N/A"
        elif speedup > 0:
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        base_str = f"{base_gf:.2f}" if base_gf > 0 else "CRASH/N/A"
        patch_str = f"{patch_gf:.2f}" if patch_gf > 0 else "CRASH/N/A"

        print(f"{shape_str:<20} {base_str:<12} {patch_str:<12} {speedup_str:<10} {status}")

    print("-" * 80)
    if count > 0:
        avg_base = total_base / count
        avg_patch = total_patch / count
        avg_speedup = avg_patch / avg_base if avg_base > 0 else 0
        print(f"Average GFLOPS (working shapes): Baseline={avg_base:.2f}, Patched={avg_patch:.2f}")
        print(f"Average speedup: {avg_speedup:.2f}x")
    print("=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: compare_benchdnn_results.py <baseline_file> <patched_file>")
        sys.exit(1)

    compare_results(sys.argv[1], sys.argv[2])