#!/usr/bin/env python3
"""
Batch-generate all NEON FP32 packed microkernels.

Run from project root:
  python3 tools/gen_all_kernels.py

Outputs C++ source files to src/gemm/generated/ and prints the
CMake source list to add to src/CMakeLists.txt.
"""

import os
import subprocess
import sys

# Kernel configurations: (Mr, Nr, K_unroll, priority)
KERNELS = [
    (4, 16, 4, 110),   # M=4 packed path
    (4, 12, 4, 105),   # N-tail friendly
    (4,  8, 4,  95),   # Small-N shapes
    (8, 12, 2, 100),   # Nr=12 tile
    (2, 16, 4,  90),   # M=2 packed path
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "gemm", "generated")
GEN_SCRIPT = os.path.join(SCRIPT_DIR, "gen_neon_gemm.py")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generated_files = []

    for mr, nr, k_unroll, priority in KERNELS:
        filename = f"neon_fp32_{mr}x{nr}.cpp"
        filepath = os.path.join(OUTPUT_DIR, filename)

        print(f"  Generating {filename} (Mr={mr}, Nr={nr}, K_unroll={k_unroll}, priority={priority})...")

        result = subprocess.run(
            [sys.executable, GEN_SCRIPT, str(mr), str(nr), str(k_unroll), str(priority)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"  ERROR generating {filename}:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            sys.exit(1)

        with open(filepath, "w") as f:
            f.write(result.stdout)

        generated_files.append(filepath)

    print(f"\nGenerated {len(generated_files)} kernels in {OUTPUT_DIR}/")
    print()
    print("Add to src/CMakeLists.txt:")
    for filepath in generated_files:
        rel = os.path.relpath(filepath, os.path.join(PROJECT_ROOT, "src"))
        print(f"  ${{CMAKE_CURRENT_SOURCE_DIR}}/{rel}")


if __name__ == "__main__":
    main()
