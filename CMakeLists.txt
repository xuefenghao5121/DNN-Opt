cmake_minimum_required(VERSION 3.16)
project(dnn-opt
    VERSION 0.1.0
    LANGUAGES C CXX ASM
    DESCRIPTION "oneDNN ARM Platform Optimization Toolkit"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================
# Architecture detection
# ============================================================
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    set(DNNOPT_AARCH64 ON)
    message(STATUS "Target architecture: AArch64")
else()
    message(WARNING "This project targets AArch64. Current arch: ${CMAKE_SYSTEM_PROCESSOR}")
    set(DNNOPT_AARCH64 OFF)
endif()

# ============================================================
# Build options
# ============================================================
option(DNNOPT_BUILD_BENCHMARKS "Build benchmark suite" ON)
option(DNNOPT_BUILD_TESTS "Build regression tests" ON)
option(DNNOPT_USE_OPENMP "Enable OpenMP" ON)
option(DNNOPT_NATIVE_ARCH "Use -march=native" ON)
option(DNNOPT_BUILD_BLAS "Build BLAS-compatible shared library (libdnnopt_blas.so)" ON)
option(DNNOPT_ENABLE_SME "Enable SME kernels (requires SME-capable hardware)" OFF)

# ============================================================
# Compiler flags
# ============================================================
if(DNNOPT_AARCH64)
    if(DNNOPT_NATIVE_ARCH)
        # Use native arch with BF16 extension for Clang (needed for BF16 intrinsics)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(ARCH_FLAGS "-march=armv8.5-a+bf16+dotprod+fp16+i8mm")
        else()
            set(ARCH_FLAGS "-march=native")
        endif()
    else()
        # Minimum target: ARMv8.2-a with common extensions
        set(ARCH_FLAGS "-march=armv8.2-a+dotprod+fp16")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ARCH_FLAGS}")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${ARCH_FLAGS}")
endif()

# ============================================================
# SME Support
# ============================================================
if(DNNOPT_ENABLE_SME)
    message(STATUS "SME support: ENABLED")
    # SME requires ARMv9-A architecture
    set(SME_FLAGS "-march=armv9-a+sme+sme-i16i64+sme-f64f64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SME_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SME_FLAGS}")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} ${SME_FLAGS}")
    add_definitions(-DDNNOPT_HAS_SME)
else()
    message(STATUS "SME support: DISABLED (use -DDNNOPT_ENABLE_SME=ON to enable)")
endif()

# Release flags
set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Debug flags
set(CMAKE_C_FLAGS_DEBUG "-O0 -g -DDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -DDEBUG")

# Default to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================
# OpenMP
# ============================================================
if(DNNOPT_USE_OPENMP)
    # When using Clang, prefer GNU OpenMP (libgomp) for compatibility with oneDNN
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX_FLAGS "-fopenmp=libgomp" CACHE STRING "" FORCE)
        set(OpenMP_C_FLAGS "-fopenmp=libgomp" CACHE STRING "" FORCE)
    endif()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(WARNING "OpenMP not found, disabling")
        set(DNNOPT_USE_OPENMP OFF)
    endif()
endif()

# ============================================================
# Source subdirectories
# ============================================================
# Core library: hardware detection + utilities
add_subdirectory(src)

if(DNNOPT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(DNNOPT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================
# Summary
# ============================================================
message(STATUS "")
message(STATUS "=== DNN-Opt Build Configuration ===")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  AArch64:       ${DNNOPT_AARCH64}")
message(STATUS "  Arch flags:    ${ARCH_FLAGS}")
message(STATUS "  OpenMP:        ${DNNOPT_USE_OPENMP}")
message(STATUS "  Benchmarks:    ${DNNOPT_BUILD_BENCHMARKS}")
message(STATUS "  Tests:         ${DNNOPT_BUILD_TESTS}")
message(STATUS "  BLAS wrapper:  ${DNNOPT_BUILD_BLAS}")
message(STATUS "  Compiler:      ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "===================================")
