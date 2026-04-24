if(DNNOPT_BUILD_TESTS)
    # Correctness tests (no oneDNN dependency)
    add_executable(test_gemm_correctness test_gemm_correctness.cpp)
    target_link_libraries(test_gemm_correctness PRIVATE dnnopt_core)
    add_test(NAME test_gemm_correctness COMMAND test_gemm_correctness)

    add_executable(test_conv_correctness test_conv_correctness.cpp)
    target_link_libraries(test_conv_correctness PRIVATE dnnopt_core)
    add_test(NAME test_conv_correctness COMMAND test_conv_correctness)

    add_executable(test_conv3d_correctness test_conv3d_correctness.cpp)
    target_link_libraries(test_conv3d_correctness PRIVATE dnnopt_core)
    add_test(NAME test_conv3d_correctness COMMAND test_conv3d_correctness)

    add_executable(test_elementwise_correctness test_elementwise_correctness.cpp)
    target_link_libraries(test_elementwise_correctness PRIVATE dnnopt_core)
    add_test(NAME test_elementwise_correctness COMMAND test_elementwise_correctness)

    if(DNNOPT_BUILD_BLAS)
        add_executable(test_cblas test_cblas.cpp)
        target_link_libraries(test_cblas PRIVATE dnnopt_blas)
        add_test(NAME test_cblas COMMAND test_cblas)
    endif()

    # Autotune functional test
    add_executable(test_autotune test_autotune.cpp)
    target_link_libraries(test_autotune PRIVATE dnnopt_core)
    add_test(NAME test_autotune COMMAND test_autotune)

    # Hardware detection
    add_executable(test_hwcaps test_hwcaps.cpp)
    target_link_libraries(test_hwcaps PRIVATE dnnopt_core)
    add_test(NAME test_hwcaps COMMAND test_hwcaps)

    # Performance tests (standalone, uses dnnopt::gemm_fp32)
    add_executable(bench_full_coverage bench_full_coverage.cpp)
    target_link_libraries(bench_full_coverage PRIVATE dnnopt_core)

    add_executable(bench_fix_verification bench_fix_verification.cpp)
    target_link_libraries(bench_fix_verification PRIVATE dnnopt_core)

    # Shape autotune analysis for special shapes (from shape_rec)
    add_executable(bench_shape_autotune bench_shape_autotune.cpp)
    target_link_libraries(bench_shape_autotune PRIVATE dnnopt_core)

    # oneDNN comparison tests (optional, requires oneDNN library)
    # Enable with: cmake -DDNNOPT_ENABLE_ONEDNN_TESTS=ON -DDNNOPT_ONEDNN_ROOT=/path/to/onednn
    option(DNNOPT_ENABLE_ONEDNN_TESTS "Enable tests that require oneDNN library" OFF)

    if(DNNOPT_ENABLE_ONEDNN_TESTS)
        if(NOT DNNOPT_ONEDNN_ROOT)
            message(WARNING "DNNOPT_ONEDNN_ROOT not set, skipping oneDNN comparison tests")
        else()
            # Validate oneDNN library exists
            set(ONEDNN_LIB "${DNNOPT_ONEDNN_ROOT}/build/src/libdnnl.so.3")
            if(NOT EXISTS "${ONEDNN_LIB}")
                # Try alternative paths
                set(ONEDNN_LIB "${DNNOPT_ONEDNN_ROOT}/src/libdnnl.so.3")
            endif()

            if(EXISTS "${ONEDNN_LIB}")
                message(STATUS "Found oneDNN: ${ONEDNN_LIB}")
                message(STATUS "Enabling oneDNN comparison tests")

                add_executable(bench_onednn_vs_dnnopt bench_onednn_vs_dnnopt.cpp)
                target_include_directories(bench_onednn_vs_dnnopt PRIVATE
                    ${DNNOPT_ONEDNN_ROOT}/include
                    ${DNNOPT_ONEDNN_ROOT}/build/include)
                target_link_libraries(bench_onednn_vs_dnnopt PRIVATE
                    dnnopt_core
                    ${ONEDNN_LIB}
                    dl)

                # Rebuild bench_full_coverage with oneDNN support
                target_compile_definitions(bench_full_coverage PRIVATE DNNOPT_USE_ONEDNN=1)
                target_include_directories(bench_full_coverage PRIVATE
                    ${DNNOPT_ONEDNN_ROOT}/include
                    ${DNNOPT_ONEDNN_ROOT}/build/include)
                target_link_libraries(bench_full_coverage PRIVATE
                    ${ONEDNN_LIB}
                    dl)
            else()
                message(WARNING "oneDNN library not found at ${DNNOPT_ONEDNN_ROOT}, skipping oneDNN tests")
            endif()
        endif()
    endif()

    # Convenience target
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CMAKE_BUILD_PARALLEL_LEVEL}
        COMMENT "Running all regression tests"
    )
endif()