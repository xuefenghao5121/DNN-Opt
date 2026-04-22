if(DNNOPT_BUILD_TESTS)
    # Correctness tests (commercial quality)
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

    # Performance regression test
    add_executable(bench_full_coverage bench_full_coverage.cpp)
    target_link_libraries(bench_full_coverage PRIVATE dnnopt_core)

    add_executable(bench_fix_verification bench_fix_verification.cpp)
    target_link_libraries(bench_fix_verification PRIVATE dnnopt_core)

    # Convenience target
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CMAKE_BUILD_PARALLEL_LEVEL}
        COMMENT "Running all regression tests"
    )
endif()