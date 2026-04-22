if(DNNOPT_BUILD_TESTS)
    # Simple test framework - no external dependencies
    add_executable(test_hwcaps test_hwcaps.cpp)
    target_link_libraries(test_hwcaps PRIVATE dnnopt_core)
    add_test(NAME test_hwcaps COMMAND test_hwcaps)

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

    add_executable(test_autotune test_autotune.cpp)
    target_link_libraries(test_autotune PRIVATE dnnopt_core)
    add_test(NAME test_autotune COMMAND test_autotune)

    add_executable(test_autotune_warmup test_autotune_warmup.cpp)
    target_link_libraries(test_autotune_warmup PRIVATE dnnopt_core)
    add_test(NAME test_autotune_warmup COMMAND test_autotune_warmup)

    # v2 blocking autotune tests
    add_executable(test_autotune_blocking test_autotune_blocking.cpp)
    target_link_libraries(test_autotune_blocking PRIVATE dnnopt_core)
    add_test(NAME test_autotune_blocking COMMAND test_autotune_blocking)

    # v2 tile size autotune tests
    add_executable(test_tile_autotune test_tile_autotune.cpp)
    target_link_libraries(test_tile_autotune PRIVATE dnnopt_core)
    add_test(NAME test_tile_autotune COMMAND test_tile_autotune)

    # v2.5 threshold autotune tests
    add_executable(test_threshold_autotune test_threshold_autotune.cpp)
    target_link_libraries(test_threshold_autotune PRIVATE dnnopt_core)
    add_test(NAME test_threshold_autotune COMMAND test_threshold_autotune)

    if(DNNOPT_BUILD_BLAS)
        add_executable(test_cblas test_cblas.cpp)
        target_link_libraries(test_cblas PRIVATE dnnopt_blas)
        add_test(NAME test_cblas COMMAND test_cblas)
    endif()

    # Benchmarks
    add_executable(bench_fix_verification bench_fix_verification.cpp)
    target_link_libraries(bench_fix_verification PRIVATE dnnopt_core)

    add_executable(bench_autotune_potential bench_autotune_potential.cpp)
    target_link_libraries(bench_autotune_potential PRIVATE dnnopt_core)

    add_executable(bench_tile_selection bench_tile_selection.cpp)
    target_link_libraries(bench_tile_selection PRIVATE dnnopt_core)

    add_executable(bench_tile_perf bench_tile_perf.cpp)
    target_link_libraries(bench_tile_perf PRIVATE dnnopt_core)

    add_executable(bench_tile_padding bench_tile_padding.cpp)
    target_link_libraries(bench_tile_padding PRIVATE dnnopt_core)

    add_executable(bench_full_autotune bench_full_autotune.cpp)
    target_link_libraries(bench_full_autotune PRIVATE dnnopt_core)

    add_executable(bench_large_shapes bench_large_shapes.cpp)
    target_link_libraries(bench_large_shapes PRIVATE dnnopt_core)

    add_executable(bench_very_large bench_very_large.cpp)
    target_link_libraries(bench_very_large PRIVATE dnnopt_core)

    add_executable(bench_exotic_shapes bench_exotic_shapes.cpp)
    target_link_libraries(bench_exotic_shapes PRIVATE dnnopt_core)

    add_executable(bench_tall_skinny_deep bench_tall_skinny_deep.cpp)
    target_link_libraries(bench_tall_skinny_deep PRIVATE dnnopt_core)

    add_executable(bench_dispatch_path bench_dispatch_path.cpp)
    target_link_libraries(bench_dispatch_path PRIVATE dnnopt_core)

    add_executable(bench_irregular_shapes bench_irregular_shapes.cpp)
    target_link_libraries(bench_irregular_shapes PRIVATE dnnopt_core)

    add_executable(bench_analyze_extremes bench_analyze_extremes.cpp)
    target_link_libraries(bench_analyze_extremes PRIVATE dnnopt_core)

    # Threshold candidate benchmark (v2.5)
    add_executable(bench_threshold_candidates bench_threshold_candidates.cpp)
    target_link_libraries(bench_threshold_candidates PRIVATE dnnopt_core)

    # oneDNN vs DNN-Opt autotune comparison for irregular shapes
    add_executable(bench_onednn_vs_autotune bench_onednn_vs_autotune.cpp)
    target_include_directories(bench_onednn_vs_autotune PRIVATE
        /root/onednn-dnnopt/include
        /root/onednn-dnnopt/build/include
    )
    target_link_libraries(bench_onednn_vs_autotune PRIVATE
        dnnopt_core
        /root/onednn-dnnopt/build/src/libdnnl.so.3
    )

    # oneDNN vs DNN-Opt comparison (requires oneDNN library)
    add_executable(bench_onednn_vs_dnnopt bench_onednn_vs_dnnopt.cpp)
    target_include_directories(bench_onednn_vs_dnnopt PRIVATE
        /root/onednn-dnnopt/include
        /root/onednn-dnnopt/build/include
    )
    target_link_libraries(bench_onednn_vs_dnnopt PRIVATE
        dnnopt_core
        /root/onednn-dnnopt/build/src/libdnnl.so.3
    )

    if(DNNOPT_BUILD_BLAS)
        add_executable(bench_tf_like_inference bench_tf_like_inference.cpp)
        # Link against oneDNN for dnnl_sgemm comparison
        target_include_directories(bench_tf_like_inference PRIVATE
            /root/onednn-dnnopt/include
            /root/onednn-dnnopt/build/include
        )
        target_link_libraries(bench_tf_like_inference PRIVATE
            dnnopt_blas
            /root/onednn-dnnopt/build/src/libdnnl.so.3
        )
    endif()

    # Convenience target
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j${CMAKE_BUILD_PARALLEL_LEVEL}
        COMMENT "Running all regression tests"
    )
endif()
