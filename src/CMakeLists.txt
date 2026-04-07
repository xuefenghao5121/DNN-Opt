add_library(dnnopt_core STATIC
    hwcaps/arm_hwcaps.cpp
    utils/timer.cpp
    utils/aligned_alloc.cpp
    gemm/gemm.cpp
    gemm/gemm_driver_fp32.cpp
    gemm/gemm_pack_fp32.cpp
    gemm/gemm_ukernel_fp32_neon.cpp
    gemm/gemm_smallm_fp32.cpp
    gemm/gemm_pack_bf16.cpp
    gemm/gemm_ukernel_bf16_neon.cpp
    gemm/gemm_driver_bf16.cpp
    gemm/gemm_pack_int8.cpp
    gemm/gemm_ukernel_int8_neon.cpp
    gemm/gemm_driver_int8.cpp
)

target_include_directories(dnnopt_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(dnnopt_core PUBLIC OpenMP::OpenMP_CXX)
endif()
