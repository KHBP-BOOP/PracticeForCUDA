cmake_minimum_required(VERSION 3.28) #设置CMake最小版本

project (PracticeForCUDA_cmake) #设置工程名

enable_language(CXX)
enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE release)
endif()

string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -Xptxas=-v")





#生成可执行文件
add_executable(EXEFILE
    src/Main.cpp
#    src/vecAdd.cu
#    src/vectorDotProduct.cu
#    src/DEVICE.cu
#    src/treeReduction.cu
    src/SGEMM.cu
    src/testSGEMM.cu
)

set_source_files_properties(src/blockReduceSum.cu PROPERTIES COMPILE_FLAGS "-g -lineinfo")

# 优先使用 CUDAToolkit 自带的 ncu（即使 CUDA bin 不在 PATH 中也能找到）
find_program(NCU_EXECUTABLE ncu HINTS "${CUDAToolkit_BIN_DIR}")
if(NCU_EXECUTABLE)
    message(STATUS "Found Nsight Compute: ${NCU_EXECUTABLE}")

    # 报告基础文件名（ncu 会自动追加 .ncu-rep 后缀）和输出目录
    set(NCU_OUTPUT_NAME "my_profile_result" CACHE STRING "Base name of the ncu report")
    set(NCU_OUTPUT_DIR "${CMAKE_BINARY_DIR}/ncu_reports" CACHE PATH "Directory where ncu reports are saved")

    # 可选：只对匹配的 kernel 做 profile（regex 语法，如 regex:blockReduceSum）；留空则 profile 全部 kernel
    set(NCU_KERNEL_FILTER "" CACHE STRING "Optional ncu --kernel-name regex; empty profiles all kernels")
    # 可选：透传给 ncu 的附加参数，例如 "--launch-count 1 --launch-skip 0"
    set(NCU_EXTRA_ARGS "" CACHE STRING "Extra arguments passed to ncu")
    # 遇到 ERR_NVGPUCTRPERM（性能计数器权限不足）时开启，用 sudo 运行 ncu
    option(NCU_USE_SUDO "Run ncu with sudo (needs passwordless sudo or an interactive terminal)" OFF)

    # 确保输出目录存在
    file(MAKE_DIRECTORY ${NCU_OUTPUT_DIR})

    set(_ncu_cmd ${NCU_EXECUTABLE})
    if(NCU_USE_SUDO)
        list(INSERT _ncu_cmd 0 sudo)
    endif()

    set(_ncu_args --set full -f)
    if(NCU_EXTRA_ARGS)
        separate_arguments(_ncu_extra NATIVE_COMMAND "${NCU_EXTRA_ARGS}")
        list(APPEND _ncu_args ${_ncu_extra})
    endif()
    if(NCU_KERNEL_FILTER)
        list(APPEND _ncu_args --kernel-name "${NCU_KERNEL_FILTER}")
    endif()
    list(APPEND _ncu_args -o "${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}" $<TARGET_FILE:EXEFILE>)

    # 3. 添加自定义分析 target
    add_custom_target(profile
        COMMAND ${_ncu_cmd} ${_ncu_args}
        DEPENDS EXEFILE
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running Nsight Compute profiling on EXEFILE -> ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
        VERBATIM
    )
    add_custom_command(TARGET profile POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Profile report saved to: ${NCU_OUTPUT_DIR}/${NCU_OUTPUT_NAME}.ncu-rep"
    )
else()
    message(WARNING "Nsight Compute (ncu) not found in PATH. Profile target will not be available.")
endif()


target_include_directories(EXEFILE PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
