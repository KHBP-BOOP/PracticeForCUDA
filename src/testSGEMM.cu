// 主机端 SGEMM 测试代码：
//   1. 正确性验证：与 CPU 双精度参考实现对比（固定随机种子，结果可复现）
//   2. 性能测试：预热 + cudaEvent 计时，输出 TFLOPS 与估算峰值利用率
//
// 核函数的输入约束（详见 src/SGEMM.cu）：
//   - M 任意；N 须为 4 的倍数（float4 写回/行对齐）；K 须为 4 的倍数（LDG.128 对齐），
//     且 K 最优取 BK=8 的倍数

#include "SGEMM.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = (call);                                                 \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d : %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

// 与 src/SGEMM.cu 中实例化的模板参数保持一致（用于计算 Grid 配置）
constexpr int BM = 128, BN = 128;
constexpr int BLOCK_SIZE = 256;

// CPU 参考实现：i-k-j 循环顺序对 cache 友好，双精度累加作为正确性基准
static void sgemm_cpu_ref(const std::vector<float>& A, const std::vector<float>& B,
                          std::vector<double>& C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            double a = A[i * K + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += a * static_cast<double>(B[k * N + j]);
            }
        }
    }
}

// 与 CPU 基准对比，返回是否通过
// 判据：绝对误差与相对误差同时超过阈值才记为失配（浮点累加顺序不同，存在固有舍入差异）
static bool verify_result(const std::vector<float>& gpu, const std::vector<double>& ref,
                          int M, int N)
{
    double max_abs_err = 0.0, max_rel_err = 0.0;
    long long mismatches = 0;
    const long long total = static_cast<long long>(M) * N;
    for (long long i = 0; i < total; ++i) {
        double abs_err = std::abs(static_cast<double>(gpu[i]) - ref[i]);
        double rel_err = abs_err / (std::abs(ref[i]) + 1e-6);
        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        if (abs_err > 1e-2 && rel_err > 1e-2) {
            if (mismatches < 5) {
                std::cout << "  失配: C[" << i << "] GPU = " << gpu[i]
                          << ", CPU = " << ref[i] << std::endl;
            }
            ++mismatches;
        }
    }
    std::cout << "最大绝对误差: " << max_abs_err
              << ", 最大相对误差: " << max_rel_err << std::endl;
    if (mismatches > 0) {
        std::cout << "失配元素个数: " << mismatches << " / " << total << std::endl;
    }
    return mismatches == 0;
}

// RTX 4060 Laptop GPU 的 FP32 峰值（TFLOPS），用于计算利用率
// 注：CUDA 13 运行时 API 已不再提供 clockRate 等设备属性，故采用常量估算
constexpr double PEAK_TFLOPS = 15.0;

// 运行一次完整的 SGEMM 测试：分配内存 → 随机初始化 → 核函数预热/计时 → 正确性校验
// iterations > 0 时进行性能测试（预热 + 循环计时取平均），为 0 时仅校验正确性
static bool run_sgemm_test(int M, int N, int K, int iterations)
{
    std::cout << "========================================" << std::endl;
    std::cout << "测试规模    : M = " << M << ", N = " << N << ", K = " << K << std::endl;

    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t b_bytes = static_cast<size_t>(K) * N * sizeof(float);
    size_t c_bytes = static_cast<size_t>(M) * N * sizeof(float);

    // 1. 分配主机内存并用 [-1, 1] 均匀分布随机数初始化
    std::vector<float> h_A(static_cast<size_t>(M) * K);
    std::vector<float> h_B(static_cast<size_t>(K) * N);
    std::vector<float> h_C(static_cast<size_t>(M) * N, 0.0f);

    std::mt19937 rng(42); // 固定种子，保证结果可复现
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_A) v = dist(rng);
    for (auto& v : h_B) v = dist(rng);

    // 2. 分配设备（GPU）内存并拷贝输入数据
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, a_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    // 3. 配置 Grid/Block 并启动核函数
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 预热 GPU（Warm-up），消除驱动懒加载和显卡从省电模式唤醒的延迟
    launch_sgemm_thread_tiling(d_A, d_B, d_C, M, N, K, grid, block);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. 循环执行多次，取平均时间以获得更稳定的性能数据
    float avg_milliseconds = 0.0f;
    if (iterations > 0) {
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            launch_sgemm_thread_tiling(d_A, d_B, d_C, M, N, K, grid, block);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float total_milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
        avg_milliseconds = total_milliseconds / iterations;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    // 5. 将结果拷贝回主机，与 CPU 基准对比
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

    std::vector<double> h_ref(static_cast<size_t>(M) * N, 0.0);
    std::cout << "Running CPU reference..." << std::endl;
    sgemm_cpu_ref(h_A, h_B, h_ref, M, N, K);
    bool pass = verify_result(h_C, h_ref, M, N);

    // 6. 打印测试报告
    std::cout << "Grid 配置   : " << grid.x << " x " << grid.y << " Blocks, "
              << BLOCK_SIZE << " Threads/Block" << std::endl;
    std::cout << "结果验证    : " << (pass ? "通过 (PASS)" : "失败 (FAIL)") << std::endl;
    if (iterations > 0) {
        // FLOPs = 2*M*N*K（每个输出元素需 K 次乘加）
        double tflops = 2.0 * M * N * K / (avg_milliseconds * 1e-3) / 1e12;
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "平均计算耗时: " << avg_milliseconds << " ms" << std::endl;
        std::cout << "计算性能    : " << tflops << " TFLOPS" << std::endl;
        std::cout << "峰值利用率  : " << 100.0 * tflops / PEAK_TFLOPS << " %" << std::endl;
    }

    // 7. 释放资源
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return pass;
}

void testSGEMM()
{
    bool pass = true;

    // 性能 + 正确性测试：规整尺寸
    pass &= run_sgemm_test(1024, 1024, 1024, 20);

    // 边界正确性测试：M/N/K 均不是 Tile 尺寸的整数倍
    // 注意：为保证 LDG.128/STG.128 的 16B 地址对齐，N 与 K 仍必须是 4 的倍数
    pass &= run_sgemm_test(999, 1028, 1020, 0);

    // 更小的非规整尺寸（M % 128 = 1, N % 128 = 4, K % 8 = 4）
    pass &= run_sgemm_test(129, 132, 132, 0);

    // 小尺寸：M、N 均小于单个 Tile，覆盖写回越界保护
    pass &= run_sgemm_test(64, 64, 64, 0);

    // 极限小尺寸：M=1, N=4, K=4，覆盖 per-element 写回路径
    pass &= run_sgemm_test(1, 4, 4, 0);

    std::cout << "========================================" << std::endl;
    std::cout << "总体结果    : " << (pass ? "全部通过 (ALL PASS)" : "存在失败 (FAIL)") << std::endl;
}
