# SGEMM（未完成）

C = A @ B

$A \in \mathbb{R}^{M \times K}$

$B \in \mathbb{R}^{K \times N}$

$C \in \mathbb{R}^{M \times N}$

***GEMM 优化的本质是用寄存器和共享内存（Shared Memory）挡住对全局内存（Global Memory）的访问。*** 


v3及之后版本的代码均为原创  
严格根据NVIDIA GeForce RTX 4060 Laptop GPU进行适配开发

GDDR6显存 8GB
L2缓存大小 24 MB  
流式多处理器个数 24  
每流式多处理器最大可驻留的Warp数 48  
每流式多处理器最大可驻留线程数 1536  

每流式多处理器寄存器文件大小 64 KB  
每流式多处理器L1数据缓存/共享内存大小 128 KB  

### 性能指标

不考虑落地工程中的端到端耗时与延迟、鲁棒性、每瓦特吞吐量等因素，算力利用率、带宽利用率是以结果为导向的衡量指标；算术强度、全局内存访问率、L1L2缓存命中率是以过程为导向的衡量指标。

![alt text](image.png)


### tiling

分块思想贯穿始终

***每下降一个内存层次，就对应线程层次的一层分块。***

## 最终version


# 优化过程


## Naive版本

SGEMM 计算强度I =

2 * M * N * K / 4 * 2 * M * N * K = 0.25FLOPs/Byte


## version 1


#include <cuda_runtime_api.h>

template <int BM, int BN, int BK, int BLOCK_SIZE>
__global__ void sgemm_block_tiling(float* A, float* B, float* C,
                                   int M, int K, int N) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int r0 = blockIdx.y * BM;//
    int c0 = blockIdx.x * BN;//
    int tid = threadIdx.x;

    // 加载 tileA 时的线程重排
    constexpr int A_BLOCK_X = BK;  // = 8
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X;  // = 32
    int a_thread_x = tid % A_BLOCK_X; // 0 ~ 7
    int a_thread_y = tid / A_BLOCK_X; // 0 ~ 31

    // 加载 tileB 时的线程重排
    constexpr int B_BLOCK_X = 32;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X;  // = 8
    int b_thread_x = tid % B_BLOCK_X;
    int b_thread_y = tid / B_BLOCK_X;

    // 计算 tileC 、写入C 时的线程排布（16×16）
    constexpr int C_BLOCK_X = 16;
    constexpr int C_BLOCK_Y = BLOCK_SIZE / C_BLOCK_X;  // = 16
    int c_thread_x = tid % C_BLOCK_X; // 0 ~ 15
    int c_thread_y = tid / C_BLOCK_X; // 0 ~ 15

    // 16 * 16 threads 负责 128 * 128 个元素
    // 每个线程负责 Tm×Tn 个输出元素
    constexpr int Tm = BM / C_BLOCK_Y;  // = 8 跨步覆盖128行 由BM、BLOCK_SIZE决定
    constexpr int Tn = BN / C_BLOCK_X;  // = 8 跨步覆盖128列 由BN决定
    float Ct[Tm][Tn] = {0.0f};

    // K-Loop
    //一次循环对应
    for (int k = 0; k < K; k += BK) {

        //r0用于A、C矩阵行索引，c0用于B、C矩阵列索引
        //A矩阵列索引、B矩阵行索引借助K-Loop中的循环变量k

        // 将tileA数据载入SMEM（使用跨步循环覆盖 BM 行）
        #pragma unroll
        for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) { //BM 128
            // i            128 * 8    32 * 8
            // 0 32 64 96
            // 1 33 65 97
            // 2 34 66 98
            // ...
            // 31 63 95 127
            // tile A  8 * 32  256
            int r = r0 + i, c = k + a_thread_x; // 128 128
            //同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定。
            As[i][a_thread_x] = (r < M && c < K) ? A[r * K + c] : 0.0f; //所有线程均运行该行代码，将HBM中的数据存入各自对应的block的SMEM
        }

        // 协作加载 tileB（使用跨步循环覆盖 BN 列）
        #pragma unroll
        for (int j = b_thread_x; j < BN; j += B_BLOCK_X) {
            int r = k + b_thread_y, c = c0 + j;
            //同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
            Bs[b_thread_y][j] = (r < K && c < N) ? B[r * N + c] : 0.0f;
        }

        //确保SMEM数据为本轮循环的数据
        __syncthreads();

        // 外积方式计算 As × Bs
        // 1个thread 64 个元素
        #pragma unroll
        for (int p = 0; p < BK; p++) {
            for (int i = 0; i < Tm; i++) {
                int row = c_thread_y + i * C_BLOCK_Y; //0~120
                //进入循环，16 * 16 个线程中，同一行线程计算出相同row，但不同于其他行
                for (int j = 0; j < Tn; j++) {
                    int col = c_thread_x + j * C_BLOCK_X; //0~120 
                    //同一列线程计算出相同col，但不同于其他列
                    Ct[i][j] += As[row][p] * Bs[p][col];
                    //8*8       128*8        8*128
                    //每一个线程负责一对As中一个元素、Bs中一个元素的FMA运算
                }
            }
        }
    
        __syncthreads(); //避免在本轮循环计算完成前，SMEM被下一轮数据覆盖
    }

    // 写回结果
    for (int i = 0; i < Tm; i++) {
        int r = r0 + c_thread_y + i * C_BLOCK_Y;
        for (int j = 0; j < Tn; j++) {
            int c = c0 + c_thread_x + j * C_BLOCK_X;
            if (r < M && c < N) C[r * N + c] = Ct[i][j];
        }
    }
}


**thread block级tiling + thread级tiling**

*block级tiling：减少对HBM的访问次数*

- 矩阵C划分为BM * BN的分块，每个 Thread Block 负责一块。相较于naive版本的从Global memory反复取值、进行FMA计算，v1先将Global memory中数据载入SMEM，再从SMEM中反复取值、进行FMA运算。

*thread级tiling实现了寄存器复用*

- 将tileA（tileB同理）分块，每个线程通过循环运输**跨A_BLOCK_Y步等距**的*BM / A_BLOCK_Y下取整*个数据

- 将tileC分为Tm * Tn个格，一个线程跨步计算每个格中的一个元素，共Tm * Tn个元素。不同于1线程1数据+内积矩乘，采用1线程多数据（扩大tileC尺寸）+外积矩乘，并通过寄存器级缓存实现寄存器复用：

借助编译器优化，1个线程加载Tm + TN个数据，完成Tm * Tn次乘加运算FMA，提高了访存比

为什么不叫tile级分块？？？


SGEMM 算数强度I =

2 * BM * BN * BK / 4 * (BM * BK + BK * BN)

BM = BN = 64  ->  I == 16 FLOPS/Byte



BK = 8
过小 -> K-Loop循环次数过多 -> 块级同步次数过多
过大 -> SMEM容量占用过多或不足

BLOCK_SIZE = 256

GPU硬件
      │
      ▼
Roofline决定需要AI
      │
      ▼
确定BM、BN
      │
      ▼
Shared Memory容量
      │
      ▼
确定BK
      │
      ▼
每线程寄存器预算
      │
      ▼
确定BLOCK_SIZE
      │
      ▼
得到Tm、Tn

二维grid 一维block256


#### 线程重排

一个tileA，128 * 8，1024个元素，在全局内存中为*行优先存储（Row-Major）*，即同行相邻列的元素在内存中是连续的，同列相邻行的元素则相隔K个距离。由一维线程块负责，包含256个thread，重排为32 * 8，使用跨步循环实现一个block覆盖tileA完整范围；


一个tileB，8 * 128，1024个元素，由一维线程块负责，包含256个thread，重排为8 * 32，使用跨步循环实现一个block覆盖tileB完整范围；

用 a_thread_x/y 和 b_thread_x/y 的线程重排索引，目的是实现合并内存访问（Coalesced Access）

一个tileC，128 * 128 个元素，由一维线程块负责，包含256个thread，重排为16 * 16，使用跨步循环实现一个block覆盖tileC完整范围；



#### 线程与tile的映射
一个block负责一排As和一列Bs的矩乘，
维度为K、步长为BK的循环中，具体一次工作流的拆解：

1. 全局内存数据载入共享内存数组As、Bs阶段：

载入As时，每个线程以A_BLOCK_Y为步长，跨步循环BM/A_BLOCK_Y（上取整）次，横跨线程对应的As数组的128行。  
载入Bs时，每个线程以B_BLOCK_X为步长，跨步循环BN/B_BLOCK_X（上取整）次，横跨线程对应的Bs数组的128列。  
每个block中所有线程同步至完成载入的阶段，此时As、Bs分别包含一个tileA、一个tileB的数据。

*进行block级别同步，是为避免部分线程载入数据仍未完成，就开始使用上次的残留数据并行计算的错误。*

2.并行计算

每个线程在Tn、Tm、BK维度的循环过程中，于Ct中累加结果，循环结束后，Ct包含As中**跨C_BLOCK_Y步等距**的多行与Bs中**跨C_BLOCK_X步等距**的多列之间的矩乘结果，它是BK维度上的，并不是对应K维度的最终可输出结果；  
每个block中所有线程同步至各自Ct计算完毕的阶段，此时每个tileA、B均完成计算，结果被浓缩进入每个线程的Ct数组； 

*进行block级别同步，是为保证先完成本次tileA、B的计算，再进行下一个tileA、B的计算，否则会出现将下一个tileA、B的数据计算结果存入本次Ct的错误*  

继续维度为K、粒度为BK的遍历；  
外层K-LOOP结束后，每个线程负责的多行多列（跨步C_BLOCK_X、C_BLOCK_Y）的计算在K维度上完成；此时Ct为该thread负责的多排多列（跨步C_BLOCK_X、C_BLOCK_Y）的最终矩乘结果。

3. 结果返回
kernel 执行结束时所有线程会自动在块内隐式同步退出

#### 写回时的分块策略

采用以16\*16为大小，分成8\*8个的策略，即一个线程跨步覆盖tileC，相邻线程合并访问相邻地址，内存事务少；  
若以8\*8为大小，分成16\*16个，warp内所有相邻线程均访问不相邻的地址，读写效率极低。




#### 外积矩阵乘法

相较于内积法，GEMM变为多个秩-1矩阵的累加，便于向量化指令、并行计算


## version 2


template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C,
                                    int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;

    // 每个线程在 C 中负责 TM×TN 的子块
    const int C_BLOCK_X = BN / TN;
    const int C_BLOCK_Y = BM / TM;
    int c_thread_x = tid / (BN / TN);
    int c_thread_y = tid / (BM / TM);
    int thread_row = c_thread_x * TM;
    int thread_col = c_thread_y * TN;

    // 寄存器存储
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};

    int by = blockIdx.y, bx = blockIdx.x;

    for (int bk = 0; bk < K; bk += BK)
    {
        // 协作加载 A、B 到 Shared Memory（省略边界检查）
        load_tile_A(A, As, by, bk, tid, M, K);
        load_tile_B(B, Bs, bx, bk, tid, K, N);
        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++)
        {
            // 从 Shared Memory 加载到寄存器
            for (int i = 0; i < TM; i++)
            {
                a_frag[i] = As[thread_row + i][k];
                // 同行的16个线程计算出的thread_row相同，执行相同操作，即16个线程读取As中的同一个地方、再将数据载入a_frag数组
                // 每一行的16个线程共享一个a_frag数组，一个block有8个a_frag数组
            }
            for (int j = 0; j < TN; j++)
            {
                b_frag[j] = Bs[k][thread_col + j];
                // 同列的16个线程计算出的thread_col相同，执行相同操作，即读取Bs中的同一个地方、再将数据载入b_frag数组
                // 每一列的16个线程共享一个b_frag数组，一个block有8个b_frag数组
            }
            // 寄存器上做外积
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    c_frag[i][j] += a_frag[i] * b_frag[j];
        }
        __syncthreads();
    }

    // 写回 Global Memory
    for (int i = 0; i < TM; i++)
    {
        int r = by * BM + i * C_BLOCK_Y + c_thread_y;
        for (int j = 0; j < TN; j++)
        {
            int c = bx * BN + j * C_BLOCK_X + c_thread_x;
            if (r < M && c < N)
                C[r * N + c] = c_frag[i][j];
        }
    }
}

v1中，外积方式计算 Ct[i][j] += As[row][p] * Bs[p][col] 时，编译器可能会为 As[row][p] 的重复读取做一定优化，但远不如v2中手工寄存器分块可控且高效



将加载部分与计算部分分离开。并行加载至寄存器，既加速数据载入，又加速计算部分；计算部分仍串行进行，但由于数据从寄存器加载，所以速度更快。

v1计算过程：  
As数组中待处理的16个元素、Bs数组中待处理的16个元素进行外积运算，该过程以循环方式串行执行，直到As、Bs计算结束；
v2：
16*16个线程并行执行，将As、Bs中待处理的16个元素（它们是跨步等距的）运输至各自线程私有的寄存器数组（此过程存在同一行/列的线程执行相同事务的情况，或者说这些线程共享同一个register file）；依次循环TM、Tn次，将As、Bs的完整一列/行各自寄存器数组，接着串行计算。  
内层循环沿BK维度遍历时，每次从共享内存加载TM个A元素和TN个B元素到寄存器a_frag、b_frag，并在寄存器上做外积累加；沿BK维度、步长为1的遍历结束后，该tileA、B的计算结束，block级同步结束后，进行下一组tileA、B的计算。

v2相较于v1，写回结果时采用了不同分块策略，

## version 3
block 级 tiling + warp 级 tiling + 线程级寄存器 tiling

template <int BM, int BN, int BK,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    const int WARP_Y = Wy; // 4
    const int WARP_X = Wx; // 8

    // 每一列 C_BLOCK_Y / WARP_Y 个 warp tile
    const int warp_tile_per_col = C_BLOCK_Y / WARP_Y;
    // 每一行 C_BLOCK_X / WARP_X 个 warp tile
    const int warp_tile_per_row = C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个数据至a_frag
            a_frag[lane_col] = As[warp_row * WARP_Y * TM + lane_row * TM + lane_col][k];

            #pragma unroll
            for (int i = 0; i < WARP_X; i++) {
                
                
                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }

            //1个线程读取2个数据至b_frag
            b_frag[lane_row * 2] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2];
            b_frag[lane_row * 2 + 1] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2 + 1];
 
            
            // 广播 b_frag
            #pragma unroll
            for (int j = 0; j < TN; j += 2) {
                
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], j * WARP_X);
                b_frag[j + 1] = __shfl_sync(0xffffffff, b_frag[j + 1], j * WARP_X);

            }

            #pragma unroll
            for (int j = 0; j < WARP_Y; j++) {
                
                b_frag[j * 2]     = __shfl_sync(0xffffffff, b_frag[j * 2],     j * WARP_X + lane_col);
                b_frag[j * 2 + 1] = __shfl_sync(0xffffffff, b_frag[j * 2 + 1], j * WARP_X + lane_col);
            }



            // if (lane_col == 0) {

            //     for (int i = 0; i < TM; i++) {
            //         a_frag[i] = As[warp_row * WARP_Y * TM + lane_row * TM + i][k];
            //     }
            // }
            

            // if (lane_row == 0) {
            //     for (int j = 0; j < TN; j++) {
            //         b_frag[j] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + j];
            //     }
            // }

            // // 广播 a_frag
            // #pragma unroll
            // for (int i = 0; i < TM; i++) {
            //     a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], lane_row * WARP_X);
            //     //每一行线程的源线程的索引跨步等距
                
            //     //a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], 0, 8);
            // }

            // // 广播 b_frag
            // #pragma unroll
            // for (int j = 0; j < TN; j++) {
            //     b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], lane_col);
            //     //每一列线程的源线程的索引相邻
            // }

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    
    // 写回用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {
        int r = base_row + i;
        for (int j = 0; j < TN; j++) {
            int c = base_col + j;
            if (r < M && c < N) C[r * N + c] = c_frag[i][j];
        }
    }

}

优化共享内存分块载入寄存器数组部分。核心为通过 warp 内 shuffle 让一条线程加载的数据被同 warp 的其它线程复用。

具体实现：  
坐标系从 “thread 在 block 内的位置”换成了“warp + lane 在 block 内的位置”；  
从warp尺寸2 * 16、block内全部线程进行读操作  
换成了  
warp尺寸4 * 8、block内**仅warp内lane的x、y方向索引为0**的*O(a * TM + b * TN)*个线程进行读操作

4*8 进行读取操作的线程数量最少

!!! micro-optimization：
彻底消除 divergence.


## version 4

向量化 +  
优化写回过程

将寄存器中8*8个数据float4向量化，每个线程每次循环执行2次float4类型数据的读写，共处理32B数据,同一warp的32个thread同时处理2 * 16 * 32B数据;
8个warp循环8次，覆盖128 * 128个float数据



```cpp
#include <cuda_runtime_api.h>

#define FLOAT4(f) *reinterpret_cast<float4*>(&f)
#define CONST_FLOAT4(f) *reinterpret_cast<const float4*>(&f)

// 协作加载 tileA: 全局内存 → Shared Memory
// r0 = blockIdx.y * BM,  k = 当前 K 维度起点
template <int BM = 128, int BK = 8, int BLOCK_SIZE>
__device__ void load_tile_A(const float *__restrict__ A, float As[BM][BK],
                            int M, int K, int r0, int k, int tid)
{
    // 线程重排: 128 * 2 布局, 实现合并访问以及float4向量化 (coalesced access)
    constexpr int A_BLOCK_X = BK / 4;                     // = 2
    constexpr int A_BLOCK_Y = BLOCK_SIZE / A_BLOCK_X; // = 128
    int a_thread_x = tid % A_BLOCK_X;                 // 0 ~ 1
    int a_thread_y = tid / A_BLOCK_X;                 // 0 ~ 127

    // float4向量化前：32 * 8个线程，每个线程做4次LDG.32，128 * 8个数据循环4次处理完成
    // float4向量化后：128 * 2个线程，每个线程做1次LDG.128，128 * 8个数据1次即可处理完成
    #pragma unroll
    for (int i = a_thread_y; i < BM; i += A_BLOCK_Y) {

        // 每个thread在block中的索引
        int r = r0 + i, c = k + a_thread_x * 4;

        // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        // 为确保LDG.128的16B对齐，K必须是4的倍数（最优情况下是 BK=8 的倍数）
        // k为8的倍数，故c一定为4的倍数；c、K均为4的倍数，故不存在数据丢失情况
        float4 temp = (r < M && c + 3 < K) ? CONST_FLOAT4(A[r * K + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

        // 同一block内，具体一次K维度的循环中，所有thread的r0一定、k一定
        FLOAT4(As[i][a_thread_x * 4]) = temp; // STS.128

        //warp32个线程中，每个线程处理一个float4类型，共512B；
        //硬件将其划分为4个128B内存事务，每个事务均占满SMEM的32个bank，流水线串行执行事务时无bank conflict
    }
}

// 协作加载 tileB: 全局内存 → Shared Memory
// c0 = blockIdx.x * BN,  k = 当前 K 维度起点
template <int BN, int BK, int BLOCK_SIZE>
__device__ void load_tile_B(const float *__restrict__ B, float Bs[BK][BN],
                            int K, int N, int c0, int k, int tid)
{
    // 线程重排: 8×32 布局, 实现合并访问
    constexpr int B_BLOCK_X = 128 / 4;
    constexpr int B_BLOCK_Y = BLOCK_SIZE / B_BLOCK_X; // = 8
    int b_thread_x = tid % B_BLOCK_X;                 // 0 ~ 31
    int b_thread_y = tid / B_BLOCK_X;                 // 0 ~ 7

    // float4向量化前：8 * 32个线程，每个线程做4次LDG.32，8 * 128个数据循环4次处理完成
    // float4向量化后：8 * 32个线程，每个线程做1次LDG.128，8 * 128个数据1次即可处理完成
    #pragma unroll
    for (int j = b_thread_y; j < BK; j += B_BLOCK_Y) {

        // 每个thread在block中的索引
        int r = k + j, c = c0 + b_thread_x * 4;

        // 计算线程块尺寸时应采用向上取整的除法，故存在线程数量大于数据总量的情况
        // 为确保LDG.128的16B对齐，N必须是4的倍数（最优情况下是 BN=128 的倍数）
        // c0 = bx * BN，故c一定为4的倍数，不存在数据丢失情况
        float4 temp = (r < K && c + 3 < N) ? CONST_FLOAT4(B[r * N + c]) : make_float4(0.f, 0.f, 0.f, 0.f); // LDG.128

        // 同一block内，具体一次K维度的循环中，所有thread的c0一定、k一定
        FLOAT4(Bs[j][b_thread_x * 4]) = temp; // STS.128
    }
}


template <int BM = 128, int BN = 128, int BK = 8,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    constexpr int WARP_Y = Wy; // 4
    constexpr int WARP_X = Wx; // 8

    //寄存器读取SMEM时线程重排，线程块尺寸
    constexpr int RLDS_C_BLOCK_X = BN / TN;
    constexpr int RLDS_C_BLOCK_Y = BM / TM;

    // 每一列 LDS_C_BLOCK_Y / WARP_Y 个 warp tile
    constexpr int warp_tile_per_col = RLDS_C_BLOCK_Y / WARP_Y;
    // 每一行 LDS_C_BLOCK_X / WARP_X 个 warp tile
    constexpr int warp_tile_per_row = RLDS_C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    //K-Loop
    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个数据至a_frag
            a_frag[lane_col] = As[warp_row * WARP_Y * TM + lane_row * TM + lane_col][k];

            // 广播 a_frag
            // 每个线程遍历各自的a_frag，从同行对应线程的或自身的a_frag中获取数据，使得同行线程的a_frag均存有正确数据且一致
            #pragma unroll
            for (int i = 0; i < TM; i++) {

                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }

            //1个线程读取2个数据至b_frag
            b_frag[lane_row * 2] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2];
            b_frag[lane_row * 2 + 1] = Bs[k][warp_col * WARP_X * TN + lane_col * TN + lane_row * 2 + 1];
 
            
            // 广播 b_frag
            // 每个线程遍历各自的b_frag，从同列对应线程的或自身的b_frag中获取数据，使得同列线程的b_frag均存有正确数据且一致
            #pragma unroll
            for (int j = 0; j < TN; j += 2) {
                
                b_frag[j] = __shfl_sync(0xffffffff, b_frag[j], j * WARP_X / 2 + lane_col);
                b_frag[j + 1] = __shfl_sync(0xffffffff, b_frag[j + 1], j * WARP_X / 2 + lane_col);

            }

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }


    // 写回全局内存，使用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {

        int r = base_row + i;
        if (r >= M) {
            //M不为8的倍数时，负责余下的不足8行数据的线程同样循环8次，但没有数据的几次循环会执行continue语句跳过
            continue;
        }

        if (base_col + TN <= N) {
            
            //2条STG.128指令
            
            float4 temp = make_float4(c_frag[i][0], c_frag[i][1], c_frag[i][2], c_frag[i][3]);
            FLOAT4(C[r * N + base_col]) = temp;

            temp = make_float4(c_frag[i][4], c_frag[i][5], c_frag[i][6], c_frag[i][7]);
            FLOAT4(C[r * N + base_col + 4]) = temp;

        }

        else {

            // N不为8的倍数时，余下的不足8个的float数据逐个传输
            for (int j = 0; j < N - base_col; ++j) {
                C[r * N + base_col + j] = c_frag[i][j];
            }

        }

    }
}

// 主机端启动封装：核函数模板在本翻译单元内完成实例化。
// 直接跨翻译单元链接 __global__ 模板实例化存在可见性问题
// （rdc=false 模式下模板实例化的 host stub 默认具有内部链接属性），
// 因此测试代码 (src/testSGEMM.cu) 通过本函数间接启动核函数
void launch_sgemm_thread_tiling(const float *A, const float *B, float *C,
                                int M, int N, int K, dim3 grid, dim3 block)
{
    sgemm_thread_tiling<128, 128, 8, 256, 8, 4, 8, 8>
        <<<grid, block>>>(A, B, C, M, N, K);
}
```


details：

tileAB载入SMEM时，向量化数据的传输分两个阶段，STS.128过程一定16B对齐，LDG.128过程是否对齐由K决定，若K为4的倍数（最优情况下是 BK=8 的倍数）则16B对齐；  

向量化数据实现合并访问；  

向量化时，用 make_float4() 而不是 *reinterpret_cast<float4*>(&)，因为取地址会强制将c_frag从寄存器溢出到local memory，  
寄存器无内存地址空间中的地址，只有寄存器编号，故不应该对寄存器中的数据取地址，否则编译器会强行将数据溢出至局部内存，在类似的寄存器直接写入HBM的情景中会产生额外开销；  




规范了对frag数组广播过程解读的注释

线程重排情况
GLM   ------------》   SMEM   -------》   REG
tileA 128\*2 tileB 8*32     4\*8warp tile

            《------------------
                4\*8warp tile  


## version5

bank c +  
benchmark

同一warp的32个线程均参与b_frag的写入，由于相邻线程读取的数据相隔8个float，会导致2way冲突，故现采用每个线程读取float4向量化数据，且相邻线程读取的数据连续。此时每个quarter-warp均读取连续的、正好填满一行bank的共128B数据。

当前各线程寄存器中数值不在正确位置，现进行以下调整：
0前 -> 0前
1前 -> 0后
2前 -> 1前
3前 -> 1后
4前 -> 2前
5前 -> 2后
6前 -> 3前
7前 -> 3后
...
...
12前 -> 6前
13前 -> 6后
14前 -> 7前
15前 -> 7后

先直接写入frag数组再原地调整 -》先将SMEM中数据写入临时float4类型的线程私有寄存器变量，再按照严格顺序进行8次洗牌广播


A 8way bc
1. padding？ 行距变成 9 floats 会破坏 STS.128 的 16B 对齐。
2. 转置

B 2way bc
连续线程读取连续数据，读取完毕后再调整至正确位置，最后广播




template <int BM = 128, int BN = 128, int BK = 8,
    int BLOCK_SIZE, int Wx, int Wy,
    int TM, int TN>
__global__ void sgemm_thread_tiling(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    int tid = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;

    // 寄存器
    float a_frag[TM];
    float b_frag[TN];
    float c_frag[TM][TN] = {0.0f};


    int warp_id = tid >> 5; // tid / 32
    int lane_id = tid & 31; // tid % 32

    constexpr int WARP_Y = Wy; // 4
    constexpr int WARP_X = Wx; // 8

    //寄存器读取SMEM时线程重排，线程块尺寸
    constexpr int RLDS_C_BLOCK_X = BN / TN;
    constexpr int RLDS_C_BLOCK_Y = BM / TM;

    // 每一列 LDS_C_BLOCK_Y / WARP_Y 个 warp tile
    constexpr int warp_tile_per_col = RLDS_C_BLOCK_Y / WARP_Y;
    // 每一行 LDS_C_BLOCK_X / WARP_X 个 warp tile
    constexpr int warp_tile_per_row = RLDS_C_BLOCK_X / WARP_X;

    // warp 在 Block 中的位置（4×2 排列）
    int warp_row = warp_id / warp_tile_per_row; // / 2   M 方向：0~3
    int warp_col = warp_id % warp_tile_per_row; // % 2   N 方向：0~1

    // lane 在 warp 内的位置（4×8 排列，行主序）
    int lane_row = lane_id / WARP_X; // M 方向：0~3
    int lane_col = lane_id % WARP_X; // N 方向：0~7

    //K-Loop
    for (int bk = 0; bk < K; bk += BK)
    {

        // 协作加载 A、B 到 Shared Memory
        load_tile_A<BM, BK, BLOCK_SIZE>(A, As, M, K, by * BM, bk, tid);
        load_tile_B<BN, BK, BLOCK_SIZE>(B, Bs, K, N, bx * BN, bk, tid);

        __syncthreads();

        // 外积累加
        for (int k = 0; k < BK; k++) {

            // 从 Shared Memory 加载到寄存器

            //1个线程读取1个float至a_frag
            //同一warp的32个线程发起1个内存事务，且数据严格连续对齐、无bank conflict
            a_frag[lane_col] = As[k][warp_row * WARP_Y * TM + lane_row * TM + lane_col];

            // 广播 a_frag
            // 每个线程遍历各自的a_frag，从同行对应线程的或自身的a_frag中获取数据，使得同行线程的a_frag均存有正确数据且一致
            #pragma unroll
            for (int i = 0; i < TM; i++) {

                a_frag[i] = __shfl_sync(0xffffffff, a_frag[i], i, 8);
            }


            // b_frag写入过程
            // 16个lane各读一个float4：相邻lane连续，无bank conflict
            float4 b_vec = make_float4(0.f, 0.f, 0.f, 0.f);
            if (lane_row < 2) {
                
                b_vec = CONST_FLOAT4(Bs[k][warp_col * WARP_X * TN + lane_id * 4]);
            }   

            // 全warp参与、无分歧：0~3来自lane 2*lane_col，4~7来自lane 2*lane_col+1
            // __shfl_sync()只支持32位或64位的基础标量类型，必须将float4类型数据拆开
            b_frag[0] = __shfl_sync(0xffffffff, b_vec.x, 2 * lane_col);
            b_frag[1] = __shfl_sync(0xffffffff, b_vec.y, 2 * lane_col);
            b_frag[2] = __shfl_sync(0xffffffff, b_vec.z, 2 * lane_col);
            b_frag[3] = __shfl_sync(0xffffffff, b_vec.w, 2 * lane_col);
            b_frag[4] = __shfl_sync(0xffffffff, b_vec.x, 2 * lane_col + 1);
            b_frag[5] = __shfl_sync(0xffffffff, b_vec.y, 2 * lane_col + 1);
            b_frag[6] = __shfl_sync(0xffffffff, b_vec.z, 2 * lane_col + 1);
            b_frag[7] = __shfl_sync(0xffffffff, b_vec.w, 2 * lane_col + 1);
            

            // 寄存器上做外积
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    c_frag[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }


    // 写回全局内存，使用 warp/lane 坐标，与 compute 一致
    // 一个线程负责一个数据间相邻的8*8部分
    int base_row = by * BM + warp_row * WARP_Y * TM + lane_row * TM;
    int base_col = bx * BN + warp_col * WARP_X * TN + lane_col * TN;
    for (int i = 0; i < TM; i++) {

        int r = base_row + i;
        if (r >= M) {
            //M不为8的倍数时，负责余下的不足8行数据的线程同样循环8次，但没有数据的几次循环会执行continue语句跳过
            continue;
        }

        if (base_col + TN <= N) {
            
            //2条STG.128指令
            
            float4 temp = make_float4(c_frag[i][0], c_frag[i][1], c_frag[i][2], c_frag[i][3]);
            FLOAT4(C[r * N + base_col]) = temp;

            temp = make_float4(c_frag[i][4], c_frag[i][5], c_frag[i][6], c_frag[i][7]);
            FLOAT4(C[r * N + base_col + 4]) = temp;

        }

        else {

            // N不为8的倍数时，余下的不足8个的float数据逐个传输
            for (int j = 0; j < N - base_col; ++j) {
                C[r * N + base_col + j] = c_frag[i][j];
            }

        }

    }
}



总结：
相邻线程应尽量访问相距较短或相邻的数据

对于彻底消除divergence的优化，在结合了后续bc的情况下，又该如何重新考虑？
访问尽量连续的数据 比 让同一warp的每个线程都参与 更重要



？？？
写回时不经过SMEM？


TM TN x y 在哪一维度列不等式？
block thread warp register tiling 原理、本质




数据地址的对齐是硬件自动完成的吗？

tensor core















参考资料：  
https://docs.nvidia.com/cuda/cuda-programming-guide/contents.html  
https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-ampere-architecture-whitepaper.pdf  
https://docs.nvidia.com/cuda/cuda-c-programming-guide/contents.html  
https://forums.developer.nvidia.com/t/how-to-understand-the-bank-conflict-of-shared-mem/260900
https://caomaolufei.github.io/AIInfraGuide/
https://zhuanlan.zhihu.com/p/584236348
