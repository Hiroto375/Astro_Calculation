#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>


// EPS2はゼロ除算回避用の微小量
constexpr float EPS2 = 1e-6f;

// CUDAエラーチェック用
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)


// __device__ でGPU上で実行することを命令
// float3は3次元ベクトルの構造体
// bodyBodyInteraction()は粒子iと粒子jの間の重力から粒子iの加速度aiを更新する関数
// biは粒子iの位置+質量
// bjは粒子jの位置+質量
// aiは粒子iの現在の加速度
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    // rは粒子iと粒子jの相対位置
    float3 r;

    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqrは距離の2乗
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;

    // invDistCubeは距離の逆3乗
    float invDistCube = 1.0f / sqrtf(distSqr * distSqr * distSqr);

    float s = bj.w * invDistCube;

    // 加速度aiを更新
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;

}

// myPositionは更新する粒子の位置+質量
// accelは更新として与えるべき加速度
// tile_calculation()は粒子の加速度を更新する関数
__device__ float3 tile_calculation(float4 myPosition, float3 accel) {
    int i;

    // externはkernel起動時にサイズが決まる動的共有メモリ
    // __shared__ は同じブロック内のスレッドで共有される高速メモリ
    // shPositionは処理中のタイルの粒子全ての情報を並べたキャッシュ
    // bodyBodyInteraction()は粒子iと粒子jの間の重力から粒子iの加速度aiを更新する関数
    extern __shared__ float4 shPosition[];
    for (i=0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }

    return accel;
}


// __global__でCPUから呼ばれるGPUカーネル
// calculate_forces()は
// globalXは粒子の位置
// globalAは加速度
__global__ void calculate_forces(const float4 *globalX, float4 *globalA, int N) {
    // shPositionは処理中のタイルの全粒子のキャッシュ
    extern __shared__ float4 shPosition[];

    float4 myPosition;

    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};

    // gtidはグローバルスレッドID
    // blockIdxはブロック番号
    // blockDimは1ブロック中にあるスレッド数
    // threadIdxはスレッド番号
    // 以上の変数はCUDAが最初から用意してくれる
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // 範囲外アクセスを回避する
    if (gtid >= N) return;

    // myPositionはスレッドが担当する粒子の位置
    myPosition = globalX[gtid];

    // iは粒子の番号
    // tileは同時に計算する粒子の集合
    for (i = 0, tile = 0; i < N; i += blockDim.x, tile++) {

        // idxは担当する粒子が含まれる行とぶつかる、計算中のタイルにある粒子の位置
        int idx = tile * blockDim.x + threadIdx.x;

        //範囲外アクセスを回避する
        if (idx < N) {
            shPosition[threadIdx.x] = globalX[idx];
        }
        else {
            shPosition[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        // __syncthreads()は同一ブロック中の全スレッドが上の計算を終えるまで待つ関数
        __syncthreads();

        // 担当する粒子の加速度を計算タイル内の全ての粒子の情報を使って更新
        acc = tile_calculation(myPosition, acc);

        __syncthreads();
    }

        // 担当する粒子の加速度を保存
        float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
        globalA[gtid] = acc4;
}

int main(){
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock ;

    std::vector<float4> hX(N);
    std::vector<float4> hA(N);

    // 初期条件
    for (int i = 0; i< N; i++) {
        hX[i].x = 0.001f * i;
        hX[i].y = 0.001f * i;
        hX[i].z = 0.001f * i;
        hX[i].w = 1.0f;
    }

    float4* dX = nullptr;
    float4* dA = nullptr;

    // GPU上のメモリを確保
    CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float4)));

    // CPUのデータをGPUにコピー
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

    // sharedMemSizeは1ブロックあたりの共有メモリのバイト数
    size_t sharedMemSize = threadsPerBlock * sizeof(float4);
    calculate_forces<<<blocks, threadsPerBlock, sharedMemSize>>>(dX, dA, N);

    // GPUでエラーが出ていないか確認
    CUDA_CHECK(cudaGetLastError());

    // GPUの計算が終わるまで待つ
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPUの計算結果をCPUに持ってくる
    CUDA_CHECK(cudaMemcpy(hA.data(), dA, N * sizeof(float4), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++) {
        std::cout << "Particle" << i << ": " << hA[i].x << ", " << hA[i].y << ", " << hA[i].z << "\n";
    }

    // GPUのメモリを解放
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dA));

    return 0;
}