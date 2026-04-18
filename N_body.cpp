#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>


// EPS2はゼロ除算回避用の微小量
float EPS2 = 


// __device__ でGPU上で実行することを命令
// float3は3次元ベクトルの構造体
// bodyBodyInteraction()は粒子iと粒子jの間の重力から粒子iの加速度aiを更新する関数
// biは粒子iの位置+質量
// bjは粒子jの位置+質量
// aiは粒子iの現在の加速度
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai){
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
__device__ float3 tile_calculation(float4 myPosition, float3 accel){
    int i;

    // externはkernel起動時にサイズが決まる動的共有メモリ
    // __shared__ は同じブロック内のスレッドで共有される高速メモリ
    // shPositionは処理中のタイルの全粒子のキャッシュ
    // bodyBodyInteraction()は
    extern __shared__ float4[] shPosition;
    for (i=0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, soPosition[i], accel);
    }

    return accel;
}


// __global__でCPUから呼ばれるGPUカーネル
// globalXは粒子の位置
// globalAは加速度
__global__ void calculate_forces(float4 *globalX, float4 *globalA){
    extern __shared__ float4[] shPosition;

    // shPositionは処理中のタイルの全粒子のキャッシュ
    float4 myPosition;

    int i, tile;
    float3 acc = {0.0f, 0.0f, 0.0f};

    // gtidはグローバルスレッドID
    // blockIdxはブロック番号
    // blockDimは1ブロック中にあるスレッド数
    // threadIdxはスレッド番号
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    // myPositionはスレッドが担当する粒子の位置
    myPosition = globalX[gtid];

    


}