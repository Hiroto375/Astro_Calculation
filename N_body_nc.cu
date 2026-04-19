#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>

constexpr float EPS2 = 1e-6f;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
    float3 r;
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
    float distSixth = distSqr * distSqr * distSqr;
    float invDistCube = 1.0f / sqrtf(distSixth);
    float s = bj.w * invDistCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ float3 tile_calculation(float4 myPosition, float3 accel) {
    extern __shared__ float4 shPosition[];
    for (int i = 0; i < blockDim.x; i++) {
        accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
    }
    return accel;
}

__global__ void calculate_forces(const float4 *globalX, float4 *globalA, int N) {
    extern __shared__ float4 shPosition[];

    float3 acc = {0.0f, 0.0f, 0.0f};
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gtid >= N) return;

    float4 myPosition = globalX[gtid];

    for (int i = 0, tile = 0; i < N; i += blockDim.x, tile++) {
        int idx = tile * blockDim.x + threadIdx.x;

        if (idx < N) {
            shPosition[threadIdx.x] = globalX[idx];
        } else {
            shPosition[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }

        __syncthreads();
        acc = tile_calculation(myPosition, acc);
        __syncthreads();
    }

    float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
    globalA[gtid] = acc4;
}

int main() {
    const int N = 1024;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float4> hX(N);
    std::vector<float4> hA(N);

    for (int i = 0; i < N; i++) {
        hX[i].x = 0.001f * i;
        hX[i].y = 0.001f * i;
        hX[i].z = 0.001f * i;
        hX[i].w = 1.0f;
    }

    float4* dX = nullptr;
    float4* dA = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float4)));

    CUDA_CHECK(cudaMemcpy(dX, hX.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

    size_t sharedMemSize = threadsPerBlock * sizeof(float4);
    calculate_forces<<<blocks, threadsPerBlock, sharedMemSize>>>(dX, dA, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hA.data(), dA, N * sizeof(float4), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 5; i++) {
        std::cout << "Particle " << i << ": "
                  << hA[i].x << ", " << hA[i].y << ", " << hA[i].z << "\n";
    }

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dA));

    return 0;
}