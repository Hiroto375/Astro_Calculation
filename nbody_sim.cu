#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

constexpr float EPS2 = 1e-6f;
constexpr float G = 1.0f;

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)         \
                      << "\ncode: " << static_cast<int>(err)               \
                      << "\nat " << __FILE__ << ":" << __LINE__ << '\n';   \
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

    float s = G * bj.w * invDistCube;

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

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= N) return;

    float4 myPosition = globalX[gtid];
    float3 acc = {0.0f, 0.0f, 0.0f};

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

    globalA[gtid] = make_float4(acc.x, acc.y, acc.z, 0.0f);
}

__global__ void integrate_euler(float4* x, float4* v, const float4* a, float dt, int N) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid >= N) return;

    v[gtid].x += a[gtid].x * dt;
    v[gtid].y += a[gtid].y * dt;
    v[gtid].z += a[gtid].z * dt;

    x[gtid].x += v[gtid].x * dt;
    x[gtid].y += v[gtid].y * dt;
    x[gtid].z += v[gtid].z * dt;
}

void save_positions_csv(const std::string& filename, const std::vector<float4>& hX, int step) {
    std::ofstream ofs(filename, std::ios::app);
    if (!ofs) {
        std::cerr << "Failed to open " << filename << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < hX.size(); i++) {
        ofs << step << "," << i << ","
            << hX[i].x << "," << hX[i].y << "," << hX[i].z << "\n";
    }
}

int main() {
    std::cout << "program start" << std::endl;

    int deviceCount = 0;
    cudaError_t err0 = cudaGetDeviceCount(&deviceCount);
    std::cout << "cudaGetDeviceCount: " << cudaGetErrorString(err0)
              << ", count = " << deviceCount << std::endl;

    if (err0 != cudaSuccess || deviceCount == 0) {
        std::cerr << "No usable CUDA device found." << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;

    const int N = 256;
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    const float dt = 0.001f;
    const int numSteps = 2000;
    const int outputInterval = 20;

    std::vector<float4> hX(N);
    std::vector<float4> hV(N);
    std::vector<float4> hA(N);

    // 初期条件: 円盤っぽく並べる例
    for (int i = 0; i < N; i++) {
        float theta = 2.0f * 3.1415926535f * i / N;
        float r = 1.0f + 0.1f * (i % 10);

        hX[i].x = r * std::cos(theta);
        hX[i].y = r * std::sin(theta);
        hX[i].z = 0.1f * std::sin(3.0f * theta);
        hX[i].w = 1.0f;  // mass

        // 適当な初速度
        hV[i].x = -0.3f * std::sin(theta);
        hV[i].y =  0.3f * std::cos(theta);
        hV[i].z =  0.0f;
        hV[i].w =  0.0f;

        hA[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    float4* dX = nullptr;
    float4* dV = nullptr;
    float4* dA = nullptr;

    CUDA_CHECK(cudaMalloc(&dX, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dV, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float4)));

    CUDA_CHECK(cudaMemcpy(dX, hX.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

    const size_t sharedMemSize = threadsPerBlock * sizeof(float4);

    // CSV初期化
    {
        std::ofstream ofs("trajectory.csv");
        ofs << "step,id,x,y,z\n";
    }

    for (int step = 0; step < numSteps; step++) {
        calculate_forces<<<blocks, threadsPerBlock, sharedMemSize>>>(dX, dA, N);
        CUDA_CHECK(cudaGetLastError());

        integrate_euler<<<blocks, threadsPerBlock>>>(dX, dV, dA, dt, N);
        CUDA_CHECK(cudaGetLastError());

        if (step % outputInterval == 0) {
            CUDA_CHECK(cudaMemcpy(hX.data(), dX, N * sizeof(float4), cudaMemcpyDeviceToHost));
            save_positions_csv("trajectory.csv", hX, step);
            std::cout << "saved step " << step << std::endl;
        }
    }

    CUDA_CHECK(cudaMemcpy(hX.data(), dX, N * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hV.data(), dV, N * sizeof(float4), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hA.data(), dA, N * sizeof(float4), cudaMemcpyDeviceToHost));

    std::cout << "finished successfully\n";
    for (int i = 0; i < 5; i++) {
        std::cout << "Particle " << i << ": pos=("
                  << hX[i].x << ", " << hX[i].y << ", " << hX[i].z
                  << "), vel=("
                  << hV[i].x << ", " << hV[i].y << ", " << hV[i].z
                  << ")\n";
    }

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dA));

    return 0;
}