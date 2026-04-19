#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "start" << std::endl;

    int count = -1;
    cudaError_t err = cudaGetDeviceCount(&count);

    std::cout << "err code = " << static_cast<int>(err) << std::endl;
    std::cout << "count = " << count << std::endl;

    if (err != cudaSuccess) {
        std::cout << "cudaGetErrorString = " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    std::cout << "prop err code = " << static_cast<int>(err) << std::endl;

    if (err == cudaSuccess) {
        std::cout << "GPU = " << prop.name << std::endl;
    }

    return 0;
}