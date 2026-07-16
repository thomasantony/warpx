#include <cuda_runtime.h>

#include <cstdlib>

namespace
{
    __global__ void writeMappedHostMemory (int* value)
    {
        *value = 0x51a7;
    }
}

int main (int argc, char** argv)
{
    int const device = argc > 1 ? std::atoi(argv[1]) : 0;
    if (cudaSetDevice(device) != cudaSuccess) { return 1; }

    int can_map = 0;
    if (cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, device) != cudaSuccess ||
        !can_map) {
        return 2;
    }

    int* host_value = nullptr;
    if (cudaHostAlloc(&host_value, sizeof(int), cudaHostAllocMapped) != cudaSuccess) { return 3; }

    int* device_value = nullptr;
    if (cudaHostGetDevicePointer(&device_value, host_value, 0) != cudaSuccess) {
        cudaFreeHost(host_value);
        return 4;
    }

    *host_value = 0;
    writeMappedHostMemory<<<1, 1>>>(device_value);
    bool const success = cudaDeviceSynchronize() == cudaSuccess && *host_value == 0x51a7;
    cudaFreeHost(host_value);
    return success ? 0 : 5;
}
