#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace
{
constexpr std::size_t allocation_sizes[] = {
    4U * 1024U,
    4U * 1024U * 1024U,
    8U * 1024U * 1024U,
    64U * 1024U * 1024U,
    256U * 1024U * 1024U,
};

bool check_cuda(cudaError_t const error, char const* const operation)
{
    if (error == cudaSuccess) {
        return true;
    }
    std::fprintf(stderr, "FAIL: %s: %s (%d)\n", operation, cudaGetErrorString(error), error);
    return false;
}

__global__ void write_pattern(std::uint32_t* const values, std::size_t const count,
                              std::uint32_t const seed)
{
    auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = seed ^ static_cast<std::uint32_t>(index * 2654435761U);
    }
}

bool verify_pattern(std::uint32_t const* const values, std::size_t const count,
                    std::uint32_t const seed)
{
    for (std::size_t i = 0; i < count; ++i) {
        auto const expected = seed ^ static_cast<std::uint32_t>(i * 2654435761U);
        if (values[i] != expected) {
            std::fprintf(stderr, "FAIL: mismatch at %zu: got 0x%08x, expected 0x%08x\n",
                         i, values[i], expected);
            return false;
        }
    }
    return true;
}

bool run_kernel_and_verify(std::uint32_t* const device_pointer,
                           std::uint32_t* const host_pointer, std::size_t const bytes,
                           char const* const label)
{
    auto const n_values = bytes / sizeof(std::uint32_t);
    constexpr std::uint32_t seed = 0x6d5a56a9U;
    constexpr int threads = 256;
    auto const blocks = static_cast<int>((n_values + threads - 1) / threads);
    write_pattern<<<blocks, threads>>>(device_pointer, n_values, seed);
    if (!check_cuda(cudaGetLastError(), "kernel launch") ||
        !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        return false;
    }
    if (!verify_pattern(host_pointer, n_values, seed)) {
        return false;
    }
    std::printf("PASS: %s GPU-write/CPU-read (%zu bytes)\n", label, bytes);
    return true;
}

__global__ void write_sparse_offsets(std::uint8_t* const allocation, std::size_t const bytes)
{
    constexpr std::uint8_t values[] = {0x11U, 0x22U, 0x33U, 0x44U, 0x55U};
    std::size_t const offsets[] = {0U, 4096U, 8U * 1024U * 1024U,
                                   bytes / 2U, bytes - 1U};
    if (threadIdx.x < 5) {
        allocation[offsets[threadIdx.x]] = values[threadIdx.x];
    }
}

bool run_pool_offset_test(std::uint8_t* const device_pointer,
                          std::uint8_t* const host_pointer, std::size_t const bytes,
                          char const* const label)
{
    constexpr std::uint8_t expected[] = {0x11U, 0x22U, 0x33U, 0x44U, 0x55U};
    std::size_t const offsets[] = {0U, 4096U, 8U * 1024U * 1024U,
                                   bytes / 2U, bytes - 1U};
    std::memset(host_pointer, 0, bytes);
    write_sparse_offsets<<<1, 5>>>(device_pointer, bytes);
    if (!check_cuda(cudaGetLastError(), "pool-offset kernel launch") ||
        !check_cuda(cudaDeviceSynchronize(), "pool-offset cudaDeviceSynchronize")) {
        return false;
    }
    for (int i = 0; i < 5; ++i) {
        if (host_pointer[offsets[i]] != expected[i]) {
            std::fprintf(stderr, "FAIL: %s pool offset %zu got 0x%02x, expected 0x%02x\n",
                         label, offsets[i], host_pointer[offsets[i]], expected[i]);
            return false;
        }
    }
    std::printf("PASS: %s sparse pooled/suballocation writes (%zu bytes)\n", label, bytes);
    return true;
}

int test_managed()
{
    for (auto const bytes : allocation_sizes) {
        std::uint32_t* pointer = nullptr;
        if (!check_cuda(cudaMallocManaged(&pointer, bytes), "cudaMallocManaged")) {
            return 1;
        }

        cudaPointerAttributes attributes{};
        if (!check_cuda(cudaPointerGetAttributes(&attributes, pointer),
                        "cudaPointerGetAttributes(managed)")) {
            cudaFree(pointer);
            return 1;
        }
        std::printf("managed pointer (%zu bytes): type=%d device=%d devicePointer=%p "
                    "hostPointer=%p\n", bytes, static_cast<int>(attributes.type),
                    attributes.device, attributes.devicePointer, attributes.hostPointer);
        if (attributes.type != cudaMemoryTypeManaged) {
            std::fprintf(stderr, "FAIL: managed allocation reported pointer type %d, expected %d\n",
                         static_cast<int>(attributes.type),
                         static_cast<int>(cudaMemoryTypeManaged));
            cudaFree(pointer);
            return 1;
        }

        std::memset(pointer, 0, bytes);
        auto const passed = run_kernel_and_verify(pointer, pointer, bytes, "cudaMallocManaged");
        auto const free_ok = check_cuda(cudaFree(pointer), "cudaFree(managed)");
        if (!passed || !free_ok) {
            return 1;
        }
    }

    constexpr std::size_t pool_bytes = 64U * 1024U * 1024U;
    std::uint8_t* pool = nullptr;
    if (!check_cuda(cudaMallocManaged(&pool, pool_bytes), "cudaMallocManaged(pool)")) {
        return 1;
    }
    auto const pool_passed = run_pool_offset_test(pool, pool, pool_bytes, "cudaMallocManaged");
    auto const pool_free_ok = check_cuda(cudaFree(pool), "cudaFree(managed pool)");
    return pool_passed && pool_free_ok ? 0 : 1;
}

int test_mapped()
{
    for (auto const bytes : allocation_sizes) {
        std::uint32_t* host_pointer = nullptr;
        if (!check_cuda(cudaHostAlloc(&host_pointer, bytes, cudaHostAllocMapped),
                        "cudaHostAllocMapped")) {
            return 1;
        }

        std::uint32_t* device_pointer = nullptr;
        if (!check_cuda(cudaHostGetDevicePointer(&device_pointer, host_pointer, 0),
                        "cudaHostGetDevicePointer")) {
            cudaFreeHost(host_pointer);
            return 1;
        }
        std::printf("mapped pointers (%zu bytes): host=%p device=%p\n",
                    bytes, host_pointer, device_pointer);
        std::memset(host_pointer, 0, bytes);
        auto const passed = run_kernel_and_verify(device_pointer, host_pointer, bytes,
                                                  "cudaHostAllocMapped");
        auto const free_ok = check_cuda(cudaFreeHost(host_pointer), "cudaFreeHost(mapped)");
        if (!passed || !free_ok) {
            return 1;
        }
    }

    constexpr std::size_t pool_bytes = 64U * 1024U * 1024U;
    std::uint8_t* host_pool = nullptr;
    if (!check_cuda(cudaHostAlloc(&host_pool, pool_bytes, cudaHostAllocMapped),
                    "cudaHostAllocMapped(pool)")) {
        return 1;
    }
    std::uint8_t* device_pool = nullptr;
    if (!check_cuda(cudaHostGetDevicePointer(&device_pool, host_pool, 0),
                    "cudaHostGetDevicePointer(pool)")) {
        cudaFreeHost(host_pool);
        return 1;
    }
    auto const pool_passed = run_pool_offset_test(device_pool, host_pool, pool_bytes,
                                                  "cudaHostAllocMapped");
    auto const pool_free_ok = check_cuda(cudaFreeHost(host_pool), "cudaFreeHost(mapped pool)");
    return pool_passed && pool_free_ok ? 0 : 1;
}

int test_managed_size(std::size_t const bytes)
{
    std::uint32_t* pointer = nullptr;
    if (!check_cuda(cudaMallocManaged(&pointer, bytes), "cudaMallocManaged(size)")) {
        return 1;
    }
    std::memset(pointer, 0, bytes);
    auto const passed = run_kernel_and_verify(pointer, pointer, bytes, "cudaMallocManaged(size)");
    auto const free_ok = check_cuda(cudaFree(pointer), "cudaFree(managed size)");
    return passed && free_ok ? 0 : 1;
}

int test_mapped_direct_size(std::size_t const bytes)
{
    std::uint32_t* pointer = nullptr;
    if (!check_cuda(cudaHostAlloc(&pointer, bytes, cudaHostAllocMapped),
                    "cudaHostAllocMapped(direct size)")) {
        return 1;
    }
    std::memset(pointer, 0, bytes);
    auto const passed = run_kernel_and_verify(pointer, pointer, bytes,
                                              "cudaHostAllocMapped(direct size)");
    auto const free_ok = check_cuda(cudaFreeHost(pointer), "cudaFreeHost(direct size)");
    return passed && free_ok ? 0 : 1;
}
} // namespace

int main(int argc, char** argv)
{
    int device = 0;
    cudaDeviceProp properties{};
    if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice") ||
        !check_cuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties")) {
        return 1;
    }
    std::printf("device: %s, compute capability %d.%d\n", properties.name, properties.major,
                properties.minor);
    std::printf("capabilities: managedMemory=%d concurrentManagedAccess=%d "
                "pageableMemoryAccess=%d hostNativeAtomicSupported=%d\n",
                properties.managedMemory, properties.concurrentManagedAccess,
                properties.pageableMemoryAccess, properties.hostNativeAtomicSupported);

    if (argc == 3 && (std::strcmp(argv[1], "managed-size") == 0 ||
                      std::strcmp(argv[1], "mapped-direct-size") == 0)) {
        auto const mib = std::strtoull(argv[2], nullptr, 10);
        auto const bytes = static_cast<std::size_t>(mib) * 1024U * 1024U;
        return std::strcmp(argv[1], "managed-size") == 0
                   ? test_managed_size(bytes)
                   : test_mapped_direct_size(bytes);
    }
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: %s managed|mapped|managed-size MiB|mapped-direct-size MiB\n",
                     argv[0]);
        return 2;
    }
    if (std::strcmp(argv[1], "managed") == 0) {
        return test_managed();
    }
    if (std::strcmp(argv[1], "mapped") == 0) {
        return test_mapped();
    }
    std::fprintf(stderr, "unknown test: %s\n", argv[1]);
    return 2;
}
