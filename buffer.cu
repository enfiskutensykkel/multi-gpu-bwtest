#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "buffer.h"


static void deleteHostBuffer(void* buffer)
{
    cudaFreeHost(buffer);
}


static void deleteDeviceBuffer(void* buffer)
{
    cudaFree(buffer);
}


BufferPtr createHostBuffer(size_t length, unsigned int flags)
{
    void* buffer;

    cudaError_t err = cudaHostAlloc(&buffer, length, flags);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return BufferPtr(buffer, &deleteHostBuffer);
}


BufferPtr createDeviceBuffer(int device, size_t length)
{
    cudaError_t err;
    void* buffer;

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaMalloc(&buffer, length);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return BufferPtr(buffer, &deleteDeviceBuffer);
}
