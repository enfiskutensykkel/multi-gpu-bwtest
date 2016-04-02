#include <cuda.h>
#include <exception>
#include <stdexcept>
#include <vector>
#include "device.h"

using std::runtime_error;
using std::vector;


// Cache number of devices
static int deviceCount = -1;


// Cache device properties
static vector<cudaDeviceProp> deviceProperties;


static void loadDeviceData()
{
    cudaError_t  err;

    err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        throw runtime_error(cudaGetErrorString(err));
    }

    deviceProperties.reserve(deviceCount);
    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp prop;

        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        deviceProperties.push_back(prop);
    }
}


bool isDeviceValid(int device)
{
    if (deviceCount < 0)
    {
        loadDeviceData();
    }

    if (device < 0 || device >= deviceCount)
    {
        return false;
    }

    const cudaDeviceProp& prop = deviceProperties[device];
    if (prop.computeMode == cudaComputeModeProhibited)
    {
        return false;
    }

    return true;
}


int countDevices()
{
    if (deviceCount < 0)
    {
        loadDeviceData();
    }

    return deviceCount;
}


void loadDeviceProperties(int device, cudaDeviceProp& prop)
{
    if (deviceCount < 0)
    {
        loadDeviceData();
    }

    prop = deviceProperties[device];
}
