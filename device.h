#ifndef __DEVICE_H__
#define __DEVICE_H__

#include <cuda.h>

// Get number of CUDA devices on the system
int countDevices();

// Check if device is not prohibited
bool isDeviceValid(int device);

// Get device properties for a device
void loadDeviceProperties(int device, cudaDeviceProp& properties);

#endif
