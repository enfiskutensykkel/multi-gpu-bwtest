#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <memory>


// Smart pointer wrapper to make buffer clean up after themselves
typedef std::shared_ptr<void> BufferPtr;


// Create a device buffer using cudaMalloc() and wrap it in a smart pointer
BufferPtr createDeviceBuffer(int device, size_t length);


// Create a host buffer using cudaHostAlloc() and wrap it in a smart pointer
BufferPtr createHostBuffer(size_t length, unsigned int flags);

#endif
