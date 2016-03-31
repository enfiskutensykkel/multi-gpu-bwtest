#ifndef __BUFFER_H__
#define __BUFFER_H__

#include <memory>


typedef std::shared_ptr<void> BufferPtr;


BufferPtr createDeviceBuffer(int device, size_t length);


BufferPtr createHostBuffer(size_t length, unsigned int flags);

#endif
