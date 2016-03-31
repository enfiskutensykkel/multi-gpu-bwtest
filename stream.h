#ifndef __STREAM_HANDLING_H__
#define __STREAM_HANDLING_H__

#include <cuda.h>
#include <memory>
#include <map>


enum StreamSharingMode
{ 
    perTransfer,    // create a stream for every transfer
    perDevice,      // create a stream per device
    singleStream    // everyone use a single stream
};


typedef std::shared_ptr<cudaStream_t> StreamPtr;


class StreamManager
{
    public:
        explicit StreamManager(StreamSharingMode streamMode);

        StreamPtr retrieveStream(int device);

    private:
        StreamSharingMode mode;
        std::map<int, StreamPtr> streams;
};

#endif
