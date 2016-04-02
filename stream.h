#ifndef __STREAM_HANDLING_H__
#define __STREAM_HANDLING_H__

#include <cuda.h>
#include <memory>
#include <map>


// How should streams be shared?
enum StreamSharingMode
{ 
    perTransfer,    // create a stream for every transfer
    perDevice,      // create a stream per device
    singleStream    // everyone use a single stream
};


// Make a smart pointer wrapper for cudaStream_t
typedef std::shared_ptr<cudaStream_t> StreamPtr;


// Helper class for simplifying stream sharing
class StreamManager
{
    public:
        // Create a StreamManager instance
        explicit StreamManager(StreamSharingMode streamMode);

        // Retrieve stream for device
        // This method respects the stream sharing mode set in the ctor
        StreamPtr retrieveStream(int device);

    private:
        StreamSharingMode mode;
        std::map<int, StreamPtr> streams;
};

#endif
