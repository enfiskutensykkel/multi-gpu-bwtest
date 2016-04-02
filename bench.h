#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <cuda.h>
#include "buffer.h"
#include "stream.h"
#include "timer.h"


// Specify a memory transfer
struct TransferSpec
{
    int             device;         // the CUDA device to transfer to or from
    BufferPtr       deviceBuffer;   // memory buffer on the device
    BufferPtr       hostBuffer;     // memory buffer on the host
    size_t          length;         // the transfer size
    cudaMemcpyKind  direction;      // the transfer direction (HtoD or DtoH)
    StreamPtr       stream;         // the stream to use for the stransfer
    TimerPtr        timer;          // timer data to record how long the transfer took
};


// Run a simple bandwidth test using cudaMemcpyAsync()
void runBandwidthTest(const std::vector<TransferSpec>& transferSpecifications);

#endif
