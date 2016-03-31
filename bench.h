#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <cuda.h>
#include "buffer.h"
#include "stream.h"
#include "timer.h"


struct TransferSpec
{
    int             device;
    BufferPtr       deviceBuffer;
    BufferPtr       hostBuffer;
    size_t          length;
    cudaMemcpyKind  direction;
    StreamPtr       stream;
    TimerPtr        timer;
};


void runBandwidthTest(const std::vector<TransferSpec>& transferSpecifications);

#endif
