#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__

#include <cuda.h>
#include "devbuf.h"
#include "hostbuf.h"
#include "stream.h"
#include "event.h"


struct TransferSpec
{
    int             device;
    size_t          length;
    DeviceBufferPtr deviceBuffer;
    HostBufferPtr   hostBuffer;
    StreamPtr       stream;
    cudaMemcpyKind  direction;
    TimingDataPtr   events;
};


void runBandwidthTest(const std::vector<TransferSpec>& transferSpecifications);

#endif
