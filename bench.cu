#include <cuda.h>
#include <vector>
#include <exception>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdio>
#include "bench.h"
#include "buffer.h"
#include "stream.h"
#include "timer.h"

using std::vector;
using std::runtime_error;
using std::string;


static string bytesToUnit(size_t size)
{
    char buffer[1024];
    const char* units[] = { "B  ", "KiB", "MiB", "GiB", "TiB" };
    size_t i = 0, n = sizeof(units) / sizeof(units[0]);

    double csize = (double) size;

    while (i < (n - 1) && csize >= 1024.0)
    {
        csize /= 1024.0;
        ++i;
    }

    snprintf(buffer, sizeof(buffer), "%.2f %s", csize, units[i]);
    return string(buffer);
}


static string transferDirectionToString(cudaMemcpyKind direction)
{
    if (direction == cudaMemcpyHostToDevice)
    {
        return string("HtoD");
    }
    if (direction == cudaMemcpyDeviceToHost)
    {
        return string("DtoH");
    }

    return string("unknown");
}


static void timeTransfers(const vector<TransferSpec>& transferSpecs)
{
    cudaError_t err;

    for (const TransferSpec& spec : transferSpecs)
    {
        cudaStream_t stream = *spec.stream;

        const void* src = spec.direction == cudaMemcpyDeviceToHost ? spec.deviceBuffer.get() : spec.hostBuffer.get();
        void* dst = spec.direction == cudaMemcpyDeviceToHost ? spec.hostBuffer.get() : spec.deviceBuffer.get();

        err = cudaEventRecord(spec.timer->started, stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaMemcpyAsync(dst, src, spec.length, spec.direction, stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventRecord(spec.timer->stopped, stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }
    }
}


static void syncStreams(const vector<TransferSpec>& transferSpecs)
{
    cudaError_t err;

    for (const TransferSpec& spec : transferSpecs)
    {
        err = cudaStreamSynchronize(*spec.stream);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }
    }
}


void runBandwidthTest(const vector<TransferSpec>& transferSpecs)
{
    cudaError_t err;

    // Create timing events on the null stream
    TimerPtr globalTimer = createTimer();
    err = cudaEventRecord(globalTimer->started);
    if (err != cudaSuccess)
    {
        throw runtime_error(cudaGetErrorString(err));
    }

    // Execute transfers
    try
    {
        fprintf(stdout, "Executing transfers..........");
        fflush(stdout);
        timeTransfers(transferSpecs);
        fprintf(stdout, "DONE\n");
        fflush(stdout);
    }
    catch (const runtime_error& e)
    {
        fprintf(stdout, "FAIL\n");
        fflush(stdout);
        throw e;
    }

    // Synchronize all streams
    try
    {
        fprintf(stdout, "Synchronizing streams........");
        fflush(stdout);

        syncStreams(transferSpecs);

        err = cudaEventRecord(globalTimer->stopped);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        err = cudaEventSynchronize(globalTimer->stopped);
        if (err != cudaSuccess)
        {
            throw runtime_error(cudaGetErrorString(err));
        }

        fprintf(stdout, "DONE\n");
        fflush(stdout);
    } 
    catch (const runtime_error& e)
    {
        fprintf(stdout, "FAIL\n");
        fflush(stdout);
        throw e;
    }


    // FIXME: Warn about low compute-capability here instead?

    // Print results
    fprintf(stdout, "\n");
    fprintf(stdout, "=====================================================================================\n");
    fprintf(stdout, " %2s   %-15s   %13s   %-8s   %-12s   %-10s\n",
            "ID", "Device name", "Transfer size", "Direction", "Time elapsed", "Bandwidth");
    fprintf(stdout, "-------------------------------------------------------------------------------------\n");
    fflush(stdout);

    size_t totalSize = 0;
    double aggrElapsed = .0;
    double timedElapsed = globalTimer->usecs();

    for (const TransferSpec& res : transferSpecs)
    {
        double elapsed = res.timer->usecs();
        double bandwidth = (double) res.length / elapsed;

        totalSize += res.length;
        aggrElapsed += elapsed;

        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, res.device);
        if (err != cudaSuccess)
        {
            prop.name[0] = 'E';
            prop.name[1] = 'R';
            prop.name[2] = 'R';
            prop.name[3] = '!';
            prop.name[4] = '\0';
        }

        fprintf(stdout, " %2d   %-15s   %13s    %8s   %9.0f µs    %10.2f MiB/s \n",
                res.device, 
                prop.name, 
                bytesToUnit(res.length).c_str(), 
                transferDirectionToString(res.direction).c_str(),
                elapsed,
                bandwidth
               );
        fflush(stdout);
    }
    fprintf(stdout, "=====================================================================================\n");

    fprintf(stdout, "\n");
    fprintf(stdout, "Aggregated total time      : %12.0f µs\n", aggrElapsed);
    fprintf(stdout, "Aggregated total bandwidth : %12.2f MiB/s\n", (double) totalSize / aggrElapsed);
    fprintf(stdout, "Estimated elapsed time     : %12.0f µs\n", timedElapsed);
    fprintf(stdout, "Timed total bandwidth      : %12.2f MiB/s\n", (double) totalSize / timedElapsed);
    fprintf(stdout, "\n");
    fflush(stdout);
}
