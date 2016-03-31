#include <cuda.h>
#include <memory>
#include <exception>
#include <stdexcept>
#include "timer.h"


static void deleteTimer(Timer* timer)
{
    cudaEventDestroy(timer->started);
    cudaEventDestroy(timer->stopped);
    delete timer;
}


TimerPtr createTimer()
{
    cudaError_t err;

    Timer* timer = new Timer;

    err = cudaEventCreate(&timer->started);
    if (err != cudaSuccess)
    {
        delete timer;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    err = cudaEventCreate(&timer->stopped);
    if (err != cudaSuccess)
    {
        cudaEventDestroy(timer->started);
        delete timer;
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return TimerPtr(timer, &deleteTimer);
}


double Timer::usecs() const
{
    float milliseconds = .0f;

    cudaError_t err = cudaEventElapsedTime(&milliseconds, started, stopped);
    if (err != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return (double) milliseconds * 1000;
}

