#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda.h>
#include <memory>


struct Timer
{
    cudaEvent_t started;
    cudaEvent_t stopped;

    double usecs() const;
};


typedef std::shared_ptr<Timer> TimerPtr;


TimerPtr createTimer();

#endif
