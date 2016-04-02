#ifndef __TIMER_H__
#define __TIMER_H__

#include <cuda.h>
#include <memory>


// Timer instance
struct Timer
{
    cudaEvent_t started;
    cudaEvent_t stopped;

    // Calculate the elapsed time between started and stopped
    double usecs() const;
};


// Timer instance pointer
typedef std::shared_ptr<Timer> TimerPtr;


// Helper function for creating a Timer instance
TimerPtr createTimer();

#endif
