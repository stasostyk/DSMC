#include "../include/timer.h"
#include <stdio.h>

void timer_start(Timer *timer)
{
    clock_gettime(CLOCK_MONOTONIC, &timer->start_time);
}

void timer_end(Timer *timer)
{
    clock_gettime(CLOCK_MONOTONIC, &timer->end_time);
    
    // Calculate elapsed time in milliseconds
    long seconds = timer->end_time.tv_sec - timer->start_time.tv_sec;
    long nanoseconds = timer->end_time.tv_nsec - timer->start_time.tv_nsec;
    
    timer->elapsed_ms = seconds * 1000.0 + nanoseconds / 1e6;
}

void timer_print(Timer *timer, const char *taskTitle)
{
    printf("%s. Elapsed time: %.4f ms\n", taskTitle, timer->elapsed_ms);
}
