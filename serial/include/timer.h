#ifndef TIMER_H
#define TIMER_H

#include <time.h>

typedef struct {
    struct timespec start_time;
    struct timespec end_time;
    double elapsed_ms;
} Timer;

void timer_start(Timer *timer);
void timer_end(Timer *timer);
void timer_print(Timer *timer, const char *taskTitle);

#endif // TIMER_H