#ifndef SECOND_H
#define SECOND_H

#include <sys/time.h>

inline double second (void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

#endif
