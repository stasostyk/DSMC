#include <stdlib.h>
#include <math.h>
#include "../include/math_utils.h"

double randu() {
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

double randn(double mean, double stddev) {
    double u1 = randu();
    double u2 = randu();

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return mean + stddev * r * cos(theta);
}
