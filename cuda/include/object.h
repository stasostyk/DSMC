#ifndef OBJECT_H
#define OBJECT_H

typedef struct {
    float WingX;
    float WingY;
    float WingLength;
    double Tw;
} Wing;

typedef struct {
    float ballCenterX;
    float ballCenterY;
    float ballCenterZ;
    float ballRadius;
    double Tb;

    float ballRadiusSquared; // derived
} Ball;

#endif