#ifndef OBJECT_H
#define OBJECT_H

typedef struct {
    float WingX;
    float WingY;
    float WingLength;
    double Tw;
} Wing;

typedef struct {
    float sphereCenterX;
    float sphereCenterY;
    float sphereCenterZ;
    float sphereRadius;
    double Tb;

    float sphereRadiusSquared; // derived
} Sphere;

#endif