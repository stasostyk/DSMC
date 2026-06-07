#ifndef DSMC_CONFIG_H
#define DSMC_CONFIG_H

#define MAX_PARTICLES 500000
#define MAX_PARTICLES_PER_CELL 2000
#define NX 10
#define NY 10
#define NZ 10

typedef struct {
    double KB;
    double NA;

    // size of domain (m)
    double Lx;
    double Ly;
    double Lz;
    double DL;

    // gas properties
    double molarMass;

    // simulation loop parameters
    double dt;
    int nSteps;
    int printPeriod;
    int particlesPerCellTarget;
    int firstSampleStep;
    int samplingPeriod;

    #ifdef WING_CASE
        // wing
        double WingX;
        double WingY;
        double WingLength;
        double Tw;

        // stream
        double MaFree;
        double PFree;
        double TFree;
        double angleOfAttack;
    #elif defined(SPHERE_CASE)
        // sphere 
        double sphereCenterX;
        double sphereCenterY;
        double sphereCenterZ;
        double sphereRadius;
        double Tb;

        // stream
        double MaFree;
        double PFree;
        double TFree;
    #endif

    // VHS model
    double omega;
    double dRef;

    // derived parameters
    double moleculeMass;

    // derived, used in collider
    double sigmaRef;
    double CrRef;
} Config;

void config_setup(Config *config);

#endif //DSMC_CONFIG_H
