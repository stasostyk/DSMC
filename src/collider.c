#include "../include/collider.h"
#include "../include/config.h"
#include "../include/math_utils.h"
#include "../include/particle.h"
#include "../include/simulation.h"
#include <math.h>

double sigmaRef;
double CrRef;

void collider_setup() {
    sigmaRef = M_PI * dRef * dRef;
    CrRef = sqrt(4.0 * KB * TFree / moleculeMass);
}

void elastic_collision(Particle *p1, Particle *p2, double Cr) {
    double N[3];
    random_isotropic_vector(N);
    Cr *= 0.5;
    double VC[3] = { 0.5 * ( p2->vx + p1->vx ), 0.5 * ( p2->vy + p1->vy ), 0.5 * ( p2->vz + p1->vz ) };
    double VCr[3] = { Cr * N[0], Cr * N[1], Cr * N[2] };
    p1->vx = VC[0] + VCr[0];
    p1->vy = VC[1] + VCr[1];
    p1->vz = VC[2] + VCr[2];
    p2->vx = VC[0] - VCr[0];
    p2->vy = VC[1] - VCr[1];
    p2->vz = VC[2] - VCr[2];
}

int noTimeCounterScheme(Particle *P, int NPC, int *IPC, double weight, double cellVolume) {
    if (NPC < 2) return 0;

    // estimate number of collisions
    double majorant = 9.0 * sigmaRef * sqrt(KB * TFree / moleculeMass);
    double estimatedCollidingPairs = 0.5 * NPC * (NPC - 1) * weight * majorant * dt / cellVolume;
    int expectedCollodingPairs = (int)estimatedCollidingPairs;
    if (randu() < estimatedCollidingPairs - expectedCollodingPairs) expectedCollodingPairs++;

    // monte carlo accept/reject pairs and collide
    int collisions = 0;
    int i, j; // two particles to collide

    for (int k = 0; k < expectedCollodingPairs; k++) {
        i = (int)(randu() * NPC);
        do {
            j = (int)(randu() * NPC);
        } while (j == i);

        i = IPC[i];
        j = IPC[j];

        double relativeVel[3] = { P[j].vx - P[i].vx, P[j].vy - P[i].vy, P[j].vz - P[i].vz };
        double relativeSpeed = sqrt(relativeVel[0] * relativeVel[0]
                                    + relativeVel[1] * relativeVel[1]
                                    + relativeVel[2] * relativeVel[2]);

        double collisionProb = sigmaRef * pow(CrRef / relativeSpeed, 2.0*omega - 1.0) * relativeSpeed / majorant;
        if (randu() < collisionProb) {
            elastic_collision( &P[i], &P[j], relativeSpeed );
            collisions++;
        }

    }
    return collisions;
}

int collide_particles(
    Particle *P, 
    int *cellCount,
    int *cellList, 
    double weight, 
    double cellVolume
) {
    int totalCollisions = 0;
    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                int collisions = noTimeCounterScheme(P,
                                                     cellCount[IDX_CELL(k, l, m)],
                                                     &cellList[IDX_LIST(k, l, m, 0)],
                                                     weight,
                                                     cellVolume);
                totalCollisions += collisions;
            }
        }
    }
    return totalCollisions;
}