#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include "../include/cell.h"
#include "../include/config.h"
#include "../include/particle.h"
#include "../include/math_utils.h"

Particle P[MAX_PARTICLES];
Cell samples[NX][NY];
int cellCount[NX][NY];
int cellList[NX][NY][MAX_PARTICLES];

// counters
int sampleSteps = 0;
int NP = 0;
long long totalCollisions = 0;

// derived quantities
double dx, dy;
double cellVolume;
double moleculeMass;
double weight;
double sigmaHS;



void setup(void) {
    memset(samples, 0, sizeof(samples));

    dx = Lx / NX;
    dy = Ly / NY;
    cellVolume = dx * dy * Lz;

    moleculeMass = molarMass / NA;

    weight = n0 * cellVolume / particlesPerCellTarget;

    NP = NX * NY * particlesPerCellTarget;

    if (NP > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles for MAX_PARTICLES\n");
        exit(1);
    }

    sigmaHS = M_PI * diameter * diameter;
}

void random_unit_vector(double *nx, double *ny, double *nz) {
    double mu = 2.0 * randu() - 1.0;      /* cos(theta) uniformly in [-1,1] */
    double phi = 2.0 * M_PI * randu();
    double s = sqrt(1.0 - mu * mu);

    *nx = s * cos(phi);
    *ny = s * sin(phi);
    *nz = mu;
}

void elastic_collision(Particle *a, Particle *b) {
    double vcx = 0.5 * (a->vx + b->vx);
    double vcy = 0.5 * (a->vy + b->vy);
    double vcz = 0.5 * (a->vz + b->vz);

    double crx = b->vx - a->vx;
    double cry = b->vy - a->vy;
    double crz = b->vz - a->vz;

    double cr = sqrt(crx * crx + cry * cry + crz * crz);

    double nx, ny, nz;
    random_unit_vector(&nx, &ny, &nz);

    double halfcr = 0.5 * cr;
    double rcx = halfcr * nx;
    double rcy = halfcr * ny;
    double rcz = halfcr * nz;

    a->vx = vcx - rcx;
    a->vy = vcy - rcy;
    a->vz = vcz - rcz;

    b->vx = vcx + rcx;
    b->vy = vcy + rcy;
    b->vz = vcz + rcz;
}

int collide_cell_primitive(int k, int l) {
    int Nc = cellCount[k][l];
    if (Nc < 2) {
        return 0;
    }

    int nColl = 0;

    for (int q1 = 0; q1 < Nc - 1; q1++) {
        int i = cellList[k][l][q1];

        for (int q2 = q1 + 1; q2 < Nc; q2++) {
            int j = cellList[k][l][q2];

            double dvx = P[j].vx - P[i].vx;
            double dvy = P[j].vy - P[i].vy;
            double dvz = P[j].vz - P[i].vz;
            double cr = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);

            double pColl = weight * sigmaHS * cr * dt / cellVolume;

            if (pColl > 1.0) {
                pColl = 1.0;
            }

            if (randu() < pColl) {
                elastic_collision(&P[i], &P[j]);
                nColl++;
            }
        }
    }

    return nColl;
}

void collide_all_cells(void) {
    long long stepCollisions = 0;

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            stepCollisions += collide_cell_primitive(k, l);
        }
    }

    totalCollisions += stepCollisions;
}

void initialize_particles(void) {
    double thermalStd = sqrt(KB * T0 / moleculeMass);

    for (int i = 0; i < NP; i++) {
        P[i].x = Lx * randu();
        P[i].y = Ly * randu();

        P[i].vx = randn(ux0, thermalStd);
        P[i].vy = randn(uy0, thermalStd);
        P[i].vz = randn(uz0, thermalStd);
    }
}

void move_particles(void) {
    for (int i = 0; i < NP; i++) {
        P[i].x += dt * P[i].vx;
        P[i].y += dt * P[i].vy;
    }
}

double rayleigh(double sigma) {
    double u = randu();
    return sigma * sqrt(-2.0 * log(u));
}

void apply_boundary_conditions(void) {
    for (int i = 0; i < NP; i++) {
        if (P[i].x < 0.0) {
            P[i].x = 0.0;
            P[i].vy = randn(0, sqrt(KB*Tleft/moleculeMass));
            P[i].vz = randn(0, sqrt(KB*Tleft/moleculeMass));
            P[i].vx = rayleigh(sqrt(KB*Tleft/moleculeMass));
        }
        else if (P[i].x >= Lx)  {
            P[i].x = Lx;
            P[i].vy = randn(0, sqrt(KB*Tright/moleculeMass));
            P[i].vz = randn(0, sqrt(KB*Tright/moleculeMass));
            P[i].vx = -rayleigh(sqrt(KB*Tright/moleculeMass));
        }

        else if (P[i].y < 0.0)  P[i].y += Ly;
        else if (P[i].y >= Ly)  P[i].y -= Ly;
    }
}

void index_particles(void) {
    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            cellCount[k][l] = 0;
        }
    }

    for (int i = 0; i < NP; i++) {
        int k = (int)(P[i].x / dx);
        int l = (int)(P[i].y / dy);

        if (k < 0) k = 0;
        if (k >= NX) k = NX - 1;
        if (l < 0) l = 0;
        if (l >= NY) l = NY - 1;

        int n = cellCount[k][l];
        cellList[k][l][n] = i;
        cellCount[k][l]++;
    }
}

void print_global_diagnostics(int step) {
    double sumVx = 0.0, sumVy = 0.0, sumVz = 0.0;

    for (int i = 0; i < NP; i++) {
        sumVx += P[i].vx;
        sumVy += P[i].vy;
        sumVz += P[i].vz;
    }

    double ux = sumVx / NP;
    double uy = sumVy / NP;
    double uz = sumVz / NP;

    double sumC2 = 0.0;
    for (int i = 0; i < NP; i++) {
        double cx = P[i].vx - ux;
        double cy = P[i].vy - uy;
        double cz = P[i].vz - uz;
        sumC2 += cx * cx + cy * cy + cz * cz;
    }

    double T = moleculeMass * sumC2 / (3.0 * KB * NP);

    printf("step=%d NP=%d mean_u=(%.6e, %.6e, %.6e) T=%.6e\n",
           step, NP, ux, uy, uz, T);
}

void accumulate_sampling(void) {
    sampleSteps++;

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            int Nc = cellCount[k][l];

            samples[k][l].countNP += Nc;

            for (int q = 0; q < Nc; q++) {
                int i = cellList[k][l][q];

                double vx = P[i].vx;
                double vy = P[i].vy;
                double vz = P[i].vz;

                samples[k][l].countVx += vx;
                samples[k][l].countVy += vy;
                samples[k][l].countVz += vz;
                samples[k][l].countV2 += vx * vx + vy * vy + vz * vz;
            }
        }
    }
}

void write_averaged_macros(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open output file\n");
        exit(1);
    }

    fprintf(fp, "# x y n ux uy uz T avg_count\n");

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            double xc = (k + 0.5) * dx;
            double yc = (l + 0.5) * dy;

            double avgNP = samples[k][l].countNP / sampleSteps;

            if (avgNP <= 0.0) {
                fprintf(fp, "%e %e %e %e %e %e %e %e\n",
                        xc, yc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                continue;
            }

            double ux = samples[k][l].countVx / samples[k][l].countNP;
            double uy = samples[k][l].countVy / samples[k][l].countNP;
            double uz = samples[k][l].countVz / samples[k][l].countNP;

            double meanV2 = samples[k][l].countV2 / samples[k][l].countNP;
            double meanU2 = ux * ux + uy * uy + uz * uz;

            double n = weight * avgNP / cellVolume;
            double T = moleculeMass * (meanV2 - meanU2) / (3.0 * KB);

            fprintf(fp, "%e %e %e %e %e %e %e %e\n",
                    xc, yc, n, ux, uy, uz, T, avgNP);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main(void) {
    srand((unsigned int)time(NULL));

    setup();
    initialize_particles();

    for (int step = 0; step <= nSteps; step++) {
        move_particles();
        apply_boundary_conditions();
        index_particles();
        collide_all_cells();

        if (step >= firstSampleStep && step % samplingPeriod == 0) {
            accumulate_sampling();
        }

        if (step % printPeriod == 0) {
            print_global_diagnostics(step);
            printf("totalCollisions = %lld\n", totalCollisions);
        }
    }

    write_averaged_macros("fields_avg.dat");

    return 0;
}