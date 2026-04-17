#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_PARTICLES 200000
#define NX 25
#define NY 25

const double KB = 1.380649e-23;      /* Boltzmann constant */
const double NA = 6.02214076e23;     /* Avogadro constant */

typedef struct {
    double x, y;
    double vx, vy, vz;
} Particle;

typedef struct {
    double countNP;   /* accumulated particle count */
    double countVx;   /* accumulated sum of vx */
    double countVy;   /* accumulated sum of vy */
    double countVz;   /* accumulated sum of vz */
    double countV2;   /* accumulated sum of vx^2 + vy^2 + vz^2 */
} CellSample;

CellSample samples[NX][NY];
int sampleSteps = 0;
const int firstSampleStep = 0;
const int sampleEvery = 1;

void reset_sampling(void) {
    sampleSteps = 0;

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            samples[k][l].countNP = 0.0;
            samples[k][l].countVx = 0.0;
            samples[k][l].countVy = 0.0;
            samples[k][l].countVz = 0.0;
            samples[k][l].countV2 = 0.0;
        }
    }
}

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;       /* effective thickness for 2D cells */

const double T0 = 300.0;     /* Kelvin */
const double n0 = 1.0e20;    /* number density, 1/m^3 */
const double molarMass = 0.028;  /* kg/mol, about nitrogen */

const double ux0 = 0.0;
const double uy0 = 0.0;
const double uz0 = 0.0;

const double dt = 1.0e-5;
const int nSteps = 200;
const int outputEvery = 20;

const int particlesPerCellTarget = 200;

int cellCount[NX][NY];
int cellList[NX][NY][MAX_PARTICLES];

// derived quantities:
double dx, dy;
double cellVolume;
double moleculeMass;
double weight;

// hard sphere assumption
const double diameter = 4.0e-10;
double sigmaHS;
long long totalCollisions = 0;

Particle P[MAX_PARTICLES];
int NP = 0;

double randu(void) {
    return (rand() + 1.0) / (RAND_MAX + 2.0);
}

double randn(double mean, double stddev) { // gaussian rng
    double u1 = randu();
    double u2 = randu();

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return mean + stddev * r * cos(theta);
}

void setup(void) {
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

double Tleft = 400.0;
double Tright = 200.0;

double normal(double mean, double stddev) {
    double u1 = randu();
    double u2 = randu();

    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * M_PI * u2;

    return mean + stddev * r * cos(theta);
}

double rayleigh(double sigma) {
    double u = randu();
    return sigma * sqrt(-2.0 * log(u));
}

void apply_periodic_bc(void) {
    for (int i = 0; i < NP; i++) {
        if (P[i].x < 0.0) {
            P[i].x = 0.0;
            P[i].vy = normal(0, sqrt(KB*Tleft/moleculeMass));
            P[i].vz = normal(0, sqrt(KB*Tleft/moleculeMass));
            P[i].vx = rayleigh(sqrt(KB*Tleft/moleculeMass));
        }
        if (P[i].x >= Lx)  {
            P[i].x = Lx;
            P[i].vy = normal(0, sqrt(KB*Tright/moleculeMass));
            P[i].vz = normal(0, sqrt(KB*Tright/moleculeMass));
            P[i].vx = -rayleigh(sqrt(KB*Tright/moleculeMass));
        }

        while (P[i].y < 0.0)  P[i].y += Ly;
        while (P[i].y >= Ly)  P[i].y -= Ly;
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

void sample_macros(int step) {
    char filename[64];
    snprintf(filename, sizeof(filename), "fields_%04d.dat", step);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open output file\n");
        exit(1);
    }

    fprintf(fp, "# x y n ux uy uz T count\n");

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            int Nc = cellCount[k][l];

            double xc = (k + 0.5) * dx;
            double yc = (l + 0.5) * dy;

            if (Nc == 0) {
                fprintf(fp, "%e %e %e %e %e %e %e %d\n",
                        xc, yc, 0.0, 0.0, 0.0, 0.0, 0.0, 0);
                continue;
            }

            double sumVx = 0.0, sumVy = 0.0, sumVz = 0.0;

            for (int q = 0; q < Nc; q++) {
                int i = cellList[k][l][q];
                sumVx += P[i].vx;
                sumVy += P[i].vy;
                sumVz += P[i].vz;
            }

            double ux = sumVx / Nc;
            double uy = sumVy / Nc;
            double uz = sumVz / Nc;

            double sumC2 = 0.0;
            for (int q = 0; q < Nc; q++) {
                int i = cellList[k][l][q];
                double cx = P[i].vx - ux;
                double cy = P[i].vy - uy;
                double cz = P[i].vz - uz;
                sumC2 += cx * cx + cy * cy + cz * cz;
            }

            double n = weight * Nc / cellVolume;
            double T = moleculeMass * sumC2 / (3.0 * KB * Nc);

            fprintf(fp, "%e %e %e %e %e %e %e %d\n",
                    xc, yc, n, ux, uy, uz, T, Nc);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
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
    reset_sampling();

    for (int step = 0; step <= nSteps; step++) {
        if (step > 0) {
            move_particles();
            apply_periodic_bc();
        }

        index_particles();
        collide_all_cells();

        if (step >= firstSampleStep && step % sampleEvery == 0) {
            accumulate_sampling();
        }

        if (step % outputEvery == 0) {
            print_global_diagnostics(step);
            printf("  totalCollisions = %lld\n", totalCollisions);
        }
    }

    write_averaged_macros("fields_avg.dat");

    return 0;
}