#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "../include/simulation.h"
#include "../include/config.h"
#include "../include/math_utils.h"

void setup(Simulation *sim) {
    // 
    // Particle P[MAX_PARTICLES];
    // Cell samples[NX][NY][NZ];
    // int cellCount[NX][NY][NZ];
    // int cellList[NX][NY][NZ][MAX_PARTICLES_PER_CELL];


    sim->sampleSteps = 0;
    sim->NP = 0;
    sim->totalCollisions = 0;

    // TODO: most of the setup can be moved to constexpr??

    memset(sim->samples, 0, sizeof(sim->samples));

    sim->dx = Lx / NX;
    sim->dy = Ly / NY;
    sim->dz = Lz / NZ;
    sim->cellVolume = sim->dx * sim->dy * sim->dz;

    sim->NP = NX * NY * NZ * particlesPerCellTarget;

    if (sim->NP > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles for MAX_PARTICLES\n");
        exit(1);
    }

    #ifdef BALL_CASE
        double angleOfAttack = 0.;
    #endif
    sim->NFree = PFree / ( KB * TFree );
    sim->UFree = MaFree * sqrt ( ( 5.0 / 3.0 ) * KB * TFree / moleculeMass );
    sim->UxFree = sim->UFree * cos ( M_PI * angleOfAttack / 180.0 );
    sim->UyFree = - sim->UFree * sin ( M_PI * angleOfAttack / 180.0 );
    sim->UzFree = 0.0;

    sim->weight = sim->NFree * sim->cellVolume / particlesPerCellTarget;
}

void index_particles(Simulation *sim) {
    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                sim->cellCount[k][l][m] = 0;
            }
        }
    }

    for (int i = 0; i < sim->NP; i++) {
        int k = (int)(sim->P[i].x / sim->dx);
        int l = (int)(sim->P[i].y / sim->dy);
        int m = (int)(sim->P[i].z / sim->dz);

        if (k < 0) k = 0;
        if (k >= NX) k = NX - 1;
        if (l < 0) l = 0;
        if (l >= NY) l = NY - 1;
        if (m < 0) m = 0;
        if (m >= NZ) m = NZ - 1;

        int n = sim->cellCount[k][l][m];
        sim->cellList[k][l][m][n] = i;
        sim->cellCount[k][l][m]++;
    }
}


void generate_particles_in_rect(
    Simulation *sim,
    double x1, double x2,
    double y1, double y2,
    double z1, double z2,
    double Tgas,
    int moveFlag
) {
    double ngas = sim->NFree;
    double ux = sim->UxFree;
    double uy = sim->UyFree;
    double uz = sim->UzFree; 

    double V = (x2 - x1) * (y2 - y1) * (z2 - z1);
    double N_add_exp = ngas * V / sim->weight;
    int Nnew;
    if ( N_add_exp > 20.0 ) { // use uniform approximation if large, poisson if small
        Nnew = (int)(N_add_exp);
        if (randu() < N_add_exp - Nnew) Nnew++;
    } else {
        Nnew = (int) randp( N_add_exp );
    }

    if (sim->NP + Nnew > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles\n");
        exit(1);
    }

    for (int i = sim->NP; i < sim->NP + Nnew; i++) {
        sim->P[i].x = x1 + (x2 - x1) * randu();
        sim->P[i].y = y1 + (y2 - y1) * randu();
        sim->P[i].z = z1 + (z2 - z1) * randu();

        sim->P[i].vx = randn(ux, sqrt(KB * Tgas / moleculeMass)); // TODO don't calc sqrt every time
        sim->P[i].vy = randn(uy, sqrt(KB * Tgas / moleculeMass));
        sim->P[i].vz = randn(uz, sqrt(KB * Tgas / moleculeMass));

        if (moveFlag) {
            sim->P[i].x += dt * sim->P[i].vx;
            sim->P[i].y += dt * sim->P[i].vy;
            sim->P[i].z += dt * sim->P[i].vz;
        }
    }

    sim->NP += Nnew;
}

void initialize_particles(Simulation *sim) {
    sim->NP = 0;
    generate_particles_in_rect(sim, 0.0, Lx, 0.0, Ly, 0.0, Lz, TFree, 0);
}


void apply_boundary_conditions_free_stream(Simulation *sim) {
    generate_particles_in_rect(sim, -DL, 0.0, 0.0, Ly, 0.0, Lz, TFree, 1);
    generate_particles_in_rect(sim, Lx, Lx + DL, 0.0, Ly, 0.0, Lz, TFree, 1);
    generate_particles_in_rect(sim, 0.0, Lx, -DL, 0.0, 0.0, Lz, TFree, 1);
    generate_particles_in_rect(sim, 0.0, Lx, Ly, Ly + DL, 0.0, Lz, TFree, 1);
    generate_particles_in_rect(sim, 0.0, Lx, 0.0, Ly, -DL, 0.0, TFree, 1);
    generate_particles_in_rect(sim, 0.0, Lx, 0.0, Ly, Lz, Lz + DL, TFree, 1);

    for (int i = 0; i < sim->NP; i++) {
        if (sim->P[i].x < 0.0 || sim->P[i].x >= Lx 
            || sim->P[i].y < 0.0 || sim->P[i].y >= Ly 
            || sim->P[i].z < 0.0 || sim->P[i].z >= Lz
        ) {
            sim->P[i] = sim->P[sim->NP - 1];
            sim->NP--;
            i--;
        }
    }
}

void move_particles(Simulation *sim) {
    for (int i = 0; i < sim->NP; i++) {
        double X0 = sim->P[i].x;
        double Y0 = sim->P[i].y;
        double Z0 = sim->P[i].z;
        sim->P[i].x += dt * sim->P[i].vx;
        sim->P[i].y += dt * sim->P[i].vy;
        sim->P[i].z += dt * sim->P[i].vz;


        #ifdef WING_CASE
        if ( ( Y0 - WingY ) * ( sim->P[i].y - WingY ) < 0.0 ) {
            // Linear interpolation to point Y = WingY
            double Xw=( X0*(WingY-sim->P[i].y)+sim->P[i].x*(Y0-WingY))/(Y0-sim->P[i].y);
            double Zw=( Z0*(WingY-sim->P[i].y)+sim->P[i].z*(Y0-WingY))/(Y0-sim->P[i].y);
            if ( Zw < 0.3 || Zw > 0.7 ) continue; // wing only occupies 0.3 < z < 0.7
            if ( Xw > WingX && Xw < WingX + WingLength ) {
                // Molecule interacts with the wing during the time step
                // Linear interpolation of the time of scattering, Eq. (6.5.4)
                double Dt1 = dt - dt * ( Y0 - WingY ) / ( Y0 - sim->P[i].y );
                // Generate velocity vector of the reflected molecule
                diffuse_scattering_y(&(sim->P[i].vx), &(sim->P[i].vy), &(sim->P[i].vz), moleculeMass,Tw,(Y0-WingY>0)?1.0:(-1.0));
                // Move the reflected molecule
                sim->P[i].x = Xw + Dt1 * sim->P[i].vx;
                sim->P[i].y = WingY + Dt1 * sim->P[i].vy;
            }
        }
        #elif defined(BALL_CASE)
        // Ray-sphere intersection test
        {
            // Initial and final positions
            double x0 = X0, y0 = Y0, z0 = Z0;
            double x1 = sim->P[i].x, y1 = sim->P[i].y, z1 = sim->P[i].z;

            // Direction of motion
            double dx = x1 - x0;
            double dy = y1 - y0;
            double dz = z1 - z0;

            // Sphere center
            double cx = ballCenterX;
            double cy = ballCenterY;
            double cz = ballCenterZ;

            // Shifted initial position
            double rx = x0 - cx;
            double ry = y0 - cy;
            double rz = z0 - cz;

            // Quadratic coefficients: |r + t d|^2 = R^2
            double a = dx*dx + dy*dy + dz*dz;
            double b = 2.0 * (rx*dx + ry*dy + rz*dz);
            double c = rx*rx + ry*ry + rz*rz - ballRadius*ballRadius;

            double disc = b*b - 4.0*a*c;

            if (disc >= 0.0) {
                double sqrt_disc = sqrt(disc);

                // time solutions
                double t1 = (-b - sqrt_disc) / (2.0*a);
                double t2 = (-b + sqrt_disc) / (2.0*a);

                // pick earliest valid intersection in [0,1]
                double t_hit = -1.0;
                if (t1 >= 0.0 && t1 <= 1.0) t_hit = t1;
                else if (t2 >= 0.0 && t2 <= 1.0) t_hit = t2;

                if (t_hit >= 0.0) {
                    // Intersection point
                    double Xw = x0 + t_hit * dx;
                    double Yw = y0 + t_hit * dy;
                    double Zw = z0 + t_hit * dz;

                    // Remaining time after collision
                    double Dt1 = dt * (1.0 - t_hit);

                    // Surface normal (outward)
                    double nx = (Xw - cx) / ballRadius;
                    double ny = (Yw - cy) / ballRadius;
                    double nz = (Zw - cz) / ballRadius;

                    // Diffuse reflection aligned with normal
                    diffuse_scattering(&(sim->P[i].vx), &(sim->P[i].vy), &(sim->P[i].vz),
                                    moleculeMass, Tb,
                                    nx, ny, nz);

                    // Move after collision
                    sim->P[i].x = Xw + Dt1 * sim->P[i].vx;
                    sim->P[i].y = Yw + Dt1 * sim->P[i].vy;
                    sim->P[i].z = Zw + Dt1 * sim->P[i].vz;
                }
            }
        }

        #endif
    }
}


void accumulate_sampling(Simulation *sim) {
    sim->sampleSteps++;

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                int Nc = sim->cellCount[k][l][m];

                sim->samples[k][l][m].countNP += Nc;

                for (int q = 0; q < Nc; q++) {
                    int i = sim->cellList[k][l][m][q];

                    double vx = sim->P[i].vx;
                    double vy = sim->P[i].vy;
                    double vz = sim->P[i].vz;

                    sim->samples[k][l][m].countVx += vx;
                    sim->samples[k][l][m].countVy += vy;
                    sim->samples[k][l][m].countVz += vz;
                    sim->samples[k][l][m].countV2 += vx * vx + vy * vy + vz * vz;
                }
            }
        }
    }
}

void clearPointers(Simulation *sim) {

}
