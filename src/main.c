#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <memory.h>
#include "../include/cell.h"
#include "../include/particle.h"
#include "../include/math_utils.h"
#include "../include/collider.h"

Particle P[MAX_PARTICLES];
Cell samples[NX][NY][NZ];
int cellCount[NX][NY][NZ];
int cellList[NX][NY][NZ][MAX_PARTICLES_PER_CELL];

// counters
int sampleSteps = 0;
int NP = 0;
long long totalCollisions = 0;

// derived quantities
double dx, dy, dz;
double cellVolume;
double weight;
double NFree;
double UFree;
double UxFree, UyFree, UzFree;

void setup(void) {
    // TODO: most of the setup can be moved to constexpr??

    memset(samples, 0, sizeof(samples));

    dx = Lx / NX;
    dy = Ly / NY;
    dz = Lz / NZ;
    cellVolume = dx * dy * dz;

    NP = NX * NY * NZ * particlesPerCellTarget;

    if (NP > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles for MAX_PARTICLES\n");
        exit(1);
    }

    NFree = PFree / ( KB * TFree );
    UFree = MaFree * sqrt ( ( 5.0 / 3.0 ) * KB * TFree / moleculeMass );
    UxFree = UFree * cos ( M_PI * angleOfAttack / 180.0 );
    UyFree = - UFree * sin ( M_PI * angleOfAttack / 180.0 );
    UzFree = 0.0;

    weight = NFree * cellVolume / particlesPerCellTarget;
}

void index_particles(void) {
    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                cellCount[k][l][m] = 0;
            }
        }
    }

    for (int i = 0; i < NP; i++) {
        int k = (int)(P[i].x / dx);
        int l = (int)(P[i].y / dy);
        int m = (int)(P[i].z / dz);

        if (k < 0) k = 0;
        if (k >= NX) k = NX - 1;
        if (l < 0) l = 0;
        if (l >= NY) l = NY - 1;
        if (m < 0) m = 0;
        if (m >= NZ) m = NZ - 1;

        int n = cellCount[k][l][m];
        cellList[k][l][m][n] = i;
        cellCount[k][l][m]++;
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
            for (int m = 0; m < NZ; m++) {
                int Nc = cellCount[k][l][m];

                samples[k][l][m].countNP += Nc;

                for (int q = 0; q < Nc; q++) {
                    int i = cellList[k][l][m][q];

                    double vx = P[i].vx;
                    double vy = P[i].vy;
                    double vz = P[i].vz;

                    samples[k][l][m].countVx += vx;
                    samples[k][l][m].countVy += vy;
                    samples[k][l][m].countVz += vz;
                    samples[k][l][m].countV2 += vx * vx + vy * vy + vz * vz;
                }
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

    fprintf(fp, "# x y z n ux uy uz T avg_count\n");

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                double xc = (k + 0.5) * dx;
                double yc = (l + 0.5) * dy;
                double zc = (m + 0.5) * dz;

                double avgNP = samples[k][l][m].countNP / sampleSteps;

                if (avgNP <= 0.0) {
                    fprintf(fp, "%e %e %e %e %e %e %e %e %e\n",
                            xc, yc, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    continue;
                }

                double ux = samples[k][l][m].countVx / samples[k][l][m].countNP;
                double uy = samples[k][l][m].countVy / samples[k][l][m].countNP;
                double uz = samples[k][l][m].countVz / samples[k][l][m].countNP;

                double meanV2 = samples[k][l][m].countV2 / samples[k][l][m].countNP;
                double meanU2 = ux * ux + uy * uy + uz * uz;

                double n = weight * avgNP / cellVolume;
                double T = moleculeMass * (meanV2 - meanU2) / (3.0 * KB);

                fprintf(fp, "%e %e %e %e %e %e %e %e %e\n",
                        xc, yc, zc, n, ux, uy, uz, T, avgNP);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void generate_particles_in_rect(double x1, double x2,
                                double y1, double y2,
                                double z1, double z2,
                                double ngas,
                                double ux, double uy, double uz,
                                double Tgas,
                                int moveFlag) {
    double V = (x2 - x1) * (y2 - y1) * (z2 - z1);
    double N_add_exp = ngas * V / weight;
    int Nnew;
    if ( N_add_exp > 20.0 ) { // use uniform approximation if large, poisson if small
        Nnew = (int)(N_add_exp);
        if (randu() < N_add_exp - Nnew) Nnew++;
    } else {
        Nnew = (int) randp( N_add_exp );
    }

    if (NP + Nnew > MAX_PARTICLES) {
        fprintf(stderr, "Too many particles\n");
        exit(1);
    }

    for (int i = NP; i < NP + Nnew; i++) {
        P[i].x = x1 + (x2 - x1) * randu();
        P[i].y = y1 + (y2 - y1) * randu();
        P[i].z = z1 + (z2 - z1) * randu();

        P[i].vx = randn(ux, sqrt(KB * Tgas / moleculeMass)); // TODO don't calc sqrt every time
        P[i].vy = randn(uy, sqrt(KB * Tgas / moleculeMass));
        P[i].vz = randn(uz, sqrt(KB * Tgas / moleculeMass));

        if (moveFlag) {
            P[i].x += dt * P[i].vx;
            P[i].y += dt * P[i].vy;
            P[i].z += dt * P[i].vz;
        }
    }

    NP += Nnew;
}

void initialize_particles(void) {
    NP = 0;
    generate_particles_in_rect(0.0, Lx, 0.0, Ly, 0.0, Lz, NFree, UxFree, UyFree, UzFree, TFree, 0);
}

void apply_boundary_conditions_free_stream() {
    generate_particles_in_rect(-DL, 0.0, 0.0, Ly, 0.0, Lz, NFree, UxFree, UyFree, UzFree, TFree, 1);
    generate_particles_in_rect(Lx, Lx + DL, 0.0, Ly, 0.0, Lz, NFree, UxFree, UyFree, UzFree, TFree, 1);
    generate_particles_in_rect(0.0, Lx, -DL, 0.0, 0.0, Lz, NFree, UxFree, UyFree, UzFree, TFree, 1);
    generate_particles_in_rect(0.0, Lx, Ly, Ly + DL, 0.0, Lz, NFree, UxFree, UyFree, UzFree, TFree, 1);
    generate_particles_in_rect(0.0, Lx, 0.0, Ly, -DL, 0.0, NFree, UxFree, UyFree, UzFree, TFree, 1);
    generate_particles_in_rect(0.0, Lx, 0.0, Ly, Lz, Lz + DL, NFree, UxFree, UyFree, UzFree, TFree, 1);


    for (int i = 0; i < NP; i++) {
        if (P[i].x < 0.0 || P[i].x >= Lx || P[i].y < 0.0 || P[i].y >= Ly || P[i].z < 0.0 || P[i].z >= Lz) {
            P[i] = P[NP - 1];
            NP--;
            i--;
        }
    }
}

void move_particles(void) {
    for (int i = 0; i < NP; i++) {
        double X0 = P[i].x;
        double Y0 = P[i].y;
        double Z0 = P[i].z;
        P[i].x += dt * P[i].vx;
        P[i].y += dt * P[i].vy;
        P[i].z += dt * P[i].vz;


        if ( ( Y0 - WingY ) * ( P[i].y - WingY ) < 0.0 ) {
            // Linear interpolation to point Y = WingY
            double Xw=( X0*(WingY-P[i].y)+P[i].x*(Y0-WingY))/(Y0-P[i].y);
            double Zw=( Z0*(WingY-P[i].y)+P[i].z*(Y0-WingY))/(Y0-P[i].y);
            if ( Zw < 0.3 || Zw > 0.7 ) continue; // wing only occupies 0.3 < z < 0.7
            if ( Xw > WingX && Xw < WingX + WingLength ) {
                // Molecule interacts with the wing during the time step
                // Linear interpolation of the time of scattering, Eq. (6.5.4)
                double Dt1 = dt - dt * ( Y0 - WingY ) / ( Y0 - P[i].y );
                // Generate velocity vector of the reflected molecule
                diffuse_scattering_y(&P[i].vx, &P[i].vy, &P[i].vz, moleculeMass,Tw,(Y0-WingY>0)?1.0:(-1.0));
                // Move the reflected molecule
                P[i].x = Xw + Dt1 * P[i].vx;
                P[i].y = WingY + Dt1 * P[i].vy;
            }
        }
    }


}

void write_vti(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open VTI file\n");
        exit(1);
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

    fprintf(fp, "<ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%e %e %e\">\n",
            NX, NY, NZ, dx, dy, dz);

    fprintf(fp, "<CellData Scalars=\"density\">\n");

    // Density
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n");
    for (int m = 0; m < NZ; m++) {
        for (int l = 0; l < NY; l++) {
            for (int k = 0; k < NX; k++) {
                double avgNP = samples[k][l][m].countNP / sampleSteps;
                double n = (avgNP > 0.0) ? weight * avgNP / cellVolume : 0.0;
                fprintf(fp, "%e ", n);
            }
        }
    }
    fprintf(fp, "\n</DataArray>\n");

    // Velocity
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (int m = 0; m < NZ; m++) {
        for (int l = 0; l < NY; l++) {
            for (int k = 0; k < NX; k++) {
                double count = samples[k][l][m].countNP;

                double ux = 0.0, uy = 0.0, uz = 0.0;
                if (count > 0.0) {
                    ux = samples[k][l][m].countVx / count;
                    uy = samples[k][l][m].countVy / count;
                    uz = samples[k][l][m].countVz / count;
                }

                fprintf(fp, "%e %e %e ", ux, uy, uz);
            }
        }
    }
    fprintf(fp, "\n</DataArray>\n");

    // Temperature
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"temperature\" format=\"ascii\">\n");
    for (int m = 0; m < NZ; m++) {
        for (int l = 0; l < NY; l++) {
            for (int k = 0; k < NX; k++) {
                double count = samples[k][l][m].countNP;

                double T = 0.0;
                if (count > 0.0) {
                    double ux = samples[k][l][m].countVx / count;
                    double uy = samples[k][l][m].countVy / count;
                    double uz = samples[k][l][m].countVz / count;

                    double meanV2 = samples[k][l][m].countV2 / count;
                    double meanU2 = ux*ux + uy*uy + uz*uz;

                    T = moleculeMass * (meanV2 - meanU2) / (3.0 * KB);
                }

                fprintf(fp, "%e ", T);
            }
        }
    }
    fprintf(fp, "\n</DataArray>\n");

    fprintf(fp, "</CellData>\n");
    fprintf(fp, "</ImageData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

void write_wing_vtp(const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open wing file\n");
        exit(1);
    }

    double x0 = WingX;
    double x1 = WingX + WingLength;
    double y  = WingY;
    double z0 = 0.3;
    double z1 = 0.7;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "<PolyData>\n");

    fprintf(fp, "<Piece NumberOfPoints=\"4\" NumberOfPolys=\"1\">\n");

    // Points
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    fprintf(fp, "%e %e %e\n", x0, y, z0);
    fprintf(fp, "%e %e %e\n", x1, y, z0);
    fprintf(fp, "%e %e %e\n", x1, y, z1);
    fprintf(fp, "%e %e %e\n", x0, y, z1);
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Points>\n");

    // One quad (4 vertices)
    fprintf(fp, "<Polys>\n");

    fprintf(fp, "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");
    fprintf(fp, "0 1 2 3\n");
    fprintf(fp, "</DataArray>\n");

    fprintf(fp, "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");
    fprintf(fp, "4\n");
    fprintf(fp, "</DataArray>\n");

    fprintf(fp, "</Polys>\n");

    fprintf(fp, "</Piece>\n");
    fprintf(fp, "</PolyData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

void write_paraview_files(unsigned int step) {
    char fname[64];
    sprintf(fname, "paraview_fields_%05d.vti", step);
    write_vti(fname);

    char wing_fname[64];
    sprintf(wing_fname, "paraview_wing_%05d.vtp", step);
    write_wing_vtp(wing_fname);
}

int main(void) {
    srand((unsigned int)time(NULL));

    setup();
    collider_setup();
    initialize_particles();

    for (int step = 0; step < nSteps; step++) {
        move_particles();
        apply_boundary_conditions_free_stream();
        index_particles();
        totalCollisions += collide_particles(P, cellCount, cellList, weight, cellVolume);

        if (step >= firstSampleStep && step % samplingPeriod == 0) {
            accumulate_sampling();
        }

        if (step % printPeriod == 0) {
            print_global_diagnostics(step);
            printf("totalCollisions = %lld\n", totalCollisions);
        }
    }

    print_global_diagnostics(nSteps);
    printf("totalCollisions = %lld\n", totalCollisions);

    write_averaged_macros("fields_avg.dat");

    write_paraview_files(nSteps);

    return 0;
}
