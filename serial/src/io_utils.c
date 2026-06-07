#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/io_utils.h"
#include "../include/simulation.h"

void print_global_diagnostics(Simulation *sim, Config *conf, int step) {
    double sumVx = 0.0, sumVy = 0.0, sumVz = 0.0;

    for (int i = 0; i < sim->NP; i++) {
        sumVx += sim->P[i].vx;
        sumVy += sim->P[i].vy;
        sumVz += sim->P[i].vz;
    }

    double ux = sumVx / sim->NP;
    double uy = sumVy / sim->NP;
    double uz = sumVz / sim->NP;

    double sumC2 = 0.0;
    for (int i = 0; i < sim->NP; i++) {
        double cx = sim->P[i].vx - ux;
        double cy = sim->P[i].vy - uy;
        double cz = sim->P[i].vz - uz;
        sumC2 += cx * cx + cy * cy + cz * cz;
    }

    double T = conf->moleculeMass * sumC2 / (3.0 * conf->KB * sim->NP);

    printf("step=%d\n", step);
    printf("  NP=%d\n", sim->NP);
    printf("  mean_u=(%.6e, %.6e, %.6e)\n", ux, uy, uz);
    printf("  T=%.6e\n", T);
    printf("  totalCollisions = %lld\n", sim->totalCollisions);
}

void write_averaged_macros(Simulation *sim, Config *conf, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open output file\n");
        exit(1);
    }

    fprintf(fp, "# x y z n ux uy uz T avg_count\n");

    for (int k = 0; k < NX; k++) {
        for (int l = 0; l < NY; l++) {
            for (int m = 0; m < NZ; m++) {
                double xc = (k + 0.5) * sim->dx;
                double yc = (l + 0.5) * sim->dy;
                double zc = (m + 0.5) * sim->dz;

                double avgNP = sim->samples[IDX_CELL(k, l, m)].countNP / sim->sampleSteps;

                if (avgNP <= 0.0) {
                    fprintf(fp, "%e %e %e %e %e %e %e %e %e\n",
                            xc, yc, zc, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                    continue;
                }

                double count = sim->samples[IDX_CELL(k, l, m)].countNP;

                double ux = sim->samples[IDX_CELL(k, l, m)].countVx / count;
                double uy = sim->samples[IDX_CELL(k, l, m)].countVy / count;
                double uz = sim->samples[IDX_CELL(k, l, m)].countVz / count;

                double meanV2 = sim->samples[IDX_CELL(k, l, m)].countV2 / count;
                double meanU2 = ux * ux + uy * uy + uz * uz;

                double n = sim->weight * avgNP / sim->cellVolume;
                double T = conf->moleculeMass * (meanV2 - meanU2) / (3.0 * conf->KB);

                fprintf(fp, "%e %e %e %e %e %e %e %e %e\n",
                        xc, yc, zc, n, ux, uy, uz, T, avgNP);
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

void write_vti(Simulation *sim, Config *conf, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open VTI file\n");
        exit(1);
    }

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");

    fprintf(fp, "<ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%e %e %e\">\n",
            NX, NY, NZ, sim->dx, sim->dy, sim->dz);

    fprintf(fp, "<CellData Scalars=\"density\">\n");

    // Density
    fprintf(fp, "<DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n");
    for (int m = 0; m < NZ; m++) {
        for (int l = 0; l < NY; l++) {
            for (int k = 0; k < NX; k++) {
                double avgNP = sim->samples[IDX_CELL(k, l, m)].countNP / sim->sampleSteps;
                double n = (avgNP > 0.0) ? sim->weight * avgNP / sim->cellVolume : 0.0;
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
                double count = sim->samples[IDX_CELL(k, l, m)].countNP;

                double ux = 0.0, uy = 0.0, uz = 0.0;
                if (count > 0.0) {
                    ux = sim->samples[IDX_CELL(k, l, m)].countVx / count;
                    uy = sim->samples[IDX_CELL(k, l, m)].countVy / count;
                    uz = sim->samples[IDX_CELL(k, l, m)].countVz / count;
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
                double count = sim->samples[IDX_CELL(k, l, m)].countNP;

                double T = 0.0;
                if (count > 0.0) {
                    double ux = sim->samples[IDX_CELL(k, l, m)].countVx / count;
                    double uy = sim->samples[IDX_CELL(k, l, m)].countVy / count;
                    double uz = sim->samples[IDX_CELL(k, l, m)].countVz / count;

                    double meanV2 = sim->samples[IDX_CELL(k, l, m)].countV2 / count;
                    double meanU2 = ux*ux + uy*uy + uz*uz;

                    T = conf->moleculeMass * (meanV2 - meanU2) / (3.0 * conf->KB);
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

#ifdef WING_CASE
void write_wing_vtp(Config *conf, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open wing file\n");
        exit(1);
    }

    double x0 = conf->WingX;
    double x1 = conf->WingX + conf->WingLength;
    double y  = conf->WingY;
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
#endif

#ifdef SPHERE_CASE
void write_sphere_vtp(Config *conf, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Could not open sphere output file\n");
        exit(1);
    }

    int n_theta = 30;
    int n_phi = 30;

    int num_points = (n_theta + 1) * (n_phi + 1);
    int num_tris   = n_theta * n_phi * 2;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "<PolyData>\n");

    fprintf(fp, "<Piece NumberOfPoints=\"%d\" NumberOfPolys=\"%d\">\n",
            num_points, num_tris);

    // --- POINTS ---
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n");

    for (int i = 0; i <= n_theta; i++) {
        double theta = M_PI * i / n_theta;  // 0 -> pi

        for (int j = 0; j <= n_phi; j++) {
            double phi = 2.0 * M_PI * j / n_phi; // 0 -> 2pi

            double x = conf->sphereCenterX + conf->sphereRadius * sin(theta) * cos(phi);
            double y = conf->sphereCenterY + conf->sphereRadius * sin(theta) * sin(phi);
            double z = conf->sphereCenterZ + conf->sphereRadius * cos(theta);

            fprintf(fp, "%e %e %e\n", x, y, z);
        }
    }

    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Points>\n");

    // --- TRIANGLES ---
    fprintf(fp, "<Polys>\n");

    // connectivity
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n");

    for (int i = 0; i < n_theta; i++) {
        for (int j = 0; j < n_phi; j++) {

            int p0 = i * (n_phi + 1) + j;
            int p1 = p0 + 1;
            int p2 = p0 + (n_phi + 1);
            int p3 = p2 + 1;

            // triangle 1
            fprintf(fp, "%d %d %d\n", p0, p2, p1);

            // triangle 2
            fprintf(fp, "%d %d %d\n", p1, p2, p3);
        }
    }

    fprintf(fp, "</DataArray>\n");

    // offsets
    fprintf(fp, "<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n");

    int offset = 0;
    for (int i = 0; i < num_tris; i++) {
        offset += 3;
        fprintf(fp, "%d\n", offset);
    }

    fprintf(fp, "</DataArray>\n");

    fprintf(fp, "</Polys>\n");

    fprintf(fp, "</Piece>\n");
    fprintf(fp, "</PolyData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}
#endif

void write_paraview_files(Simulation *sim, Config *conf, unsigned int step) {
    char fname[64];
    sprintf(fname, "paraview_fields_%05d.vti", step);
    write_vti(sim, conf, fname);

    #ifdef WING_CASE
        char wing_fname[64];
        sprintf(wing_fname, "paraview_wing_%05d.vtp", step);
        write_wing_vtp(conf, wing_fname);
    #elif defined(SPHERE_CASE)
        char sphere_fname[64];
        sprintf(sphere_fname, "paraview_sphere_%05d.vtp", step);
        write_sphere_vtp(conf, sphere_fname);
    #endif
}