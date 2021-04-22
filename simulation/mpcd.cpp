#include <random>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace std;


float uniform() {
    return ((float) rand()) / (float) RAND_MAX;
}

float random_sign() {
    return float(2 * (rand() % 2) - 1);
}

void init_particles(float *r, int *types, int n, const float box[2], const float large_particle_fraction) {
    for (int i = 0; i < n; i++) {
        types[i] = uniform() <= large_particle_fraction ?  1 : 0;
        for (int d = 0; d < 2; d++)
            r[i + d*n] = box[d] * uniform();
    }
}


void init_velocities(float *v, int n, float v_0) {
    float v_cmsx = 0.0f, v_cmsy = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i]     = v_0 * 2.f * uniform() - 1.0f;
        v[i + n] = v_0 * 2.f * uniform() - 1.0f;
        v_cmsx += v[i];
        v_cmsy += v[i + n];
    }

    v_cmsx /= n;
    v_cmsy /= n;

    // Substract off V CMS
    for (int i = 0; i < n; i++) {
        v[i] -= v_cmsx + 1.0f;
        v[i + n] -= v_cmsy;
    }
}


void streaming(float *r, float *v, int n, float dt, const float box[2]) {
    for (int i = 0; i < n; i++) {
        for (int d = 0; d < 2; d++) {
            r[i + d*n] += v[i + d*n] * dt;

            if (d == 0) { // X - Periodic boundary conditions
                if (r[i + d*n] > box[d]) {
                    r[i + d*n] -= box[d];
                }
                else if (r[i + d*n] < 0.0f) {
                    r[i + d*n] += box[d];
                }
            }
            // else { // Y - Slippery Walls d=1
            //     if (r[i + d*n] > box[d] || r[i + d*n] < 0) {
            //         r[i + d*n] -= v[i + d*n] * dt;
            //         v[i + d*n] = -v[i + d*n];
            //     }
            // }
            else { // Y - Non-slip Walls d=1
                if (r[i + n] > box[d] || r[i + n] < 0) {
                    r[i]     -= v[i] * dt;
                    r[i + n] -= v[i + n] * dt;
                    v[i]     = -v[i];
                    v[i + n] = -v[i + n];
                }
            }
        }
    }
}


float offsetX = 0.0f;
float offsetY = 0.0f;


void random_cell_shift(const float& l) {
    offsetX = l * uniform();
    offsetY = l * uniform();
}


void reset_m_cms(float *m_cms, const int m) {
    for (int i = 0; i < m; i++) {
        m_cms[i] = 0.0f;
    }
}


void reset_v_cms(float *v_cms, const int m) {
    for (int i = 0; i < m; i++) {
        v_cms[i]     = 0.0f;
        v_cms[i + m] = 0.0f;
    }
}


void rotate(float &vx, float &vy, const float& vx_cms, const float& vy_cms, const float& alpha) {
    const float cos_alpha = cosf(alpha), sin_alpha = sinf(alpha);
    vx = vx_cms + (vx - vx_cms) * cos_alpha - (vy - vy_cms) * sin_alpha;
    vy = vy_cms + (vx - vx_cms) * sin_alpha + (vy - vy_cms) * cos_alpha;
}


void collision(float *r,
               float *v,
               int *types,
               int n,
               const int n_cells[2],
               const float& cellL,
               const float& alpha,
               const float mass[2]
               ) {

    const int M = n_cells[0] * n_cells[1];

    float *m_cms = new float[M];
    float *v_cms = new float[2 * M];

    reset_m_cms(m_cms, M);
    reset_v_cms(v_cms, M);

    // For particle, sum velocities and count particles in cells
    int grid_x, grid_y, grid_index;
    for (int i = 0; i < n; i++) {
        grid_x = int((r[i]     + offsetX) / cellL) % n_cells[0];
        grid_y = int((r[i + n] + offsetY) / cellL) % n_cells[1];
        grid_index = grid_x + grid_y * n_cells[0];

        // cnts[grid_index]++;s
        m_cms[grid_index]     += mass[types[i]];
        v_cms[grid_index]     += mass[types[i]] * v[i];
        v_cms[grid_index + M] += mass[types[i]] * v[i + n];
    }

    // For each cell, convert to CMS velocites
    for (int i = 0; i < M; i++) {
        if (m_cms[i] > 0) {
            v_cms[i]     /= m_cms[i];
            v_cms[i + M] /= m_cms[i];
        }
    }

    // pick random sign for each cell
    int * sign = new int[M];
    for (int i = 0; i < M; i++) {
        sign[i] = random_sign();
    }

    // For each particle, Rotate particles
    for (int i = 0; i < n; i++) {
        grid_x = int((r[i]     + offsetX) / cellL) % n_cells[0];
        grid_y = int((r[i + n] + offsetY) / cellL) % n_cells[1];
        grid_index = grid_x + grid_y * n_cells[0];

        const float alp = sign[grid_index] * alpha;
        const float cos_alpha = cosf(alp), sin_alpha = sinf(alp);
        const float vx = v[i], vy = v[i + n], vx_cms = v_cms[grid_index], vy_cms = v_cms[grid_index + M];
        v[i]     = vx_cms + (vx - vx_cms) * cos_alpha - (vy - vy_cms) * sin_alpha;
        v[i + n] = vy_cms + (vx - vx_cms) * sin_alpha + (vy - vy_cms) * cos_alpha;
    }

    delete[] sign, m_cms, v_cms;
}


void save(float *r, int *type, int n) {
    ofstream fout;
    fout.open("particles.xyz", std::ios::out | std::ios::app);
    fout << n << endl;
    fout << "MPCD Simulation Test" << endl;
    const char color[2] = {'A', 'B'};
    for (int i = 0; i < n; i++) {
        fout << color[type[i]] << " " << r[i] << " " << r[i + n] << " 0 " << endl;
    }
    fout.close();
}


int main(int argc, char* argv[]) {

    // Input parameters
    const float ns      = 20;
    const float alpha   = 130.f * M_PI / 180.f;
    const float cellL   = 5.f;
    const float box[2]  = {200.f, 50.f};
    const float dt      = 0.1f;
    const int   T       = 10000;
    const float v0      = 1.5f;
    const int prate     = 50;
    const float mass[2] = {1.0f, 10.f};
    cout << "### Parameters input" << endl;

    // Calculated parameters
    const float cellA     = cellL * cellL;
    const float density   = ns / cellA;
    const float boxA      = box[0] * box[1];
    const int   N         = int(density * boxA);
    const int   Ncells[2] = {int(ceil(box[0] / cellL)), int(ceil(box[1] / cellL))};
    cout << "### Calculated parameters" << endl;
    cout << "  cellA = " << cellA << endl;
    cout << "density = " << density << endl;
    cout << "   boxA = " << boxA << endl;
    cout << "      N = " << N << endl;
    cout << " Ncells = " << Ncells[0] << "x" << Ncells[1] << "=" << Ncells[0] * Ncells[1] << endl;

    // Data variables
    float* r     = new float[2 * N];
    float* v     = new float[2 * N];
    float* v_cms = new float[2 * Ncells[0] * Ncells[1]];
    int*   type  = new int[N];
    cout << "### Memory allocated" << endl;

    init_particles(r, type, N, box, 0.1f);
    init_velocities(v, N, v0);

    for (int t = 0; t < T; t++) {
        cout << "T = " << t << endl;
        if (t % prate == 0)
            save(r, type, N);

        random_cell_shift(cellL);
        streaming(r, v, N, dt, box);
        collision(r, v, type, N, Ncells, cellL, alpha, mass);
    }

    return 0;
}