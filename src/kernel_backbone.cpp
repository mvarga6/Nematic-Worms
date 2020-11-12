#include "kernels_impl.h"
#include "math.cuh"


__global__
void FilamentBackboneForceKernel(ptype *position, ptype *force, int n_filaments, int filament_size, float spring_constant, float bond_length)
{
    int filament_i = threadIdx.x + blockDim.x * blockIdx.x;
    if (filament_i < n_filaments)
    {
        float3 dr = make_float3(0., 0., 0.);
        float3 fdr = make_float3(0., 0., 0.);
        float r, f;
        for (int p = 0; p < filament_size; p++)
        {
            int i = filament_i * filament_size + p;

            // If particle has a neighbor ahead
            if (p < filament_size - 1)
            {
                r = distance(position[i], position[i + 1], &dr);
                f = spring_constant * (r - bond_length) / r;
                fdr = f * dr;
                force[i] += fdr;
                force[i + 1] -= fdr;
            }

            // If particle has a neighbor behind
            if (p > 0)
            {
                r = distance(position[i], position[i - 1], &dr);
                f = spring_constant * (r - bond_length) / r;
                fdr = f * dr;
                force[i] += fdr;
                force[i - 1] -= fdr;
            }
        }
    }
}