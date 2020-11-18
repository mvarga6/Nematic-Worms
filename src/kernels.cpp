#include "kernels.h"
#include "kernels_impl.h"


void FilamentBackboneForce(
    Particles* particles,
    int n_filaments,
    int filament_size,
    float spring_constant,
    float bond_length,
    int threads_per_block
)
{
    int num_blocks = n_filaments / threads_per_block + 1;
    FilamentBackboneForceKernel<<<num_blocks, threads_per_block>>>(
        particles->r,
        particles->f,
        n_filaments,
        filament_size,
        spring_constant,
        bond_length
    );
}

void ApplyPositionFunction(
    PositionFunction h_func,
    Particles* particles,
    float3 box,
    int threads_per_block
)
{
    int num_blocks = particles->count / threads_per_block + 1;
    ApplyPositionFunctionKernel<<<num_blocks, threads_per_block>>>(
        h_func,
        particles->dev_r,
        particles->count,
        box
    );
}