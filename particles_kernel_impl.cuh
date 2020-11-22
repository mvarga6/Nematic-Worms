/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * CUDA particle system kernel code.
 */

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include <stdio.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

// simulation parameters in constant memory
__constant__ SimParams params;


struct integrate_functor
{
    float dt;

    __host__ __device__
    integrate_functor(float delta_time) : dt(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<0>(t);
        volatile float4 velData = thrust::get<1>(t);
        volatile float4 forceData = thrust::get<2>(t);
        volatile float4 forceOldData = thrust::get<3>(t);
        float3 pos   = make_float3(posData.x, posData.y, posData.z);
        float3 vel   = make_float3(velData.x, velData.y, velData.z);
        float3 f     = make_float3(forceData.x, forceData.y, forceData.z);
        float3 f_old = make_float3(forceOldData.x, forceOldData.y, forceOldData.z);

        f += params.gravity;

        // Velocity Verlet update
        pos += vel * dt + 0.5f * f_old * dt * dt;
        vel += 0.5f * (f + f_old) * dt;
        vel *= params.globalDamping;

        // set this to zero to disable collisions with cube sides
#if 0

        if (pos.x > 1.0f - params.particleRadius)
        {
            pos.x = 1.0f - params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.x < -1.0f + params.particleRadius)
        {
            pos.x = -1.0f + params.particleRadius;
            vel.x *= params.boundaryDamping;
        }

        if (pos.y > 1.0f - params.particleRadius)
        {
            pos.y = 1.0f - params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

        if (pos.z > 1.0f - params.particleRadius)
        {
            pos.z = 1.0f - params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

        if (pos.z < -1.0f + params.particleRadius)
        {
            pos.z = -1.0f + params.particleRadius;
            vel.z *= params.boundaryDamping;
        }

#endif

        // Ground
        // if (pos.y < -1.0f + params.particleRadius)
        // {
        //     pos.y = -1.0f + params.particleRadius;
        //     vel.y *= params.boundaryDamping;
        // }

        if (pos.y < params.origin.y + params.particleRadius)
        {
            pos.y = params.origin.y + params.particleRadius;
            vel.y *= params.boundaryDamping;
        }

#if PBC_X
        if (pos.x < params.origin.x) pos.x += params.boxSize.x;
        else if (pos.x > params.boxSize.x + params.origin.x) pos.x -= params.boxSize.x;
#endif
#if PBC_Y
        if (pos.y < params.origin.y) pos.y += params.boxSize.y;
        else if (pos.y > params.boxSize.y + params.origin.y) pos.y -= params.boxSize.y;
#endif
#if PBC_Z
        if (pos.z < params.origin.z) pos.z += params.boxSize.z;
        else if (pos.z > params.boxSize.z + params.origin.z) pos.z -= params.boxSize.z;
#endif

        // store new position, velocity, and forces
        thrust::get<0>(t) = make_float4(pos, posData.w);
        thrust::get<1>(t) = make_float4(vel, velData.w);
        thrust::get<2>(t) = make_float4(0.f);
        thrust::get<3>(t) = make_float4(f, forceOldData.w);
    }
};

// calculate position in uniform grid
__device__ int3 calcGridPos(float3 p)
{
    int3 gridPos;
    gridPos.x = floor((p.x - params.origin.x) / params.cellSize.x);
    gridPos.y = floor((p.y - params.origin.y) / params.cellSize.y);
    gridPos.z = floor((p.z - params.origin.z) / params.cellSize.z);
    return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ uint calcGridHash(int3 gridPos)
{
    gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
    gridPos.y = gridPos.y & (params.gridSize.y-1);
    gridPos.z = gridPos.z & (params.gridSize.z-1);
    return __umul24(__umul24(gridPos.z, params.gridSize.y), params.gridSize.x) + __umul24(gridPos.y, params.gridSize.x) + gridPos.x;
}

// calculate address in grid from position (clamping to edges)
__device__ float lengthPeriodic(float3& dr)
{
    float3 L = params.boxSize;
    float3 Lo2 = L / 2.0f;
#if PBC_X
    if (dr.x > Lo2.x) dr.x -= L.x;
    else if (dr.x < -Lo2.x) dr.x += L.x;
#endif
#if PBX_Y
    if (dr.y > Lo2.y) dr.y -= L.y;
    else if (dr.y < -Lo2.y) dr.y += L.y;
#endif
#if PBC_Z
    if (dr.z > Lo2.z) dr.z -= L.z;
    else if (dr.z < -Lo2.z) dr.z += L.z;
#endif
    return length(dr);
}

__device__
float3 getTangent(float3 posA, float3 posB)
{
    float3 relPos = posB - posA;
    float dist = lengthPeriodic(relPos);
    return relPos / dist;
}

// calculate grid hash value for each particle
__global__
void calcHashD(uint   *gridParticleHash,  // output
               uint   *gridParticleIndex, // output
               float4 *pos,               // input: positions
               uint    numParticles)
{
    uint index = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    volatile float4 p = pos[index];

    // get address in grid
    int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z));
    uint hash = calcGridHash(gridPos);

    // store grid hash and particle index
    gridParticleHash[index] = hash;
    gridParticleIndex[index] = index;
}

// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
                                  uint   *cellEnd,          // output: cell end index
                                  float4 *sortedPos,        // output: sorted positions
                                  float4 *sortedVel,        // output: sorted velocities
                                  uint   *gridParticleHash, // input: sorted grid hashes
                                  uint   *gridParticleIndex,// input: sorted particle indices
                                  float4 *pos,              // input: sorted position array
                                  float4 *vel,              // input: sorted velocity array
                                  uint    numParticles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ uint sharedHash[];    // blockSize + 1 elements
    uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

    uint hash;

    // handle case when no. of particles not multiple of block size
    if (index < numParticles)
    {
        hash = gridParticleHash[index];

        // Load hash data into shared memory so that we can look
        // at neighboring particle's hash value without loading
        // two hash values per thread
        sharedHash[threadIdx.x+1] = hash;

        if (index > 0 && threadIdx.x == 0)
        {
            // first thread in block must load neighbor particle hash
            sharedHash[0] = gridParticleHash[index-1];
        }
    }

    cg::sync(cta);

    if (index < numParticles)
    {
        // If this particle has a different cell index to the previous
        // particle then it must be the first particle in the cell,
        // so store the index of this particle in the cell.
        // As it isn't the first particle, it must also be the cell end of
        // the previous particle's cell

        if (index == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = index;

            if (index > 0)
                cellEnd[sharedHash[threadIdx.x]] = index;
        }

        if (index == numParticles - 1)
        {
            cellEnd[hash] = index + 1;
        }

        // Now use the sorted index to reorder data array
        uint sortedIndex = gridParticleIndex[index];
        sortedPos[index] = pos[sortedIndex];
        sortedVel[index] = vel[sortedIndex];
    }


}

// collide two spheres using DEM method
__device__
float3 collideParticles(float3 posA, float3 posB,
                      float3 velA, float3 velB,
                      float radiusA, float radiusB,
                      float attraction)
{
    // calculate relative position
    float3 relPos = posB - posA;

    float dist = lengthPeriodic(relPos);
    float collideDist = radiusA + radiusB;

    float3 force = make_float3(0.0f);

    if (dist < collideDist)
    {
        float3 norm = relPos / dist;

        // relative velocity
        float3 relVel = velB - velA;

        // relative tangential velocity
        float3 tanVel = relVel - (dot(relVel, norm) * norm);

        // spring force
        force = -params.spring*(collideDist - dist) * norm;
        // dashpot (damping) force
        force += params.damping*relVel;
        // tangential shear force
        force += params.shear*tanVel;
        // attraction
        force += attraction*relPos;
    }

    return force;
}


// bond to particles with Hookean spring
__device__
float3 bondHookean(float3 posA, float3 posB, float k, float L)
{
    float3 relPos = posB - posA;
    float dist = lengthPeriodic(relPos);
    float3 norm = relPos / dist;
    return -k * (L - dist) * norm;
}


// // compute extensile drive forces
// __device__
// float3 extensileForce(float3 posA, float3 posB, float k, float L)
// {
//     float3 relPos = posB - posA;
//     float dist = lengthPeriodic(relPos);
//     float3 norm = relPos / dist;
//     return -k * (L - dist) * norm;
// }



// collide a particle against all other particles in a given cell
__device__
float3 collideCell(int3    gridPos,
                   uint    index,
                   float3  r,
                   float3  v,
                   float4 *pos,
                   float4 *vel,
                   uint   *cellStart,
                   uint   *cellEnd,
                   uint   *gridParticleIndex)
{
    uint gridHash = calcGridHash(gridPos);

    // get start of bucket for this cell
    uint startIndex    = cellStart[gridHash];
    uint filamentIndex = gridParticleIndex[index] / params.filamentSize;
    uint chainIndex    = gridParticleIndex[index] % params.filamentSize;
    uint filamentIndex2, chainIndex2;

    float3 force = make_float3(0.0f);

    if (startIndex != 0xffffffff)          // cell is not empty
    {
        // iterate over particles in this cell
        uint endIndex = cellEnd[gridHash];

        for (uint j=startIndex; j<endIndex; j++)
        {
            if (j != index)                // check not colliding with self
            {
                float3 r2      = make_float3(pos[j]);
                float3 v2      = make_float3(vel[j]);
                filamentIndex2 = gridParticleIndex[j] / params.filamentSize;
                chainIndex2    = gridParticleIndex[j] % params.filamentSize;

                // filament bonding
                if (filamentIndex == filamentIndex2 && ((chainIndex + 1 == chainIndex2) || (chainIndex - 1 == chainIndex2)))
                {
                    force += bondHookean(r, r2, params.bondSpringK, params.bondSpringL);
                }
                else // collide particles
                {
                    force += collideParticles(r, r2, v, v2, params.particleRadius, params.particleRadius, params.attraction);
                }
            }
        }
    }

    return force;
}


__global__
void collideKernel(float4 *forces,          // update: unsorted forces array
              float4 *sortedPos,            // input: sorted positions
              float4 *sortedVel,            // input: sorted velocities
              uint   *gridParticleIndex,    // input: sorted particle indices
              uint   *cellStart,
              uint   *cellEnd,
              uint    numParticles)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numParticles) return;

    // read particle data from sorted arrays
    float3 pos = make_float3(sortedPos[index]);
    float3 vel = make_float3(sortedVel[index]);

    // get address in grid
    int3 gridPos = calcGridPos(pos);

    // examine neighbouring cells
    float3 force = make_float3(0.0f);

    for (int z=-1; z<=1; z++)
    {
        for (int y=-1; y<=1; y++)
        {
            for (int x=-1; x<=1; x++)
            {
                int3 neighbourPos = gridPos + make_int3(x, y, z);
                force += collideCell(neighbourPos, index, pos, vel, sortedPos, sortedVel, cellStart, cellEnd, gridParticleIndex);
            }
        }
    }

    // write new velocity back to original unsorted location
    uint originalIndex = gridParticleIndex[index];
    forces[originalIndex] += make_float4(force, 0.0f);
}


//
// Implementation of the GROMACS harmonic angle potential
// http://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#harmonic-angle-potential
//
__global__
void filamentKernel(float4 *forces,     // update: particle forces
                    float4 *tangent,    // output: filament tangent at particle
                    float4 *pos,        // input:  particle positions
                    uint numFilaments)
{
    uint index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index >= numFilaments) return;

    const uint size = params.filamentSize;
    const float k = params.bondBendingK;

    uint i = 0;
    uint head_i = index * size;

    float3 r_i  = make_float3(0.0f), r_j  = make_float3(0.0f), r_k  = make_float3(0.0f);
    float3 f_i  = make_float3(0.0f), f_j  = make_float3(0.0f), f_k  = make_float3(0.0f);
    float3 r_ij = make_float3(0.0f), r_jk = make_float3(0.0f), r_ik = make_float3(0.0f);
    float A = 0.0f, d_ij = 0.0f, d_jk = 0.0f, d_ij_jk = 0.0f;
    float3 B = make_float3(0.0f), C = make_float3(0.0f);



    // Bond bending forces
    for (int p = 0; p < size - 2; p++)
    {
        i    = head_i + p;
        r_i  = make_float3(pos[i]);
        r_j  = make_float3(pos[i + 1]);
        r_k  = make_float3(pos[i + 2]);

        r_ij = r_j - r_i;
        r_jk = r_k - r_j;
        d_ij = lengthPeriodic(r_ij);
        d_jk = lengthPeriodic(r_jk);
        d_ij_jk = dot(r_ij, r_jk);

        // Derivation 1
        // A = (k / (d_ij*d_ij*d_jk*d_jk));
        // f_i = -A * ( (d_ij_jk * d_ij_jk / (d_ij*d_ij)) * r_ij - r_jk );
        // f_k = -A * ( (d_ij_jk * d_ij_jk / (d_jk*d_jk)) * r_jk - r_ij );
        // f_j = -f_i - f_k;

        // Derivation 2
        A = k / (d_ij * d_jk);
        B = r_ij * d_ij_jk / dot(r_ij, r_ij);
        C = r_jk * d_ij_jk / dot(r_jk, r_jk);

        f_i = -A * (r_jk - B);
        f_k = -A * (C - r_ij);
        f_j = -f_i - f_k;

        forces[i]     += make_float4(f_i, 0.0f);
        forces[i + 1] += make_float4(f_j, 0.0f);
        forces[i + 2] += make_float4(f_k, 0.0f);

        // Compute tangent at j
        tangent[i + 1] = make_float4(getTangent(r_i, r_k), 0.0f);

        if (p == 0) // Head tangent
        {
            tangent[i] = make_float4(getTangent(r_i, r_j), 0.0f);
        }
        else if (p == size - 2) // last iteration do tail as well
        {
            tangent[i + 2] = make_float4(getTangent(r_j, r_k), 0.0f);
        }
    }
}

#endif
