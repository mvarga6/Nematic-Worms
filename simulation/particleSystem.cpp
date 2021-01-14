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

// OpenGL Graphics includes
// #define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
// #include <helper_gl.h>

#include "particleSystem.h"
#include "particleSystem.cuh"
#include "particles_kernel.cuh"

#include <cuda_runtime.h>

#include <helper_functions.h>
#include <helper_math.h>
#include <helper_cuda.h>

#include <assert.h>
#include <math.h>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>

#ifndef CUDART_PI_F
#define CUDART_PI_F         3.141592654f
#endif

ParticleSystem::ParticleSystem(uint numFilaments, uint filamentSize, uint3 gridSize) :
    m_bInitialized(false),
    m_numFilaments(numFilaments),
    m_filamentSize(filamentSize),
    m_numParticles(numFilaments * filamentSize),
    m_hPos(0),
    m_hVel(0),
    m_hForce(0),
    m_hTangent(0),
    m_dPos(0),
    m_dVel(0),
    m_dTangent(0),
    m_dForce(0),
    m_dForceOld(0),
    m_dRandom(0),
    m_gridSize(gridSize),
    m_timer(NULL),
    m_solverIterations(1)
{
    // Filament parameters
    m_params.numFilaments = m_numFilaments;
    m_params.filamentSize = m_filamentSize;
    m_params.numParticles = m_numParticles;
    m_params.particleRadius = 0.5f;

    // System size/boundaries
    m_gridSortBits = 18;    // increase this for larger grids
    m_numGridCells = m_gridSize.x*m_gridSize.y*m_gridSize.z;
    m_params.gridSize = m_gridSize;
    m_params.numCells = m_numGridCells;
    float cellSize = m_params.particleRadius * 2.0f;  // cell size equal to particle diameter
    m_params.cellSize = make_float3(cellSize, cellSize, cellSize);
    m_params.boxSize = make_float3(gridSize.x*cellSize, gridSize.y*cellSize, gridSize.z*cellSize);
    m_params.origin = -m_params.boxSize / 2.0f;
    m_params.boundaryX = BoundaryType::PERIODIC;
    m_params.boundaryY = BoundaryType::PERIODIC;
    m_params.boundaryZ = BoundaryType::PERIODIC;

    // Particle-Particle bonding
    m_params.bondSpringK = 57.f;
    m_params.bondSpringL = m_params.particleRadius * 0.8f;
    m_params.bondBendingK = 0.001f;

    // Particle-Particle forces
    m_params.spring = 0.5f;
    m_params.damping = 0.02f;
    m_params.shear = 0.1f;
    m_params.attraction = 0.0f;
    m_params.boundaryDamping = -0.5f;

    // Active forces
    m_params.activity = 0.1f;
    m_params.reverseProbability = 0.0f;

    // Langevin thermostat
    m_params.kbT = 1.0f;
    m_params.gamma = 0.01;

    // Global interations
    m_params.gravity = make_float3(0.0f);

    _initialize(m_numParticles);
}

ParticleSystem::~ParticleSystem()
{
    _finalize();
    m_numParticles = m_numFilaments = m_filamentSize = 0;
}

// create a color ramp
void colorRamp(float t, float *r)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 1.0, 0.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
    };
    t = t * (ncolors-1);
    int i = (int) t;
    float u = t - floor(t);
    r[0] = lerp(c[i][0], c[i+1][0], u);
    r[1] = lerp(c[i][1], c[i+1][1], u);
    r[2] = lerp(c[i][2], c[i+1][2], u);
}

void
ParticleSystem::_initialize(int numParticles)
{
    assert(!m_bInitialized);

    m_numParticles = numParticles;

    // allocate host storage
    m_hPos     = new float[m_numParticles*4];
    m_hVel     = new float[m_numParticles*4];
    m_hForce   = new float[m_numParticles*4];
    m_hTangent = new float[m_numParticles*4];
    memset(m_hPos, 0, m_numParticles*4*sizeof(float));
    memset(m_hVel, 0, m_numParticles*4*sizeof(float));
    memset(m_hForce, 0, m_numParticles*4*sizeof(float));
    memset(m_hTangent, 0, m_numParticles*4*sizeof(float));

    m_hCellStart = new uint[m_numGridCells];
    memset(m_hCellStart, 0, m_numGridCells*sizeof(uint));

    m_hCellEnd = new uint[m_numGridCells];
    memset(m_hCellEnd, 0, m_numGridCells*sizeof(uint));

    // allocate GPU data
    unsigned int memSize = sizeof(float) * 4 * m_numParticles;
    allocateArray((void **)&m_dPos, memSize);
    allocateArray((void **)&m_dVel, memSize);
    allocateArray((void **)&m_dTangent, memSize);
    allocateArray((void **)&m_dForce, memSize);
    allocateArray((void **)&m_dForceOld, memSize);
    allocateArray((void **)&m_dRandom, memSize);
    allocateArray((void **)&m_dSortedPos, memSize);
    allocateArray((void **)&m_dSortedVel, memSize);
    allocateArray((void **)&m_dSortedTangent, memSize);
    allocateArray((void **)&m_dGridParticleHash, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dGridParticleIndex, m_numParticles*sizeof(uint));
    allocateArray((void **)&m_dCellStart, m_numGridCells*sizeof(uint));
    allocateArray((void **)&m_dCellEnd, m_numGridCells*sizeof(uint));

    sdkCreateTimer(&m_timer);
    setParameters(&m_params);

    m_bInitialized = true;
}

void
ParticleSystem::_finalize()
{
    assert(m_bInitialized);

    delete [] m_hPos;
    delete [] m_hVel;
    delete [] m_hTangent;
    delete [] m_hCellStart;
    delete [] m_hCellEnd;

    freeArray(m_dPos);
    freeArray(m_dVel);
    freeArray(m_dTangent);
    freeArray(m_dSortedPos);
    freeArray(m_dSortedVel);

    freeArray(m_dGridParticleHash);
    freeArray(m_dGridParticleIndex);
    freeArray(m_dCellStart);
    freeArray(m_dCellEnd);
}

// step the simulation
void
ParticleSystem::update(float deltaTime)
{
    assert(m_bInitialized);

    // update constants
    setParameters(&m_params);

    // integrate
    integrateSystem(
        m_dPos,
        m_dVel,
        m_dForce,
        m_dForceOld,
        deltaTime,
        m_numParticles);

    if (m_params.reverseProbability > 0.0f)
    {
        reverseFilaments(
            m_dForce,
            m_dForceOld,
            m_dVel,
            m_dPos,
            m_dRandom,
            m_numFilaments);
    }

    if (m_params.kbT > 0.0f || m_params.gamma > 0.0f)
    {
        langevinThermostat(
            m_dForce,
            m_dVel,
            m_dRandom,
            m_params.gamma,
            m_params.kbT,
            m_numParticles);
    }

    // forces of filament bonds
    filamentForces(
        m_dForce,
        m_dTangent,
        m_dPos,
        m_numFilaments);

    // calculate grid hash
    calcHash(
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_numParticles);

    // sort particles based on hash
    sortParticles(m_dGridParticleHash, m_dGridParticleIndex, m_numParticles);

    // reorder particle arrays into sorted order and
    // find start and end of each cell
    reorderDataAndFindCellStart(
        m_dCellStart,
        m_dCellEnd,
        m_dSortedPos,
        m_dSortedVel,
        m_dSortedTangent,
        m_dGridParticleHash,
        m_dGridParticleIndex,
        m_dPos,
        m_dVel,
        m_dTangent,
        m_numParticles,
        m_numGridCells);

    // process collisions
    collide(
        m_dForce,
        m_dSortedPos,
        m_dSortedVel,
        m_dSortedTangent,
        m_dGridParticleIndex,
        m_dCellStart,
        m_dCellEnd,
        m_numParticles,
        m_numGridCells);
}

void
ParticleSystem::dumpGrid()
{
    // dump grid information
    copyArrayFromDevice(m_hCellStart, m_dCellStart, 0, sizeof(uint)*m_numGridCells);
    copyArrayFromDevice(m_hCellEnd, m_dCellEnd, 0, sizeof(uint)*m_numGridCells);
    uint maxCellSize = 0;

    for (uint i=0; i<m_numGridCells; i++)
    {
        if (m_hCellStart[i] != 0xffffffff)
        {
            uint cellSize = m_hCellEnd[i] - m_hCellStart[i];
            if (cellSize > maxCellSize)
            {
                maxCellSize = cellSize;
            }
        }
    }

    printf("maximum particles per cell = %d\n", maxCellSize);
}

void
ParticleSystem::dumpParticles(uint start, uint count)
{
    copyArrayFromDevice(m_hVel, m_dPos, 0, sizeof(float)*4*count);
    copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*count);

    for (uint i=start; i<start+count; i++)
    {
        printf("pos: (%.4f, %.4f, %.4f, %.4f)\n", m_hPos[i*4+0], m_hPos[i*4+1], m_hPos[i*4+2], m_hPos[i*4+3]);
        printf("vel: (%.4f, %.4f, %.4f, %.4f)\n", m_hVel[i*4+0], m_hVel[i*4+1], m_hVel[i*4+2], m_hVel[i*4+3]);
    }
}

void
ParticleSystem::writeOutputs(const std::string& fileName)
{
    copyArrayFromDevice(m_hPos, m_dPos, 0, sizeof(float)*4*m_numParticles);
    // copyArrayFromDevice(m_hVel, m_dVel, 0, sizeof(float)*4*m_numParticles);
    // copyArrayFromDevice(m_hForce, m_dForce, 0, sizeof(float)*4*m_numParticles);
    copyArrayFromDevice(m_hTangent, m_dTangent, 0, sizeof(float)*4*m_numParticles);

    std::ofstream fout;
    fout.open(fileName, std::ios::out | std::ios::app);
    fout << m_numParticles << std::endl;
    fout << "Active Filaments Simulation" << std::endl;
    for (int i = 0; i < m_numParticles; i++)
    {
        fout << "A " << m_hPos[i*4] << " " << m_hPos[i*4+1] << " " << m_hPos[i*4+2]
            //  << " "  << m_hVel[i*4] << " " << m_hVel[i*4+1] << " " << m_hVel[i*4+2]
            //  << " "  << m_hForce[i*4] << " " << m_hForce[i*4+1] << " " << m_hForce[i*4+2]
             << " "  << m_hTangent[i*4] << " " << m_hTangent[i*4+1] << " " << m_hTangent[i*4+2]
             << std::endl;
    }
    fout.close();
}

float *
ParticleSystem::getArray(ParticleArray array)
{
    assert(m_bInitialized);

    float *hdata = 0;
    float *ddata = 0;
    struct cudaGraphicsResource *cuda_vbo_resource = 0;

    switch (array)
    {
        default:
        case POSITION:
            hdata = m_hPos;
            ddata = m_dPos;
            break;

        case VELOCITY:
            hdata = m_hVel;
            ddata = m_dVel;
            break;
    }

    copyArrayFromDevice(hdata, ddata, &cuda_vbo_resource, m_numParticles*4*sizeof(float));
    return hdata;
}

void
ParticleSystem::setArray(ParticleArray array, const float *data, int start, int count)
{
    assert(m_bInitialized);

    switch (array)
    {
        default:
        case POSITION:
            copyArrayToDevice(m_dPos, data, start*4*sizeof(float), count*4*sizeof(float));
            break;

        case VELOCITY:
            copyArrayToDevice(m_dVel, data, start*4*sizeof(float), count*4*sizeof(float));
            break;

        case TANGENT:
            copyArrayToDevice(m_dTangent, data, start*4*sizeof(float), count*4*sizeof(float));
            break;
    }
}

inline float frand()
{
    return rand() / (float) RAND_MAX;
}


void
ParticleSystem::reset()
{
    float filamentLength = m_params.filamentSize * m_params.bondSpringL + 2.0f * m_params.particleRadius;
    int xdim_max = int(floor(m_params.boxSize.x / filamentLength));
    int ydim_max = int(floor(m_params.boxSize.y / (2 * m_params.particleRadius)));
    int xyplane_max = xdim_max * ydim_max;
    float x,y,z,x_head;
    uint w, p = 0, v = 0, t = 0;

    printf("CONFIG_GRID:\n");
    printf("filamentLength: %.4f\n", filamentLength);
    printf("filamentsXdim:  %i\n",   xdim_max);
    printf("filamentsYdim:  %i\n",   ydim_max);

    for (uint i = 0; i < m_numParticles; i++)
    {
        w = (i / m_params.filamentSize);
        x_head = (w % xdim_max) * filamentLength + m_params.particleRadius;
        x = x_head + (i % m_params.filamentSize) * m_params.bondSpringL;
        y = 2.f * m_params.particleRadius * ((w / xdim_max) % ydim_max + 1);
        z = 2.f * m_params.particleRadius * (w / xyplane_max + 1);
        m_hPos[p++] = x + m_params.origin.x;
        m_hPos[p++] = y + m_params.origin.y;
        m_hPos[p++] = z + m_params.origin.z;
        m_hPos[p++] = m_params.particleRadius; // radius
        m_hVel[v++] = 0.1f * (frand()*2.0f-1.0f);
        m_hVel[v++] = 0.1f * (frand()*2.0f-1.0f);
        m_hVel[v++] = 0.1f * (frand()*2.0f-1.0f);
        m_hVel[v++] = 0.0f;
        m_hTangent[t++] = 1.0f;
        m_hTangent[t++] = m_hTangent[t++] = m_hTangent[t++] = 0.0f;
    }

    // Flip half the filaments
    for (w = 0; w < m_params.numFilaments; w++)
    {
        if (frand() < 0.5f) continue;

        uint i,j;
        float m;
        for (p = 0; p < m_params.filamentSize / 2; p++)
        {
            i = w      * m_params.filamentSize      + p; // starts at head
            j = ((w+1) * m_params.filamentSize) - 1 - p; // starts at tail
            x = m_hPos[i*4 + 0]; // tmp save head pos
            y = m_hPos[i*4 + 1];
            z = m_hPos[i*4 + 2];
            m = m_hPos[i*4 + 3];
            m_hPos[i*4 + 0] = m_hPos[j*4 + 0]; // Assign j pos to i
            m_hPos[i*4 + 1] = m_hPos[j*4 + 1];
            m_hPos[i*4 + 2] = m_hPos[j*4 + 2];
            m_hPos[i*4 + 3] = m_hPos[j*4 + 3];
            m_hPos[j*4 + 0] = x; // Assign i pos to j
            m_hPos[j*4 + 1] = y;
            m_hPos[j*4 + 2] = z;
            m_hPos[j*4 + 3] = m;
            m_hTangent[i*4 + 0] = -m_hTangent[i*4 + 0]; // Invert tangent i
            m_hTangent[i*4 + 1] = -m_hTangent[i*4 + 1];
            m_hTangent[i*4 + 2] = -m_hTangent[i*4 + 2];
            m_hTangent[i*4 + 3] = -m_hTangent[i*4 + 3];
            m_hTangent[j*4 + 0] = -m_hTangent[j*4 + 0]; // Invert tangent j
            m_hTangent[j*4 + 1] = -m_hTangent[j*4 + 1];
            m_hTangent[j*4 + 2] = -m_hTangent[j*4 + 2];
            m_hTangent[j*4 + 3] = -m_hTangent[j*4 + 3];
        }
    }

    setArray(POSITION, m_hPos, 0, m_numParticles);
    setArray(VELOCITY, m_hVel, 0, m_numParticles);
    setArray(TANGENT, m_hTangent, 0, m_numParticles);
}

void
ParticleSystem::addSphere(int start, float *pos, float *vel, int r, float spacing)
{
    uint index = start;

    for (int z=-r; z<=r; z++)
    {
        for (int y=-r; y<=r; y++)
        {
            for (int x=-r; x<=r; x++)
            {
                float dx = x*spacing;
                float dy = y*spacing;
                float dz = z*spacing;
                float l = sqrtf(dx*dx + dy*dy + dz*dz);
                float jitter = m_params.particleRadius*0.01f;

                if ((l <= m_params.particleRadius*2.0f*r) && (index < m_numParticles))
                {
                    m_hPos[index*4]   = pos[0] + dx + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+1] = pos[1] + dy + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+2] = pos[2] + dz + (frand()*2.0f-1.0f)*jitter;
                    m_hPos[index*4+3] = pos[3];

                    m_hVel[index*4]   = vel[0];
                    m_hVel[index*4+1] = vel[1];
                    m_hVel[index*4+2] = vel[2];
                    m_hVel[index*4+3] = vel[3];
                    index++;
                }
            }
        }
    }

    setArray(POSITION, m_hPos, start, index);
    setArray(VELOCITY, m_hVel, start, index);
}
