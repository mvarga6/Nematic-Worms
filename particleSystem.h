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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#define DEBUG_GRID 0
#define DO_TIMING 0

#include <string>
#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"

// Particle system class
class ParticleSystem
{
    public:
        ParticleSystem(uint numFilaments, uint filamentSize, uint3 gridSize);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        enum ParticleArray
        {
            POSITION,
            VELOCITY,
            TANGENT,
        };

        void update(float deltaTime);
        void reset(ParticleConfig config);

        float *getArray(ParticleArray array);
        void   setArray(ParticleArray array, const float *data, int start, int count);

        int    getNumParticles() const
        {
            return m_numParticles;
        }

        void dumpGrid();
        void dumpParticles(uint start, uint count);
        void writeOutputs(const std::string& fileName);

        void setIterations(int i)
        {
            m_solverIterations = i;
        }

        void setDamping(float x)
        {
            m_params.globalDamping = x;
        }
        void setGravity(float x)
        {
            m_params.gravity = make_float3(0.0f, x, 0.0f);
        }

        void setActivity(float x)
        {
            m_params.activity = x;
        }
        void setReverseProbability(float x)
        {
            m_params.reverseProbability = x;
        }

        void setCollideSpring(float x)
        {
            m_params.spring = x;
        }
        void setCollideDamping(float x)
        {
            m_params.damping = x;
        }
        void setCollideShear(float x)
        {
            m_params.shear = x;
        }
        void setCollideAttraction(float x)
        {
            m_params.attraction = x;
        }
        void setFilamentSize(uint n)
        {
            m_filamentSize = n;
            m_params.filamentSize = n;
            updateNumParticles();
        }
        void setNumFilaments(uint n)
        {
            m_numFilaments = n;
            m_params.numFilaments = n;
            updateNumParticles();
        }
        void setBondSpringConstant(float x)
        {
            m_params.bondSpringK = x;
        }
        void setBondSpringLength(float x)
        {
            m_params.bondSpringL = x;
        }
        void setBondBendingConstant(float x)
        {
            m_params.bondBendingK = x;
        }

        float getParticleRadius()
        {
            return m_params.particleRadius;
        }
        uint3 getGridSize()
        {
            return m_params.gridSize;
        }
        float3 getOrigin()
        {
            return m_params.origin;
        }
        float3 getCellSize()
        {
            return m_params.cellSize;
        }

        void addSphere(int index, float *pos, float *vel, int r, float spacing);

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticles);
        void _finalize();

        void initGrid(uint *size, float spacing, float jitter, uint numParticles);

        void updateNumParticles()
        {
            m_numParticles = m_numFilaments * m_filamentSize;
            m_params.numParticles = m_numParticles;
        }

    protected: // data
        bool m_bInitialized;
        uint m_numParticles;
        uint m_numFilaments;
        uint m_filamentSize;

        // CPU data
        float *m_hPos;              // particle positions
        float *m_hVel;              // particle velocities
        float *m_hForce;            // particle forces
        float *m_hTangent;          // tangent of filament at particle position

        uint  *m_hParticleHash;
        uint  *m_hCellStart;
        uint  *m_hCellEnd;

        // GPU data
        float *m_dPos;
        float *m_dVel;
        float *m_dTangent;
        float *m_dForce;
        float *m_dForceOld;
        float *m_dRandom;

        float *m_dSortedPos;
        float *m_dSortedVel;
        float *m_dSortedTangent;

        // grid data for sorting method
        uint  *m_dGridParticleHash; // grid hash value for each particle
        uint  *m_dGridParticleIndex;// particle index for each particle
        uint  *m_dCellStart;        // index of start of each cell in sorted list
        uint  *m_dCellEnd;          // index of end of cell

        uint   m_gridSortBits;

        // params
        SimParams m_params;
        uint3 m_gridSize;
        uint m_numGridCells;

        StopWatchInterface *m_timer;

        uint m_solverIterations;
};

#endif // __PARTICLESYSTEM_H__
