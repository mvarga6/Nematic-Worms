#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__
float3 operator*(const float &lhs, const float3 &rhs);
__host__ __device__
float3 operator*(const float3 &lhs, const float3 &rhs);
__host__ __device__
float3 operator-(const float3 &lhs, const float3 &rhs);
__host__ __device__
void operator+=(float3 &lhs, const float3 &rhs);
__host__ __device__
void operator-=(float3 &lhs, const float3 &rhs);
__host__ __device__
float dot(float3 a, float3 b);
__host__ __device__
float distance(float3 pt1, float3 pt2, float3 *dr = NULL);
__host__ __device__ void displacement_pbc(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_x(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_y(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void displacement_pbc_z(float3 p1, float3 p2, float3 box, float* distance, float3* displacement);
__host__ __device__ void apply_pbc(float3 box, float3* p);
__host__ __device__ void apply_pbc_x(float3 box, float3* p);
__host__ __device__ void apply_pbc_y(float3 box, float3* p);
__host__ __device__ void apply_pbc_z(float3 box, float3* p);
