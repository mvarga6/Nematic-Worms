#include "math.cuh"
#include "io.h"


__host__ __device__
float3 operator*(const float &lhs, const float3 &rhs)
{
    return make_float3(lhs*rhs.x, lhs*rhs.y, lhs*rhs.z);
}

__host__ __device__
float3 operator*(const float3 &lhs, const float3 &rhs)
{
    return make_float3(lhs.x*rhs.x, lhs.y*rhs.y, lhs.z*rhs.z);
}

__host__ __device__
float3 operator-(const float3 &lhs, const float3 &rhs)
{
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

__host__ __device__
void operator+=(float3 &lhs, const float3 &rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
}

__host__ __device__
void operator-=(float3 &lhs, const float3 &rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
}

__host__ __device__
float dot(float3 a, float3 b)
{
  return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__
float distance(float3 pt1, float3 pt2, float3 *dr)
{
    float3 v = pt2 - pt1;
    if (dr != NULL) *dr = v;
    return sqrt(dot(v,v));
}

__host__ __device__ void displacement_pbc(float3 p1, float3 p2, float3 box, float* distance, float3* displacement)
{
    float3 dr = p2 - p1;
    if (dr.x > box.x / 2.0f) dr.x -= box.x;
    else if (dr.x < -box.x / 2.0f) dr.x += box.x;
    if (dr.y > box.y / 2.0f) dr.y -= box.y;
    else if (dr.y < -box.y / 2.0f) dr.y += box.y;
    if (dr.z > box.z / 2.0f) dr.z -= box.z;
    else if (dr.z < -box.z / 2.0f) dr.z += box.z;
    (*displacement) = dr;
    (*distance) = sqrt(dot(dr, dr));
}

__host__ __device__ void displacement_pbc_x(float3 p1, float3 p2, float3 box, float* distance, float3* displacement)
{
    float3 dr = p2 - p1;
    if (dr.x > box.x / 2.0f) dr.x -= box.x;
    else if (dr.x < -box.x / 2.0f) dr.x += box.x;
    (*displacement) = dr;
    (*distance) = sqrt(dot(dr, dr));
}

__host__ __device__ void displacement_pbc_y(float3 p1, float3 p2, float3 box, float* distance, float3* displacement)
{
    float3 dr = p2 - p1;
    if (dr.y > box.y / 2.0f) dr.y -= box.y;
    else if (dr.y < -box.y / 2.0f) dr.y += box.y;
    (*displacement) = dr;
    (*distance) = sqrt(dot(dr, dr));
}

__host__ __device__ void displacement_pbc_z(float3 p1, float3 p2, float3 box, float* distance, float3* displacement)
{
    float3 dr = p2 - p1;
    if (dr.z > box.z / 2.0f) dr.z -= box.z;
    else if (dr.z < -box.z / 2.0f) dr.z += box.z;
    (*displacement) = dr;
    (*distance) = sqrt(dot(dr, dr));
}

__host__ __device__ void apply_pbc(float3 box, float3* p)
{
    if (p->x > box.x) p->x -= box.x;
    else if (p->x < 0.0f) p->x += box.x;
    if (p->y > box.y) p->y -= box.y;
    else if (p->y < 0.0f) p->y += box.y;
    if (p->z > box.z) p->z -= box.z;
    else if (p->z < 0.0f) p->z += box.z;
}

__host__ __device__ void apply_pbc_x(float3 box, float3* p)
{
    if (p->x > box.x) p->x -= box.x;
    else if (p->x < 0.0f) p->x += box.x;
}

__host__ __device__ void apply_pbc_y(float3 box, float3* p)
{
    if (p->y > box.y) p->y -= box.y;
    else if (p->y < 0.0f) p->y += box.y;
}

__host__ __device__ void apply_pbc_z(float3 box, float3* p)
{
    if (p->z > box.z) p->z -= box.z;
    else if (p->z < 0.0f) p->z += box.z;
}


__device__ PositionFunction p_apply_pbc   = apply_pbc;
__device__ PositionFunction p_apply_pbc_x = apply_pbc_x;
__device__ PositionFunction p_apply_pbc_y = apply_pbc_y;
__device__ PositionFunction p_apply_pbc_z = apply_pbc_z;

__device__ DistanceFunction p_displacement_pbc   = displacement_pbc;
__device__ DistanceFunction p_displacement_pbc_x = displacement_pbc_x;
__device__ DistanceFunction p_displacement_pbc_y = displacement_pbc_y;
__device__ DistanceFunction p_displacement_pbc_z = displacement_pbc_z;

PositionFunction h_apply_pbc;
PositionFunction h_apply_pbc_x;
PositionFunction h_apply_pbc_y;
PositionFunction h_apply_pbc_z;

#include <iostream>

__host__ void copy_device_function_symbols()
{
    std::cout << "Size of PositionFunction: " << sizeof(PositionFunction) << std::endl;
    ShowError(cudaMemcpyFromSymbol(&h_apply_pbc, p_apply_pbc, sizeof(PositionFunction)));
    ShowError(cudaMemcpyFromSymbol(&h_apply_pbc_x, p_apply_pbc_x, sizeof(PositionFunction)));
    ShowError(cudaMemcpyFromSymbol(&h_apply_pbc_y, p_apply_pbc_y, sizeof(PositionFunction)));
    ShowError(cudaMemcpyFromSymbol(&h_apply_pbc_z, p_apply_pbc_z, sizeof(PositionFunction)));
}