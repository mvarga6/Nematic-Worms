#pragma once

#include <cuda.h>
#include <cuda_runtime.h>



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
float distance(float3 pt1, float3 pt2, float3 *dr = NULL)
{
    float3 v = pt2 - pt1;
    if (dr != NULL) *dr = v;
    return sqrt(dot(v,v));
}