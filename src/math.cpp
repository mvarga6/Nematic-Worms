#include "math.cuh"


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
