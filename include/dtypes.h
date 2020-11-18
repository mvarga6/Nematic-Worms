#pragma once

#include <cuda.h>

typedef float3 ptype;

using PositionFunction = void (*) (float3, float3*);
using DistanceFunction = void (*) (float3, float3, float3, float*, float3*);