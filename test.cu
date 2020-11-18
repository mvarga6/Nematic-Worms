#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

typedef float (* fp)(float, float, float4);

struct functor
{
    float c0, c1;
    fp f;

    __device__ __host__
    functor(float _c0, float _c1, fp _f) : c0(_c0), c1(_c1), f(_f) {};

    __device__ __host__
    float operator()(float4 x) { return f(c0, c1, x); };
};

__global__
void kernel(float c0, float c1, fp f, const float4 * x, float * y, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    struct functor op(c0, c1, f);
    for(int i = tid; i < N; i  = blockDim.x * gridDim.x) {
        y[i] = op(x[i]);
    }
}

__device__ __host__
float f1 (float a, float b, float4 c)
{
    return a  + (b * c.x) + (b * c.y) + (b * c.z) + (b * c.w);
}

__device__ __host__
float f2 (float a, float b, float4 c)
{
    return a + b + c.x + c.y + c.z + c.w;
}

__constant__ fp function_table[] = {f1, f2};

int main(void)
{
    cudaSetDevice(1);

    const float c1 = 1.0f, c2 = 2.0f;
    const int n = 20;
    float4 vin[n];
    float vout1[n], vout2[n];
    for(int i=0, j=0; i<n; i++ ) {
        vin[i].x = j  ; vin[i].y = j  ;
        vin[i].z = j  ; vin[i].w = j  ;
    }

    float4 * _vin;
    float * _vout1, * _vout2;
    size_t sz4 = sizeof(float4) * size_t(n);
    size_t sz1 = sizeof(float) * size_t(n);
    cudaMalloc((void **)&_vin, sz4);
    cudaMalloc((void **)&_vout1, sz1);
    cudaMalloc((void **)&_vout2, sz1);
    cudaMemcpy(_vin, &vin[0], sz4, cudaMemcpyHostToDevice);

    fp funcs[2];
    cudaMemcpyFromSymbol(&funcs, "function_table", 2 * sizeof(fp));

    kernel<<<1,32>>>(c1, c2, funcs[0], _vin, _vout1, n);
    cudaMemcpy(&vout1[0], _vout1, sz1, cudaMemcpyDeviceToHost);

    kernel<<<1,32>>>(c1, c2, funcs[1], _vin, _vout2, n);
    cudaMemcpy(&vout2[0], _vout2, sz1, cudaMemcpyDeviceToHost);

    struct functor func1(c1, c2, f1), func2(c1, c2, f2);
    for(int i=0; i<n; i++) {
        printf("- %6.f %6.f (%6.f,%6.f,%6.f,%6.f ) %6.f %6.f %6.f %6.f\n",
                i, c1, c2, vin[i].x, vin[i].y, vin[i].z, vin[i].w,
                vout1[i], func1(vin[i]), vout2[i], func2(vin[i]));
    }

    return 0;
}