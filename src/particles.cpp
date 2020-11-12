#include "particles.h"


Particles::Particles(uint n)
{
    this->count = n;
    this->alloc_size = sizeof(ptype) * n;
    this->r = new ptype[n];
    this->v = new ptype[n];
    this->f = new ptype[n];
    cudaMalloc((void**)&(this->dev_r), this->alloc_size);
    cudaMalloc((void**)&(this->dev_v), this->alloc_size);
    cudaMalloc((void**)&(this->dev_f), this->alloc_size);
}


Particles::~Particles()
{
    delete[] this->r, this->v, this->f;
    cudaFree(this->dev_r);
    cudaFree(this->dev_v);
    cudaFree(this->dev_f);
}


void Particles::Zero()
{
    this->ZeroDevice();
    this->ZeroHost();
}


void Particles::ZeroHost()
{
    memset(this->r, 0, this->alloc_size);
    memset(this->v, 0, this->alloc_size);
    memset(this->f, 0, this->alloc_size);
}


void Particles::ZeroDevice()
{
    cudaMemset((void**)this->dev_r, 0, this->alloc_size);
    cudaMemset((void**)this->dev_v, 0, this->alloc_size);
    cudaMemset((void**)this->dev_f, 0, this->alloc_size);
}


void Particles::ToDevice(bool onlyParticles)
{
	cudaMemcpy(this->dev_r, this->r, this->alloc_size, cudaMemcpyHostToDevice);
    if (!onlyParticles)
    {
        cudaMemcpy(this->dev_v, this->r, this->alloc_size, cudaMemcpyHostToDevice);
        cudaMemcpy(this->dev_f, this->r, this->alloc_size, cudaMemcpyHostToDevice);
    }
}


void Particles::ToHost(bool onlyParticles)
{
	cudaMemcpy(this->r, this->dev_r, this->alloc_size, cudaMemcpyDeviceToHost);
    if (!onlyParticles)
    {
        cudaMemcpy(this->v, this->dev_v, this->alloc_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(this->f, this->dev_f, this->alloc_size, cudaMemcpyDeviceToHost);
    }
}