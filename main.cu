#include "particles.h"
#include "math.cuh"
#include "kernels.h"
#include <iostream>


int main(int argc, char *argv[])
{
	SetupDeviceFunctionTables();

	auto box = make_float3(10.0f, 10.0f, 10.0f);
	auto particles = new Particles(10);
	particles->ZeroHost();
	particles->r[0].x = 5.0f;
	particles->r[0].y = 5.0f;
	particles->r[0].z = 11.0f;
	particles->ToDevice();

	auto pbc = GetPositionFunction(0);

	ApplyPositionFunction(pbc, particles, box, 128);

	particles->ToHost();
	std::cout << particles->r[0].x << " " << particles->r[0].y << " " << particles->r[0].z << std::endl;

	return 0;
}