#include "particles.h"
#include "math.cuh"
#include <iostream>


int main(int argc, char *argv[])
{
	// auto particles = new Particles(1000);
	// particles->ZeroHost();
	// particles->ToDevice();

	std::cout << "Before copy: " << h_apply_pbc << std::endl;
	copy_device_function_symbols();
	std::cout << "After copy: " << h_apply_pbc << std::endl;
	return 0;
}