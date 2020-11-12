#include "particles.h"

int main(int argc, char *argv[])
{
	auto particles = new Particles(1000);
	particles->ZeroHost();
	particles->ToDevice();
	return 0;
}