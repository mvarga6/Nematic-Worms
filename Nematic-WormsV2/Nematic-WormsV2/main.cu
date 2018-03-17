

#include "include/SimulationController.h"
#include "include/physics/HoegerPhysicsModel.cuh"

using namespace NW;

int main(int argc, char *argv[])
{

	SystemMetrics			* systemMetrics		= new SquareBoxMetrics();
	HoegerModelParameters	* modelParameters	= new HoegerModelParameters();
	PhysicsModel			* physicsModel		= new HoegerPhysicsModel(modelParameters, systemMetrics);
	
	
	SimulationController	* controller		= new SimulationController(physicsModel);
	
	return 0;
}