
//#include "NWSim.h"
//#include "NWHalloweenSimulation.h"
#include "NWSimulation.h"

int main(int argc, char *argv[])
{
    //return NematicWormsSimulation(argc,argv);
	//HalloweenSimulation halloSim;
	//halloSim.Run();

	NWSimulation mySim(argc, argv);
	mySim.Run();
	return 0;
}