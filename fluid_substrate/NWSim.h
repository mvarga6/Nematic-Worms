/*******************************
Active Flexible Worms Simulation
********************************/
#ifndef _NWSIM_H
#define _NWSIM_H

#include "NWFluid.h"
#include "NWWorms.h"
#include "NWParams.h"
#include "NWForceXchanger.h"
#include "NWRandomNumberGenerator.h"
//#include "XYZPrinter.h"
#include "NWHostFunctions.h"

using namespace std;

/*
*	Does not work becuase the amount of memory required to do all the random numbers
*	causes errors.  Need to find a way to allocate random numbers segment by segment 
*	and give them to any objects that need them.  Possibly make an object to be referenced
*	on creation of Worms, Fluid, and ForceXchanger that hands off a device ptr when random
*	numbers are needed.  Also, the size of xlist only needs to be as big as _NPARTICLES,
*	do not need to store mostly -1's for all fluid particles.
*/
int NematicWormsSimulation(int argc, char *argv[])
{
	//.. get the shit right
	if (ProcessCommandLine(argc, argv) != 0)
		return 10;

	//.. save parameters for simulation
	SaveSimulationConfiguration();

	//.. random number generator
	GRNG randnumgen(2 * _NFLUID, 0.0f, 1.0f);

	//.. the physical stuff
	Worms worms(5, 256, randnumgen);
	Fluid fluid(32, 512, randnumgen);

	//.. means of force exchange between sets
	ForceXchanger xchanger(worms, fluid, randnumgen);

	worms.Init();
	fluid.Init();
	xchanger.Init();

	worms.ResetNeighborsList();
	fluid.ResetNeighborsList();
	printf("\n\t-------Init--------");
	DisplayErrors(worms, fluid, xchanger, randnumgen);

	PrintXYZ(worms.X, worms.Y, fluid.X, fluid.Y, xchanger.GetXList());

	/////////////////////////////////////////////////////////////////
	//////////////////// SIMULATION TIME LOOP ///////////////////////
	/////////////////////////////////////////////////////////////////
	
	int frame = 0;
	std::clock_t start_t = std::clock();
	std::clock_t intra_t = 0.0;
	std::clock_t inter_t = 0.0;
	
	for (int itime = 0; itime < _NSTEPS; itime++){

		//.. Error Handling
		//ErrorHandler(cudaGetLastError());
		printf("\n\t-------Begin--------");
		ErrorHandler(cudaGetLastError());
		DisplayErrors(worms, fluid, xchanger, randnumgen);

		//.. start clock
		std::clock_t intra_start_t = std::clock();

		//.. calculate internal forces
		worms.InternalForces();
		fluid.InternalForces();

		printf("\n\t-------Internal--------");
		ErrorHandler(cudaGetLastError());
		DisplayErrors(worms, fluid, xchanger, randnumgen);

		//.. set neighbor lists
		if (itime % _LISTSETGAP == 0)
		{
			//.. reset neighbor list for N2
			worms.ResetNeighborsList();
			fluid.ResetNeighborsList();
			xchanger.UpdateXList();

			printf("\n\t-------SetNlist--------");
			ErrorHandler(cudaGetLastError());
			DisplayErrors(worms, fluid, xchanger, randnumgen);
		}

		//.. clock midpoint
		std::clock_t midpoint = std::clock();
		intra_t += (midpoint - intra_start_t);
		std::clock_t inter_start_t = midpoint;

		//.. calculate lennard jones forces
		worms.LJForces();
		fluid.LJForces();

		printf("\n\t--------LJ---------");
		ErrorHandler(cudaGetLastError());
		DisplayErrors(worms, fluid, xchanger, randnumgen);

		//.. calculate xchange forces
		xchanger.XchangeForces();

		printf("\n\t-------Xchange--------");
		ErrorHandler(cudaGetLastError());
		DisplayErrors(worms, fluid, xchanger, randnumgen);

		inter_t += (std::clock() - inter_start_t);
		
		//cudaDeviceSynchronize();
		worms.Update();
		fluid.Update();

		printf("\n\t-------Update--------");
		ErrorHandler(cudaGetLastError());
		DisplayErrors(worms, fluid, xchanger, randnumgen);

		// Frame calculations
		if (itime%_FRAMERATE == 0)
		{
			worms.DataDeviceToHost();
			fluid.DataDeviceToHost();
			xchanger.XListToHost();

			printf("\n\t-------Print--------");
			ErrorHandler(cudaGetLastError());
			DisplayErrors(worms, fluid, xchanger, randnumgen);
			
			PrintXYZ(worms.X, worms.Y, fluid.X, fluid.Y, xchanger.GetXList());
			ClockWorks(frame, start_t, intra_t, inter_t);
		}
	}
	return EXIT_SUCCESS;
}

#endif