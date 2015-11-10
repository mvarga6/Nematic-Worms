// Functions for GPU Device
// 5.12.15
// Mike Varga

#ifndef _KERNELS_H
#define _KERNELS_H

#include "cuda_runtime.h"

/*
*	Global scope function to print the parameters defined in the simulation
*	based on what files have been included and if they've been memcpy-ed t0
*	the device.
*/
__global__ void CheckParametersOnDevice(){
#ifdef __SIMULATION_PARAMETERS_H__
	printf("\n\tSimulation Parameters:\n");
	printf("\t----------------------\n");
	printf("\tXBOX = %f\n\tYBOX = %f\n", dev_simParams._XBOX, dev_simParams._YBOX);
	printf("\tDT = %f\n\tNSTEPS = %i\n", dev_simParams._DT, dev_simParams._NSTEPS);
	printf("\tPRATE = %i\n\tMAXFILE = %i\n", dev_simParams._FRAMERATE, dev_simParams._FRAMESPERFILE);
#endif

#ifdef __WORMS_PARAMETERS_H__
	printf("\n\tWorms Parameters:\n");
	printf("\t-----------------\n");
	printf("\tNP = %i\n\tNWORMS = %i\n", dev_Params._NP, dev_Params._NWORMS);
	printf("\tNPARTICLES = %i\n", dev_Params._NPARTICLES);
	printf("\tLISTSETGAP = %i\n\tNMAX = %i\n", dev_Params._LISTSETGAP, dev_Params._NMAX);
	printf("\tEPSILON = %f\n\tSIGMA = %f\n", dev_Params._EPSILON, dev_Params._SIGMA);
	printf("\tDRIVE = %f\n", dev_Params._DRIVE);
	printf("\tK1 = %f\n\tK2 = %f\n\tK3 = %f\n", dev_Params._K1, dev_Params._K2, dev_Params._K3);
	printf("\tL1 = %f\n\tL2 = %f\n\tL3 = %f\n", dev_Params._L1, dev_Params._L2, dev_Params._L3);
	printf("\tKBT = %f\n\tGAMMA = %f\n\tDAMP = %f\n", dev_Params._KBT, dev_Params._GAMMA, dev_Params._DAMP);
	printf("\tSIG6 = %f\n\t2SIG6 = %f\n\tLJAMP = %f\n", dev_Params._SIGMA6, dev_Params._2SIGMA6, dev_Params._LJ_AMP);
	printf("\tRMIN = %f\n\tR2MIN = %f\n", dev_Params._RMIN, dev_Params._R2MIN);
	printf("\tRCUT = %f\n\tR2CUT = %f\n", dev_Params._RCUT, dev_Params._R2CUT);
	printf("\tBUFFER = %f\n", dev_Params._BUFFER);
#endif

#ifdef __FLUID_PARAMETERS_H__
	printf("\n\tFluid Parameters:\n");
	printf("\t-----------------\n");
	printf("\tNFLUID = %i\n", dev_flParams._NFLUID);
	printf("\tLISTSETGAP = %i\n\tNMAX = %i\n", dev_flParams._LISTSETGAP, dev_flParams._NMAX);
	printf("\tEPSILON = %f\n\tSIGMA = %f\n", dev_flParams._EPSILON, dev_flParams._SIGMA);
	printf("\tKBT = %f\n\tGAMMA = %f\n", dev_flParams._KBT, dev_flParams._GAMMA);
	printf("\tSIG6 = %f\n\t2SIG6 = %f\n\tLJAMP = %f\n", dev_flParams._SIGMA6, dev_flParams._2SIGMA6, dev_flParams._LJ_AMP);
	printf("\tRMIN = %f\n\tR2MIN = %f\n", dev_flParams._RMIN, dev_flParams._R2MIN);
	printf("\tRCUT = %f\n\tR2CUT = %f\n", dev_flParams._RCUT, dev_flParams._R2CUT);
	printf("\tBUFFER = %f\n", dev_flParams._BUFFER);
#endif
}

#endif