// Global Parameters for Simulation
// 5.12.15
// Mike Varga

#ifndef _PARAMS_H
#define _PARAMS_H

#include <math.h>
#include <vector>
#include <fstream>

//.. mathamatical constants
#define _2PI	6.283185307f
#define PI		3.141592654f
#define PIo2	1.570796327f
#define _216	1.12246205f

const char* _VERSION = "v1.1";

//.. landscape parameters
#define _N		8
//#define _KK		(2 * PI * _N / _XBOX)
#define _LD		1.0f

//.. driving mechanism parameters
//#define _DRIVE		 0.5f  
#define _KBIND		 200.0f
#define _MAKEBINDING 1.8f // compared to stddev of gaussian
#define _LOSSBINDING 1.2f
#define _WALKBINDING 0.000f

//.. random variations on all particles
#define _NOISE

//.. Langevin therostat
#define _DRAG

//.. dampeners for inter worm springs
#define _DAMPING1
#define _DAMPING2
#define _DAMPING

//.. turns on polarity similar attractive interactions
//#define _STICKY
#define _XCUT2		0.25f

std::ifstream load;
std::ofstream routput;
const std::string xyzfile = "nwsim";
const std::string xyz = ".xyz";
const std::string xyzv = ".xyzv";
const std::string xyzc = ".xyzc";
const std::string csv = ".csv";
const std::string dat = ".dat";
const std::string datafolder = "./app_data/";

#endif