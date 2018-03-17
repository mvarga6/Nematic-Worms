#pragma once

// ---------------------------------
// Dimensions of simulation space
// and thus requires recompilation
// when switched from 2 to 3 or
// vice versa !!!
// ---------------------------------
#define _D_ 2
#define for_D_ for(int d = 0; d < _D_; d++)
// ---------------------------------
// The Current version of sim
// ---------------------------------
#define __NW_VERSION__ 2.1
#define real float
//----------------------------------
// Uncomment to print debugging
// information
//
// #define __DEBUG__
// 
//----------------------------------
// for cuda storage classes
#define __storage__ __host__ __device__