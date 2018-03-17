#include "../../include/SquareBox.h"


__storage__
NW::PeriodicSquareBox::PeriodicSquareBox()
{
}

__storage__
NW::PeriodicSquareBox::~PeriodicSquareBox()
{
}

// Return the function used to update postions at the boundary (Periodic etc)
NW::BoundaryFunction NW::PeriodicSquareBox::GetBoundaryFunction()
{
	return &this->boundaryFunction;
}

// Return the function used to calculate forces at the boundaries
NW::BoundaryForce NW::PeriodicSquareBox::GetBoundaryForce()
{
	return &this->boundaryForce;
}

void NW::PeriodicSquareBox::boundaryFunction(real r[_D_])
{

}

void NW::PeriodicSquareBox::boundaryForce(real r[_D_], real f[_D_])
{
	return;
}
