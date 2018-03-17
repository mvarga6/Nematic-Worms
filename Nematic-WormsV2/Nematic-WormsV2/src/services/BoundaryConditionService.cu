#include "../../include/BoundaryConditionService.h"


__storage__ 
NW::BoundaryConditionService::BoundaryConditionService(BoundaryConditions boundaryConditions)
{
	this->boundaryConditions = boundaryConditions;

	// chose the shape function
	switch (boundaryConditions.Shape)
	{
	case BoundaryShape::Square:
		this->boundaryFunction = this->SquareBC;
		this->boundaryForce = this->SquareBF;
		break;

	case BoundaryShape::Circular:
		this->boundaryFunction = this->CircularBC;
		this->boundaryForce = this->CircularBF;
		break;

	case BoundaryShape::Trapozoidal:
		this->boundaryFunction = this->TrapBC;
		this->boundaryForce = this->TrapBF;
	}

	// chose the type functions inside
	for_D_
	{
		switch (boundaryConditions.Type[d])
		{
		case BoundaryConditionType::Periodic:

			break;

		case BoundaryConditionType::SoftWalls:

			break;

		case BoundaryConditionType::HardWalls:

			break;

		case BoundaryConditionType::Open:

			break;
		}
	}

	// put values on gpu
	cudaError err = cudaMemcpyToSymbol(&NW::Internal::BC, &boundaryConditions, sizeof(BoundaryConditions));

	// TODO: Handle error
}

__storage__
NW::BoundaryFunction NW::BoundaryConditionService::GetBoundaryFunction()
{
	return this->boundaryFunction;
}

__storage__
NW::BoundaryForce NW::BoundaryConditionService::GetBoundaryForce()
{
	return this->boundaryForce;
}

__storage__ 
void NW::BoundaryConditionService::SquareBC(real r[_D_])
{
	using namespace NW;
	for_D_ // for each dimension
	{
		this->boundaryFunctions[d](r[d]);
	}

		//switch (Internal::BC.Type[d]) // implement the type of BC
		//{
		//case BoundaryConditionType::Periodic:
		//	if (r[d] > Internal::BC.Specs[d]) r[d] -= Internal::BC.Specs[d];
		//	else if (r[d] < 0) r[d] += Internal::BC.Specs[d];
		//	break;

		//case BoundaryConditionType::SoftWalls:
		//	// TODO: soft wall boundary conditions
		//	break;

		//case BoundaryConditionType::Open:

		//	break;

		//case BoundaryConditionType::HardWalls:
		//	// TODO: hard wall boundary conditions
		//	break;
		//}
	//}
}

__storage__ 
void NW::BoundaryConditionService::CircularBC(real r[_D_])
{

}

__storage__ 
void NW::BoundaryConditionService::TrapBC(real r[_D_])
{

}

__storage__
void NW::BoundaryConditionService::SquareBF(real r[_D_], real f[_D_])
{
	using namespace NW;
	for_D_ // for each dimension
	{
		switch (Internal::BC.Type[d]) // implement the type of BC
		{
		case BoundaryConditionType::Periodic:
			break;

		case BoundaryConditionType::SoftWalls:
			// TODO: soft wall boundary conditions
			break;

		case BoundaryConditionType::Open:

			break;

		case BoundaryConditionType::HardWalls:
			// TODO: hard wall boundary conditions
			break;
		}
	}
}

__storage__ 
void NW::BoundaryConditionService::CircularBF(real r[_D_], real f[_D_])
{

}

__storage__ 
void NW::BoundaryConditionService::TrapBF(real r[_D_], real f[_D_])
{

}