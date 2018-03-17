#include "../../../include/physics/HoegerPhysicsModel.cuh"
#include "../../../include/physics/HoegerPhysicsKernels.cuh"
#include "../../../include_main.h"


NW::HoegerPhysicsModel::HoegerPhysicsModel(HoegerModelParameters * parameters, SystemMetrics *metrics, BoundaryConditionService *boundaryService)
{
	this->parameters = parameters;
	this->metrics = metrics;
	this->boundary = boundaryService;
}

NW::HoegerPhysicsModel::~HoegerPhysicsModel()
{
}

void NW::HoegerPhysicsModel::InternalForces(Particles* particles, Filaments *filaments)
{
	NW::HoegerInternalForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f, 
			particles->v, 
			particles->r, 
			particles->N, 
			filaments->N,
			filaments->GlobalProperties->Kbond,
			filaments->GlobalProperties->Lbond,
			metrics->GetDistanceMetric(),
			intrafilament_interaction
		);
}



void NW::HoegerPhysicsModel::ThermalForces(Particles* particles, Filaments *filaments, float kBT)
{
	real *gpu_rn = random->Get(particles->N * _D_, false);

	// TODO: calculate thermal scaler
	real thermal_scaler = kBT;

	NW::HoegerThermalForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f,
			particles->N,
			gpu_rn,
			thermal_scaler	
		);
}



void NW::HoegerPhysicsModel::ParticleParticleForces(Particles* particles, Filaments *filaments, NeighborsGraph* neighbors)
{
	NW::HoegerParticleParticleForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f,
			particles->r,
			particles->N,
			filaments->N,
			neighbors->List,
			neighbors->Max,
			metrics->GetDistanceMetric(),
			particle_particle_interaction
		);
}



void NW::HoegerPhysicsModel::ActiveForces(Particles* particles, Filaments *filaments)
{
	NW::HoegerActiveForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f,
			particles->r, 
			particles->N,
			filaments->N,
			filaments->Activity,
			metrics->GetDistanceMetric()
		);
}



void NW::HoegerPhysicsModel::ExternalForces(Particles* particles, Filaments *filaments, Environment *environment)
{
	NW::HoegerExternalForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f, 
			particles->v,
			particles->N,
			environment->FluidDrag
		);
}



void NW::HoegerPhysicsModel::ContainerForces(Particles* particles, Filaments *filaments)
{
	NW::HoegerContainerForces<<<launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f,
			particles->r,
			particles->N,
			boundary->GetBoundaryForce()
		);
}



void NW::HoegerPhysicsModel::BondBendingForces(Particles* particles, Filaments *filaments)
{
	NW::HoegerBondBendingForces<<<launchInfo->BodiesGridStructure, launchInfo->BodiesBlockStructure >>>
		(
			particles->f, 
			particles->r,
			particles->N,
			filaments->N,
			filaments->GlobalProperties->Np,
			filaments->GlobalProperties->Kbend,
			metrics->GetDisplacementMetric()
		);
}



void NW::HoegerPhysicsModel::UpdateSystem(Particles* particles, real dt)
{
	NW::HoegerUpdateSystem<<< launchInfo->ParticlesGridStructure, launchInfo->ParticlesBlockStructure >>>
		(
			particles->f,
			particles->f_old,
			particles->v,
			particles->r,
			particles->N,
			boundary->GetBoundaryFunction(),
			dt
		);
}



//void NW::HoegerPhysicsModel::UpdateSystemFast(Particles* particles, Environment *environment, real dt)
//{
//	NW::HoegerPhysics::__UpdateSystemFast
//		<<<launchInfo->ParticleGridStructure, launchInfo->ParticleBlockStructure >>>
//		(f, fshift, f_old, foshift, v, vshift, r, rshift, cell, cshift, dt);
//}
//
