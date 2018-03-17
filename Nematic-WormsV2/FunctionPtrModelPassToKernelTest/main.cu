
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//////////////////////////////
// Function pointer types ////
//////////////////////////////

typedef void(*InitFunction)(float &);
typedef float(*OperationType1)(float, float);
typedef float(*OperationType2)(float, float&);

////////////////////////////////////////////////
// The Kernel we're sending the function ptrs //
////////////////////////////////////////////////
__global__ void ExecuteModelKernel(float *inA, float *inB, float *out, int N,
	InitFunction init,
	OperationType1 op1,
	OperationType2 op2)
{
	int id = threadIdx.x + blockDim.x * blockIdx.x;
	if (id < N)
	{
		// Call init function
		if (init != NULL)
		{
			init(out[id]);
			inA[id] = 1.0f;
			inB[id] = 2.0f;
		}
			

		// Operation 1
		if (op1 != NULL)
		{
			out[id] = op1(inA[id], inB[id]);
		}

		// Operation 2
		if (op2 != NULL)
		{
			out[id] = op2(inA[id], inB[id]);
		}
			

		printf("%d: %f\n", id, out[id]);
	}
}

/////////////////////////////////
// The functions to point to ////
/////////////////////////////////

// InitFunction methods 
__device__
void Zero(float &a)
{
	a = 0;
}

__device__
void Unity(float &a)
{
	a = 1;
}
__device__
void Ignore(float &a)
{
}

__device__ InitFunction dZero = Zero;
__device__ InitFunction dUnity = Unity;
__device__ InitFunction dIgnore = Ignore;

// OperationType1 methods 
__device__
float Add(float _a, float _b)
{
	return _a + _b;
}

__device__
float Subtract(float a, float b)
{
	return a - b;
}

__device__
float Times(float a, float b)
{
	return a * b;
}

__device__ OperationType1 dAdd = Add;
__device__ OperationType1 dSubtract = Subtract;
__device__ OperationType1 dTimes = Times;

// OperationType2 methods
__device__
float AddAndTimes(float a, float &b)
{
	float result = a + b;
	b *= a;
	return result;
}

__device__
float SubtractAndDivide(float a, float &b)
{
	float result = a - b;
	b = a / b;
	return result;
}

__device__
float TimesAndDivide(float a, float &b)
{
	float result = a * b;
	b = a / b;
	return result;
}

__device__ OperationType2 dAddAndTimes = AddAndTimes;
__device__ OperationType2 dSubtractAndDivide = SubtractAndDivide;
__device__ OperationType2 dTimesAndDivide = TimesAndDivide;

///////////////////////////////////////////////////
// A Model that stores pointers to these methods //
// do define runtime behavior with a model ////////
///////////////////////////////////////////////////

// Abstract parent
// lives on host always
class BaseFunctionModel
{
public:
	InitFunction Init;
	OperationType1 Op1;
	OperationType2 Op2;

	//__host__ __device__
	//virtual ~BaseFunctionModel() = 0;
};

// A specific model 
class FunctionModelA : public BaseFunctionModel
{
public:
	//__host__ __device__
	FunctionModelA()
	{
		// assign the functions that define 
		// the model in the constructor
		cudaMemcpyFromSymbol(&Init, dUnity, sizeof(InitFunction));
		cudaMemcpyFromSymbol(&Op1, dAdd, sizeof(OperationType1));
		cudaMemcpyFromSymbol(&Op2, dTimesAndDivide, sizeof(OperationType2));
	}

	//__host__ __device__
	~FunctionModelA()
	{
		Init = NULL;
		Op1 = NULL;
		Op2 = NULL;
	}
};

// Another different specific model 
class FunctionModelB : public BaseFunctionModel
{
public:
	//__host__ __device__
	FunctionModelB()
	{
		// assign the functions that define 
		// the model in the constructor
		cudaMemcpyFromSymbol(&Init, dUnity, sizeof(InitFunction));
		cudaMemcpyFromSymbol(&Op1, dSubtract, sizeof(OperationType1));
		cudaMemcpyFromSymbol(&Op2, dSubtractAndDivide, sizeof(OperationType2));
	}

	//__host__ __device__
	~FunctionModelB()
	{
		Init = NULL;
		Op1 = NULL;
		Op2 = NULL;
	}
};

///////////////////////////////////////
// The Data Model that lives on host //
// and wraps function pointers       //
///////////////////////////////////////

class DataModel
{
public:
	int N;
	float *a;
	float *b;
	float *c;

	//__host__
	DataModel(int N)
	{
		this->N = N;
	}

	//__host__
	void AllocGpu()
	{
		size_t size = sizeof(float) * N;
		cudaMalloc((void**)&a, size);
		cudaMalloc((void**)&b, size);
		cudaMalloc((void**)&c, size);
		cudaMemset(a, 1, size);
		cudaMemset(b, 1, size);
		cudaMemset(c, 1, size);
	}
};



cudaError_t ExecuteModel(DataModel *data, BaseFunctionModel *model);

int main()
{
	int L = 16;
	 
	// get the data on the gpu with data model pointing to it
	DataModel *dataA = new DataModel(L);
	DataModel *dataB = new DataModel(L);
	dataA->AllocGpu();
	dataB->AllocGpu();

	// define the behavior you'd like
	BaseFunctionModel *modelA = new FunctionModelA();
	BaseFunctionModel *modelB = new FunctionModelB();

	// execute with same kernel but receie different results
	cudaError errA = ExecuteModel(dataA, modelA);
	cudaError errB = ExecuteModel(dataB, modelB);

	printf("%s: %s\n%s: %s\n", cudaGetErrorName(errA)
		,cudaGetErrorString(errA) 
		,cudaGetErrorName(errB)
		,cudaGetErrorString(errB)
	);

	// copy the data back
	size_t size = sizeof(float) * L;
	float *aA = new float[L];
	float *bA = new float[L];
	float *cA = new float[L];

	cudaMemcpy(aA, dataA->a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(bA, dataA->b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cA, dataA->c, size, cudaMemcpyDeviceToHost);

	// copy the data back
	float *aB = new float[L];
	float *bB = new float[L];
	float *cB = new float[L];
	cudaMemcpy(aB, dataB->a, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(bB, dataB->b, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(cB, dataB->c, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < L; i++)
	{
		printf("A: %f %f %f \t B: %f %f %f\n", aA[i], bA[i], cA[i], aB[i], bB[i], cB[i]);
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t ExecuteModel(DataModel *data, BaseFunctionModel *model)
{
	ExecuteModelKernel <<< 4, data->N / 4 >>>
		(
			data->a,
			data->b,
			data->c,
			data->N,
			model->Init,
			model->Op1,
			NULL
			);

	return cudaGetLastError();
}
