/*
 * Pattern analysis applied to example from http://ppc.cs.aalto.fi/ch4/v0/
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "pattern_recorder.cuh"

#define kernel_v0
//#define kernel_v1

#if defined(kernel_v0)
const char* patterns_out_path = "access-patterns-v0.json";
#elif defined(kernel_v1)
const char* patterns_out_path = "access-patterns-v1.json";
#endif

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

#if defined(kernel_v0)
template <class KernelData>
__global__
void mykernel(float* r, KernelData d, int n) {
	d.enter_kernel();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= n || j >= n)
		return;
	float v = HUGE_VALF;
	for (int k = 0; k < n; ++k) {
		float x = d[n*i + k];
		float y = d[n*k + j];
		float z = x + y;
		v = min(v, z);
	}
	r[n*i + j] = v;
}
#elif defined(kernel_v1)
template <class KernelData>
__global__
void mykernel(float* r, KernelData d, int n) {
	d.enter_kernel();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (i >= n || j >= n)
		return;
	float v = HUGE_VALF;
	for (int k = 0; k < n; ++k) {
		float x = d[n*j + k];
		float y = d[n*k + i];
		float z = x + y;
		v = min(v, z);
	}
	r[n*j + i] = v;
}
#endif

inline int static divup(int a, int b) {
	return (a + b - 1) / b;
}


void step(float* r, const float* d, int n) {
	// Allocate memory & copy data to GPU
	float* dGPU = NULL;
	CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
	float* rGPU = NULL;
	CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Run kernel
	dim3 dimBlock(32, 32); // This was increased to allocate less SMs
	dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));

	// We will be making 2 passes over the data by calling mykernel twice
	// 1. Compute amount of required space to store all accesses
	// 2. Record all accesses and timestamps
	unsigned long long max_access_count = 0u;
	{
		// Compute the maximum amount of accesses a thread block will make
		pr::AccessCounter<float> counter(dGPU, dimGrid);
		mykernel<pr::AccessCounter<float> ><<<dimGrid, dimBlock>>>(rGPU, counter, n);
		// Sync is required since the counter is using CUDA managed memory which does not require explicit device-to-host memcpy
		CHECK(cudaDeviceSynchronize());
		// Show a small summary of memory access counts (optional)
		counter.dump_access_statistics(std::cout);
		// For the next step, get the maximum number of memory accesses made by a block
		max_access_count = counter.get_max_access_count();
	}
	{
		// Allocate device memory for all possible accesses and record the actual access pattern
		pr::PatternRecorder<float> recorder(dGPU, dimGrid, max_access_count);
		mykernel<pr::PatternRecorder<float> ><<<dimGrid, dimBlock>>>(rGPU, recorder, n);
		// Again, sync to make sure the kernel call has been finished
		CHECK(cudaDeviceSynchronize());
		// Write results as JSON somewhere and specify number of rows and columns in the array that is being accessed
		std::ofstream outf(patterns_out_path);
		recorder.dump_json_results(outf, n, n);
		std::cout << "Wrote " << patterns_out_path << "\n";
	}

	CHECK(cudaGetLastError());

	// Copy data back to CPU & release memory
	CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK(cudaFree(dGPU));
	CHECK(cudaFree(rGPU));
}


__host__
float next_float() {
	static std::random_device rd;
	static std::default_random_engine e(rd());
	static std::uniform_real_distribution<float> floats(0.0, 1.0);
	return floats(e);
}


__host__
int main() {
	// Generate data
	int n = 64;
	std::vector<float> matrix(n * n);
	std::generate(matrix.begin(), matrix.end(), next_float);
	std::vector<float> result(n * n);
	// Compute stuff
	step(result.data(), matrix.data(), n);
	// Write dummy output
	std::ofstream outf("/dev/null");
	std::copy(result.begin(), result.end(), std::ostream_iterator<float>(outf, " "));
}
