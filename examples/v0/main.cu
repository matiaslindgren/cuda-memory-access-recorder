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

// When counting number of accesses
/* using Wrapper = pr::AccessCounter<float>; */
// When recording access patterns and timings
using Wrapper = pr::PatternRecorder<float>;

const std::string patterns_out_path("access-patterns.txt");
const std::string results_out_path("/dev/null");


__global__
void mykernel(float* r, Wrapper d, int n) {
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


inline int static divup(int a, int b) {
	return (a + b - 1) / b;
}


void step(float* r, const float* d, int n) {
	// Allocate memory & copy data to GPU
	float* dGPU = NULL;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
	float* rGPU = NULL;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Run kernel
	dim3 dimBlock(16, 16);
	dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));

#if 0
	{
		// Count amount of memory accesses per thread block
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Wrapper counter(dGPU, num_blocks);
		mykernel<<<dimGrid, dimBlock>>>(rGPU, counter, n);
		counter.dump_num_accesses();
	}
#endif

#if 1
	{
		// Record SM cycle timestamps of every memory access to device_data
		// This magic number comes from running the Counter wrapper (in the previous block) on the input data
		const int max_access_count = 32768;
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Wrapper recorder(dGPU, num_blocks, max_access_count);
		mykernel<<<dimGrid, dimBlock>>>(rGPU, recorder, n);
		std::ofstream outf(patterns_out_path);
		recorder.dump_access_clocks(outf);
	}
#endif

	CHECK_CUDA_ERROR(cudaGetLastError());

	// Copy data back to CPU & release memory
	CHECK_CUDA_ERROR(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(dGPU));
	CHECK_CUDA_ERROR(cudaFree(rGPU));
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
	// Write output
	std::ofstream outf(results_out_path);
	std::copy(result.begin(), result.end(), std::ostream_iterator<float>(outf, " "));
}
