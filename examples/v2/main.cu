/*
 * Pattern analysis applied to example from http://ppc.cs.aalto.fi/ch4/v2/
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
void mykernel(float* r, Wrapper d, int n, int nn) {
	d.enter_kernel();
	int ia = threadIdx.x;
	int ja = threadIdx.y;
	int ic = blockIdx.x;
	int jc = blockIdx.y;

	// pr::PatternRecorder and pr::AccessCounter do not currently support pointer arithmetic
	/* const float* t = d + nn * nn; */

	float v[8][8];
	for (int ib = 0; ib < 8; ++ib) {
		for (int jb = 0; jb < 8; ++jb) {
			v[ib][jb] = HUGE_VALF;
		}
	}
	for (int k = 0; k < n; ++k) {
		float x[8];
		float y[8];
		for (int ib = 0; ib < 8; ++ib) {
			int i = ic * 64 + ib * 8 + ia;
			/* x[ib] = t[nn*k + i]; */
			x[ib] = d[nn * nn + nn*k + i];
		}
		for (int jb = 0; jb < 8; ++jb) {
			int j = jc * 64 + jb * 8 + ja;
			y[jb] = d[nn*k + j];
		}
		for (int ib = 0; ib < 8; ++ib) {
			for (int jb = 0; jb < 8; ++jb) {
				v[ib][jb] = min(v[ib][jb], x[ib] + y[jb]);
			}
		}
	}
	for (int ib = 0; ib < 8; ++ib) {
		for (int jb = 0; jb < 8; ++jb) {
			int i = ic * 64 + ib * 8 + ia;
			int j = jc * 64 + jb * 8 + ja;
			if (i < n && j < n) {
				r[n*i + j] = v[ib][jb];
			}
		}
	}
}


__global__ void myppkernel(const float* r, float* d, int n, int nn) {
	int ja = threadIdx.x;
	int i = blockIdx.y;

	float* t = d + nn * nn;

	for (int jb = 0; jb < nn; jb += 64) {
		int j = jb + ja;
		float v = (i < n && j < n) ? r[n*i + j] : HUGE_VALF;
		d[nn*i + j] = v;
		t[nn*j + i] = v;
	}
}

inline int static divup(int a, int b) {
	return (a + b - 1) / b;
}

inline int static roundup(int a, int b) {
	return divup(a, b) * b;
}

void step(float* r, const float* d, int n) {
	int nn = roundup(n, 64);

	// Allocate memory & copy data to GPU
	float* dGPU = NULL;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
	float* rGPU = NULL;
	CHECK_CUDA_ERROR(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(rGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Run kernel
	{
		dim3 dimBlock(64, 1);
		dim3 dimGrid(1, nn);
		myppkernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
		CHECK_CUDA_ERROR(cudaGetLastError());
	}

	dim3 dimBlock(8, 8);
	dim3 dimGrid(nn / 64, nn / 64);
#if 0
	{
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Wrapper counter(dGPU, num_blocks);
		mykernel<<<dimGrid, dimBlock>>>(rGPU, counter, n, nn);
		counter.dump_num_accesses();
	}
#endif

#if 1
	{
		const int max_access_count = 65536;
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Wrapper recorder(dGPU, num_blocks, max_access_count);
		mykernel<<<dimGrid, dimBlock>>>(rGPU, recorder, n, nn);
		std::ofstream outf(patterns_out_path);
		recorder.dump_access_clocks(outf);
	}
#endif

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
