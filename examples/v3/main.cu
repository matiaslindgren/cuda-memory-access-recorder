/*
 * Pattern analysis applied to example from http://ppc.cs.aalto.fi/ch4/v3/
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "pattern_recorder.cuh"


inline void check(cudaError_t err, const char* context) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error: " << context << ": "
			<< cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
}

#define CHECK(x) check(x, #x)


const char* patterns_out_path = "access-patterns-v3.json";


template <class KernelData>
__global__
void mykernel(float* r, KernelData d, int n, int nn) {
	d.enter_kernel();
	int ia = threadIdx.x;
	int ja = threadIdx.y;
	int ic = blockIdx.x;
	int jc = blockIdx.y;

	/* const float* t = d + nn * nn; */

	__shared__ float xx[4][64];
	__shared__ float yy[4][64];

	float v[8][8];
	for (int ib = 0; ib < 8; ++ib) {
		for (int jb = 0; jb < 8; ++jb) {
			v[ib][jb] = HUGE_VALF;
		}
	}
	for (int ks = 0; ks < n; ks += 4) {
		int ija = ja * 8 + ia;
		int i = ic * 64 + ija;
		int j = jc * 64 + ija;
		for (int f = 0; f < 4; ++f) {
			int k = ks + f;
			/* xx[f][ija] = t[nn*k + i]; */
			xx[f][ija] = d[nn * nn + nn*k + i];
			yy[f][ija] = d[nn*k + j];
		}

		__syncthreads();

		#pragma unroll
		for (int f = 0; f < 4; ++f) {
			float y[8];
			for (int jb = 0; jb < 8; ++jb) {
				y[jb] = yy[f][jb * 8 + ja];
			}
			for (int ib = 0; ib < 8; ++ib) {
				float x = xx[f][ib * 8 + ia];
				for (int jb = 0; jb < 8; ++jb) {
					v[ib][jb] = min(v[ib][jb], x + y[jb]);
				}
			}
		}

		__syncthreads();
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
	CHECK(cudaMalloc((void**)&dGPU, 2 * nn * nn * sizeof(float)));
	float* rGPU = NULL;
	CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
	CHECK(cudaMemcpy(rGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

	// Run normalization kernel
	{
		dim3 dimBlock(64, 1);
		dim3 dimGrid(1, nn);
		myppkernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n, nn);
		CHECK(cudaGetLastError());
	}

	// Run computation kernel twice to compute access patterns
	dim3 dimBlock(8, 8);
	dim3 dimGrid(nn / 64, nn / 64);
	unsigned long long max_access_count = 0u;
	{
		pr::AccessCounter<float> counter(dGPU, dimGrid);
		mykernel<pr::AccessCounter<float> ><<<dimGrid, dimBlock>>>(rGPU, counter, n, nn);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
		max_access_count = counter.get_max_access_count();
		counter.dump_access_statistics(std::cout);
	}
	{
		pr::PatternRecorder<float> recorder(dGPU, dimGrid, max_access_count);
		mykernel<pr::PatternRecorder<float> ><<<dimGrid, dimBlock>>>(rGPU, recorder, n, nn);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaGetLastError());
		std::ofstream outf(patterns_out_path);
		recorder.dump_json_results(outf, 2 * nn, nn);
		std::cout << "Wrote " << patterns_out_path << "\n";
	}

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
