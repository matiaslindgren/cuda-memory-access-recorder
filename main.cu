#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "pattern_recorder.h"


using Recorder = pr::PatternRecorder<float>;
using Counter = pr::AccessCounter<float>;

const std::string patterns_out_path("/tmp/access-patterns.txt");
const std::string results_out_path("/tmp/correlations.txt");

__global__
/* void row_dot_products(int ny, int nx, const float* data, float* result) { */
/* void row_dot_products(int ny, int nx, Counter data, float* result) { */
void row_dot_products(int ny, int nx, Recorder data, float* result) {
	data.enter_kernel();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if (j > i || i >= ny)
		return;
	float ij_dot_prod = 0.0f;
	for (int k = 0; k < nx; k++) {
		float x = data[nx*i + k];
		float y = data[nx*j + k];
		ij_dot_prod += x * y;
	}
	result[i + j*ny] = ij_dot_prod;
}


__host__
inline void normalize(int ny, int nx, const float* data, float* result) {
	for (int i = 0; i < ny; i++) {
		float tmp = 0.0f;
		for (int j = 0; j < nx; j++) {
			tmp += data[i*nx + j];
		}
		tmp /= nx;
		for (int j = 0; j < nx; j++) {
			result[i*nx + j] = data[i*nx + j] - tmp;
		}
		tmp = 0.0f;
		for (int j = 0; j < nx; j++) {
			float x = result[i*nx + j];
			tmp += x*x;
		}
		tmp = 1.0f / std::sqrt(tmp);
		for (int j = 0; j < nx; j++) {
			result[i*nx + j] *= tmp;
		}
	}
}


__host__
inline int divup(int a, int b) {
	return (a + b - 1) / b;
}


__host__
void correlate(int ny, int nx, const float* data, float* result) {
	const int data_size = ny * nx * sizeof(float);
	const int result_size = ny * ny * sizeof(float);

	std::vector<float> data_normalized(data_size / sizeof(float), 0.0f);
	normalize(ny, nx, data, data_normalized.data());

	// Initialize device memory
	float* device_data = nullptr;
	float* device_result = nullptr;
	CHECK_CUDA_ERROR(cudaMalloc(&device_data, data_size));
	CHECK_CUDA_ERROR(cudaMalloc(&device_result, result_size));
	CHECK_CUDA_ERROR(cudaMemcpy(device_data, data_normalized.data(), data_size, cudaMemcpyHostToDevice));

	dim3 dimBlock(16, 16);
	dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));

#if 0
	{ // Run without analysis
		row_dot_products<<<dimGrid, dimBlock>>>(ny, nx, device_data, device_result);
	}
#endif

#if 0
	{ // Count amount of memory accesses per thread block
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Counter counter(device_data, num_blocks);
		row_dot_products<<<dimGrid, dimBlock>>>(ny, nx, counter, device_result);
		counter.dump_num_accesses();
	}
#endif

#if 1
	{ // Record SM cycle timestamps of every memory access to device_data
		// This magic number comes from running the Counter wrapper (in the previous block) on the input data
		const int max_access_count = 16384;
		const int num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		Recorder recorder(device_data, num_blocks, max_access_count);
		row_dot_products<<<dimGrid, dimBlock>>>(ny, nx, recorder, device_result);
		std::ofstream outf(patterns_out_path);
		recorder.dump_access_clocks(outf);
	}
#endif

	CHECK_CUDA_ERROR(cudaGetLastError());

	// Copy results from device
	CHECK_CUDA_ERROR(cudaMemcpy(result, device_result, result_size, cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(device_data));
	CHECK_CUDA_ERROR(cudaFree(device_result));
}


__host__
float next_float() {
	static std::random_device rd;
	static std::default_random_engine e(rd());
	static std::uniform_real_distribution<float> floats(0.0, 1.0);
	return floats(e);
}


// Driver program that computes pairwise correlations of a random float matrix.
__host__
int main() {
	// Generate data
	int nx = 32;
	int ny = 32;
	std::vector<float> matrix(nx * ny);
	std::generate(matrix.begin(), matrix.end(), next_float);
	std::vector<float> result(nx * ny);
	// Compute stuff
	correlate(ny, nx, matrix.data(), result.data());
	// Write output
	std::ofstream outf(results_out_path);
	outf << nx << " " << ny << "\n";
	std::copy(result.begin(), result.end(), std::ostream_iterator<float>(outf, " "));
	std::cout << "wrote access patterns to '" << patterns_out_path << "'\n";
	std::cout << "wrote pairwise correlation output to '" << results_out_path << "'\n";
}
