#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "pattern_recorder.cuh"

#define CHECK_CUDA_ERROR(x) pr::check((x), __FILE__, __LINE__, #x)


constexpr const char* patterns_out_path = "access-patterns-matmul.json";


template <class KernelData>
__global__
void matrix_square_gpu(float* out, KernelData in, int n) {
	in.enter_kernel();
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	double sum = 0;
	for (int k = 0; k < n; ++k) {
		float x = in[n*i + k];
		float y = in[n*k + j];
		double z = x * y;
		sum += z;
	}
	out[n*i + j] = (float)sum;
}


inline int static divup(int a, int b) {
	return (a + b - 1) / b;
}


void matrix_square(float* output_cpu, const float* input_cpu, std::size_t n) {
	float* input_gpu = nullptr;
	CHECK_CUDA_ERROR(cudaMalloc(&input_gpu, n * n * sizeof(float)));
	float* output_gpu = nullptr;
	CHECK_CUDA_ERROR(cudaMalloc(&output_gpu, n * n * sizeof(float)));
	CHECK_CUDA_ERROR(cudaMemcpy(input_gpu, input_cpu, n * n * sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimBlock(4, 4);
	dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));

	unsigned long long max_access_count = 0u;
	{
		std::cout << "counting amount of memory accesses\n";
		auto t0 = std::chrono::high_resolution_clock::now();

		pr::AccessCounter<float> counter(input_gpu, dimGrid);
		matrix_square_gpu<pr::AccessCounter<float> ><<<dimGrid, dimBlock>>>(output_gpu, counter, n);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		counter.dump_access_statistics(std::cout);
		max_access_count = counter.get_max_access_count();

		auto t1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
		std::cout << "took " << dt.count() * 1e3 << " ms\n";
	}

	{
		std::cout << "recording access pattern\n";
		auto t0 = std::chrono::high_resolution_clock::now();

		pr::PatternRecorder<float> recorder(input_gpu, dimGrid, max_access_count);
		matrix_square_gpu<pr::PatternRecorder<float> ><<<dimGrid, dimBlock>>>(output_gpu, recorder, n);
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());

		auto t1 = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> dt = t1 - t0;
		std::cout << "took " << dt.count() * 1e3 << " ms\n";

		std::ofstream outf{patterns_out_path};
		recorder.dump_json_results(outf, n, n);
		std::cout << "Wrote " << patterns_out_path << "\n";
	}

	CHECK_CUDA_ERROR(cudaGetLastError());

	CHECK_CUDA_ERROR(cudaMemcpy(output_cpu, output_gpu, n * n * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA_ERROR(cudaFree(input_gpu));
	CHECK_CUDA_ERROR(cudaFree(output_gpu));
}


__host__
float next_float() {
	static std::random_device rd;
	static std::default_random_engine e(rd());
	static std::normal_distribution<float> floats(0.0, 1.0);
	return floats(e);
}


__host__
void dump_matrix(const std::vector<float>& v, std::size_t n, const std::string& path) {
	assert(v.size() % n == 0);

	std::ofstream outf{path};
	for (auto row = v.begin(); row != v.end(); row += n) {
		std::copy(row, row + n, std::ostream_iterator<float>(outf, " "));
		outf << "\n";
	}
}


__host__
int main() {
	std::size_t n{4*32};
	std::vector<float> matrix(n * n);
	std::generate(matrix.begin(), matrix.end(), next_float);

	dump_matrix(matrix, n, "matrix.txt");

	std::vector<float> result(n * n);
	matrix_square(result.data(), matrix.data(), n);

	dump_matrix(result, n, "result.txt");
}
