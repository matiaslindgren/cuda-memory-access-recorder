/*
 * Utilities for analyzing memory access patterns on data passed into a CUDA kernel.
 *
 * AccessCounter uses atomicAdd to count all memory accesses by every thread block.
 * PatternRecorder writes all accesses and SM cycle counter values to global device memory.
 */
#ifndef PATTERN_RECORDER_H
#define PATTERN_RECORDER_H
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// You can set the verbosity in your program before you include this header
// 0 quiet (except when explicitly calling ostream write methods)
// 1 informative
// 2 insanity
#ifndef PR_VERBOSITY
#define PR_VERBOSITY 0
#endif

// When the amount of memory required to store all access patterns exceeds this size, print a warning
#define MAX_BYTES_WARNING_LIMIT 200000000 /* 200M */

namespace pr {

__host__
inline void check(cudaError_t err, const char* file, int lineno, const char* context) {
	if (err != cudaSuccess) {
		std::cerr
			<< "CUDA error in '" << file << "' at line " << lineno << ":\n" << context << "\n"
			<< "Error string is '" << cudaGetErrorString(err) << "'\n";
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
}
#define CHECK_CUDA_ERROR(x) pr::check((x), __FILE__, __LINE__, #x)


// Wrapper around a KernelDataType pointer that records every memory access and SM cycle value during the access.
// Results are always written linearly into global device memory that should minimize interference with the memory access pattern being recorded.
template <typename KernelDataType>
class PatternRecorder {
	// Pointers to global device memory, shared among all copies of a PatternRecorder object.
	//
	// Data for which we are recording access patterns.
	// (pointer is not owned and assumed to be managed)
	const KernelDataType* data;
	// Indexes of device memory accesses to array 'data'.
	// (pointer is owned and managed)
	int* accesses;
	// Values of streaming multiprocessors (SM) cycle counters when a device memory access was made.
	// Note: each SM has an own cycle counter, independent from other SM's counters.
	// (pointer is owned and managed)
	unsigned* clocks;
	// For every thread block, ID of the SM that the block got scheduled on.
	// (pointer is owned and managed)
	unsigned* processor_ids;

	// Variables for limiting the amount of device memory to be used.
	//
	// Maximum amount of accesses one thread block will perform during kernel execution.
	// This number should not be exceeded.
	const unsigned long long accesses_per_block;
	// The regular, second parameter given to a CUDA kernel invocation.
	// Used to compute the maximum amount of blocks
	const dim3 dimGrid;
	// True iff this PatternRecorder object manages the device memory.
	// E.g. the copy constructor will set this to false for every copy.
	const bool is_master;


	// Variables local to each thread block during kernel execution.
	// These variables are assumed to be undefined and unused outside kernel calls.
	// They will be initialized independently in each thread block.
	//
	// Value of the SM cycle counter when a block started executing the kernel.
	long long int start_clock;
	// Running index to the accesses and clocks arrays.
	// Will be offset linearly within each thread block.
	int buffer_idx;
	// Linear index for a thread within its block.
	int thread_idx_in_block;
	// Linear index for a block within its grid.
	int block_idx_in_grid;

public:
	__host__
	PatternRecorder(const KernelDataType* device_data, dim3 dimGrid, unsigned long long ab) :
		data(device_data),
		accesses(nullptr),
		clocks(nullptr),
		processor_ids(nullptr),
		accesses_per_block(ab),
		dimGrid(dimGrid),
		block_idx_in_grid(0),
		thread_idx_in_block(0),
		is_master(true)
	{
#if (PR_VERBOSITY > 0)
		std::cerr << "Constructing PatternRecorder " << this << " for device data " << device_data << "\n";
#endif
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		const size_t clocks_size = num_blocks * accesses_per_block * sizeof(unsigned);
		const size_t accesses_size = num_blocks * accesses_per_block * sizeof(int);
		const size_t processor_ids_size = num_blocks * sizeof(unsigned);
		const size_t total_memory_required = clocks_size + accesses_size + processor_ids_size;
#if (PR_VERBOSITY > 0)
		std::cerr << "Trying to allocate " << total_memory_required << " bytes on the device\n";
#endif
#if (PR_VERBOSITY > 1)
		std::cerr
			<< clocks_size << '\t' << "for SM cycle clock values\n"
			<< accesses_size << '\t' << "for memory access indexes\n"
			<< processor_ids_size << '\t' << "for SM IDs\n";
#endif
		assert(clocks_size > 0);
		assert(accesses_size > 0);
		assert(processor_ids_size > 0);
		if (total_memory_required > MAX_BYTES_WARNING_LIMIT) {
			std::cerr
				<< "Warning: PatternRecorder is trying to allocate " << total_memory_required << " bytes of device memory for storing memory accesses.\n"
				<< "Reduce the sample size of the data that is being analyzed or increase MAX_BYTES_WARNING_LIMIT to silence this warning\n";
		}
		CHECK_CUDA_ERROR(cudaMallocManaged(&clocks, clocks_size));
		CHECK_CUDA_ERROR(cudaMallocManaged(&accesses, accesses_size));
		CHECK_CUDA_ERROR(cudaMallocManaged(&processor_ids, processor_ids_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		// ~0 denotes unused clock cycle timestamp
		CHECK_CUDA_ERROR(cudaMemset(clocks, ~0, clocks_size));
		// -1 denotes memory index that was never accessed
		CHECK_CUDA_ERROR(cudaMemset(accesses, -1, accesses_size));
		// ~0 denotes unused SM ID
		CHECK_CUDA_ERROR(cudaMemset(processor_ids, ~0, processor_ids_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#if (PR_VERBOSITY > 0)
		std::cerr << "PatternRecorder " << this << " allocated " << (clocks_size + accesses_size + processor_ids_size) << " bytes on the device\n";
#endif
	}

	// Copies of a PatternRecorder should not allocate new device memory but write to the same device memory area as source PatternRecorder object
	__host__
	PatternRecorder(const PatternRecorder& other) :
		data(other.data),
		accesses(other.accesses),
		clocks(other.clocks),
		processor_ids(other.processor_ids),
		accesses_per_block(other.accesses_per_block),
		dimGrid(other.dimGrid),
		block_idx_in_grid(0),
		thread_idx_in_block(0),
		is_master(false) // Makes destructor a no-op
	{
	}

	__host__
	~PatternRecorder() {
		if (is_master) {
#if (PR_VERBOSITY > 0)
			std::cerr << "~PatternRecorder releasing device memory allocated by " << this << "\n";
#endif
			CHECK_CUDA_ERROR(cudaFree(clocks));
			CHECK_CUDA_ERROR(cudaFree(accesses));
			CHECK_CUDA_ERROR(cudaFree(processor_ids));
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#if (PR_VERBOSITY > 0)
			std::cerr << "Successfully released device memory for " << this << "\n";
#endif
		} else {
#if (PR_VERBOSITY > 0)
			std::cerr << "~PatternRecorder called on " << this << " but it is copy constructed, will not release device memory yet\n";
#endif
		}
	}

	// This function should be called once somewhere in the beginning of the kernel
	__device__ __forceinline__
	void enter_kernel() {
		// Row-major order indexes of the current thread and block
		thread_idx_in_block = (
			threadIdx.x +
			threadIdx.y * blockDim.x +
			threadIdx.z * blockDim.x * blockDim.y
		);
		block_idx_in_grid = (
			blockIdx.x +
			blockIdx.y * gridDim.x +
			blockIdx.z * gridDim.x * gridDim.y
		);
		// Extract id of the SM this thread block got scheduled on
		uint32_t SM_id;
		// %smid is a predefined PTX identifier
		asm volatile("mov.u32 %0, %%smid;" : "=r"(SM_id));
		processor_ids[block_idx_in_grid] = static_cast<unsigned>(SM_id);
		// Use the current cycle counter value of this SM as 'time zero'
		start_clock = clock64();
		// Offset write buffer of accesses and SM cycle clocks to correct position for measurements from this thread block
		buffer_idx = thread_idx_in_block + block_idx_in_grid * accesses_per_block;
#if (PR_VERBOSITY > 1)
		printf("%p entered kernel, bufidx %d, start_clock %llu on sm %u, t %d b %d\n",
			this,
			buffer_idx,
			start_clock,
			SM_id,
			thread_idx_in_block,
			block_idx_in_grid
		);
#endif
	}

	__device__ __forceinline__
	KernelDataType operator[](int idx) {
		// Record which data index was accessed.
		accesses[buffer_idx] = idx;
		// Record current SM clock cycle.
		// We assume that the program will not be running for more than 2^32 cycles.
		clocks[buffer_idx] = static_cast<unsigned>(clock64() - start_clock);
		// Offset access buffer indexes.
		int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
		buffer_idx += threads_per_block;
#if (PR_VERBOSITY > 1)
		printf("%p accessed %d bufferidx offset %d->%d, SM cycle %llu, t %d b %d\n",
			this,
			idx,
			buffer_idx - threads_per_block,
			buffer_idx,
			clocks[buffer_idx - threads_per_block],
			thread_idx_in_block,
			block_idx_in_grid
		);
#endif
		return data[idx];
	}

	// No need for a json-library since we only need to write json
	__host__
	inline void dump_json_results(std::ostream& out, unsigned num_rows, unsigned num_cols) const {
		out << "{";
		out << "\"rows\":" << num_rows << ",\"cols\":" << num_cols;
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		const int num_accesses = num_blocks * accesses_per_block;
		out << ",\"accesses\":[";
		bool no_separator = true;
		for (int i = 0; i < num_accesses; ++i)
		{
			unsigned c = clocks[i];
			int a = accesses[i];
			// index a was never accessed <=> timestamp c is empty
			assert((a != -1 || c == ~0u) && (a == -1 || c != ~0u));
			if (a != -1) {
				// Index a was accessed at clock cycle c
				if (no_separator) {
					no_separator = false;
				} else {
					out << ',';
				}
				out << a;
			}
		}
		out << "]";
		out << ",\"clocks\":[";
		no_separator = true;
		for (int i = 0; i < num_accesses; ++i)
		{
			unsigned c = clocks[i];
			if (c != ~0u) {
				if (no_separator) {
					no_separator = false;
				} else {
					out << ',';
				}
				out << c;
			}
		}
		out << "]";
		out << ",\"SM_ids\":[";
		out << processor_ids[0];
		for (int b = 1; b < num_blocks; ++b) {
			out << ',' << processor_ids[b];
		}
		out << "]";
		out << "}\n";
	}

	__host__
	inline void dump_access_statistics(std::ostream& out, char sep='\t') const {
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		const unsigned max_SM_id = *std::max_element(processor_ids, processor_ids + num_blocks);
		std::vector<unsigned> counts(max_SM_id + 1, 0u);
		for (int b = 0; b < num_blocks; ++b) {
			++counts[processor_ids[b]];
		}
		const unsigned SMs_used = std::count_if(counts.begin(), counts.end(), [](unsigned c) { return c > 0; });
		out << SMs_used << sep << "different streaming multiprocessors used\n";
	}
};


// Wrapper around an array of type KernelDataType for recording amount of memory accesses to that array.
// Class members are same as in PatternRecorder
template <typename KernelDataType>
class AccessCounter {
	const KernelDataType* data;
	// Amount of memory accesses performed by threads in a thread block.
	unsigned long long* access_counters;

	const dim3 dimGrid;
	const bool is_master;

	int block_idx_in_grid;

public:
	__host__
	AccessCounter(const KernelDataType* device_data, dim3 dimGrid) :
		data(device_data),
		access_counters(nullptr),
		dimGrid(dimGrid),
		block_idx_in_grid(0),
		is_master(true)
	{
#if (PR_VERBOSITY > 0)
		std::cerr << "Constructing AccessCounter " << this << " for device data " << device_data << "\n";
#endif
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		const size_t access_counters_size = num_blocks * sizeof(unsigned long long);
#if (PR_VERBOSITY > 0)
		std::cerr << "Trying to allocate " << access_counters_size << " bytes on the device\n";
#endif
		assert(access_counters_size > 0);
		CHECK_CUDA_ERROR(cudaMallocManaged(&access_counters, access_counters_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaMemset(access_counters, 0, access_counters_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	}

	__host__
	AccessCounter(const AccessCounter& other) :
		data(other.data),
		access_counters(other.access_counters),
		dimGrid(other.dimGrid),
		block_idx_in_grid(0),
		is_master(false)
	{
	}

	__host__
	~AccessCounter() {
		if (is_master) {
#if (PR_VERBOSITY > 0)
			std::cerr << "~AccessCounter releasing device memory allocated by " << this << "\n";
#endif
			CHECK_CUDA_ERROR(cudaFree(access_counters));
			CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#if (PR_VERBOSITY > 0)
			std::cerr << "Successfully released device memory for " << this << "\n";
#endif
		} else {
#if (PR_VERBOSITY > 0)
			std::cerr << "~AccessCounter called on " << this << " but it is copy constructed, will not release device memory yet\n";
#endif
		}
	}

	__device__ __forceinline__
	void enter_kernel() {
		block_idx_in_grid = (
			blockIdx.x +
			blockIdx.y * gridDim.x +
			blockIdx.z * gridDim.x * gridDim.y
		);
#if (PR_VERBOSITY > 1)
		int thread_idx_in_block = (
			threadIdx.x +
			threadIdx.y * blockDim.x +
			threadIdx.z * blockDim.x * blockDim.y
		);
		printf("%p entered kernel, t %d b %d\n",
			this,
			thread_idx_in_block,
			block_idx_in_grid
		);
#endif
	}

	__device__ __forceinline__
	KernelDataType operator[](int idx) {
#if (PR_VERBOSITY > 1)
		int thread_idx_in_block = (
			threadIdx.x +
			threadIdx.y * blockDim.x +
			threadIdx.z * blockDim.x * blockDim.y
		);
		printf("%p accessed %d, t %d b %d\n",
			this,
			idx,
			thread_idx_in_block,
			block_idx_in_grid
		);
#endif
		// We don't care about efficiency so much here since we only want the amount of memory accesses
		atomicAdd(&access_counters[block_idx_in_grid], 1);
		return data[idx];
	}

	__host__
	inline unsigned long long get_max_access_count() const {
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		return *std::max_element(access_counters, access_counters + num_blocks);
	}

	__host__
	inline void dump_access_statistics(std::ostream& out, char sep='\t') const {
		const unsigned num_blocks = dimGrid.x * dimGrid.y * dimGrid.z;
		// Count all blocks that did no memory accesses
		auto num_zero_elements = std::count(access_counters, access_counters + num_blocks, 0);
		auto num_non_zero_elements = num_blocks - num_zero_elements;
		// Filter out all blocks that did no memory accesses
		std::vector<unsigned long long> non_zeros(num_non_zero_elements);
		std::copy_if(
			access_counters,
			access_counters + num_blocks,
			non_zeros.begin(),
			[](unsigned long long c) { return c > 0; }
		);
		const auto min_access_count = *std::min_element(
			non_zeros.begin(),
			non_zeros.end()
		);
		const auto max_access_count = *std::max_element(
			non_zeros.begin(),
			non_zeros.end()
		);
		// Re-order elements such that median is in the middle
		std::nth_element(
			non_zeros.begin(),
			non_zeros.begin() + non_zeros.size() / 2,
			non_zeros.end()
		);
		const auto median = non_zeros[non_zeros.size()/2];
		size_t req_access_pattern_size = (
			num_blocks * max_access_count * sizeof(unsigned) +
			num_blocks * max_access_count * sizeof(int) +
			num_blocks * sizeof(unsigned)
		);
		// Required PatternRecorder size might be long, add some commas
		std::string req_size_str = std::to_string(req_access_pattern_size);
		for (int pos = req_size_str.size() - 3; pos > 0; pos -= 3) {
			req_size_str.insert(pos, 1, ',');
		}
		out << "AccessCounter statistics:\n"
			<< num_non_zero_elements << sep << "thread blocks did at least one memory access\n"
			<< num_zero_elements << sep << "thread blocks did no memory accesses\n"
			<< "Statistics for thread blocks that did at least one memory access:\n"
			<< min_access_count << sep << "least amount of accesses\n"
			<< median << sep << "median amount of accesses\n"
			<< max_access_count << sep << "most amount of accesses\n"
			<< "The amount of device memory required by an PatternRecorder instance to record all accesses would be:\n"
			<< req_size_str << sep << "bytes\n";
	}
};

} // namespace pr

#undef PR_VERBOSITY
#undef CHECK_CUDA_ERROR
#undef MAX_BYTES_WARNING_LIMIT

#endif // PATTERN_RECORDER_H
