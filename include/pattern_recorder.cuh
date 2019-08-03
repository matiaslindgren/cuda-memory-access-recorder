#ifndef PATTERN_RECORDER_H
#define PATTERN_RECORDER_H
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <cuda_runtime.h>
#include <assert.h>

// 0 quiet (except when explicitly calling ostream dump methods)
// 1 informative
// 2 insanity
#define PR_VERBOSITY 0

namespace pr {

__host__
inline void check(cudaError_t err, const char* file, int lineno, const char* context) {
	if (err != cudaSuccess) {
		std::cerr
			<< "CUDA error in '" << file << "' at line " << lineno << ":\n" << context << "\n"
			<< "Error string is '" << cudaGetErrorString(err) << "'\n";
		std::exit(EXIT_FAILURE);
	}
}
#define CHECK_CUDA_ERROR(x) pr::check((x), __FILE__, __LINE__, #x)


// Wrapper around an array of type KernelDataType for recording device memory accesses on the array.
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
	uint32_t* processor_ids;

	// Variables for limiting the amount of device memory to be used.
	//
	// Maximum amount of accesses one thread block will perform during kernel execution.
	// This number should not be exceeded.
	const unsigned long long accesses_per_block;
	// Amount of thread blocks in the grid.
	const unsigned num_blocks;
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
	PatternRecorder(const KernelDataType* device_data, unsigned nb, unsigned long long ab) :
		data(device_data),
		accesses(nullptr),
		clocks(nullptr),
		processor_ids(nullptr),
		accesses_per_block(ab),
		num_blocks(nb),
		block_idx_in_grid(0),
		thread_idx_in_block(0),
		is_master(true)
	{
#if (PR_VERBOSITY > 0)
		std::cerr << "Constructing PatternRecorder " << this << " for device data " << device_data << "\n";
#endif
		const size_t clocks_size = num_blocks * accesses_per_block * sizeof(unsigned);
		const size_t accesses_size = num_blocks * accesses_per_block * sizeof(int);
		const size_t processor_ids_size = num_blocks * sizeof(uint32_t);
#if (PR_VERBOSITY > 0)
		std::cerr << "Trying to allocate " << (clocks_size + accesses_size + processor_ids_size) << " bytes on the device\n";
#endif
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
		accesses_per_block(other.accesses_per_block),
		num_blocks(other.num_blocks),
		is_master(false) // Makes destructor a no-op
	{
		data = other.data;
		accesses = other.accesses;
		clocks = other.clocks;
		processor_ids = other.processor_ids;
		block_idx_in_grid = 0;
		thread_idx_in_block = 0;
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
		assert(block_idx_in_grid < num_blocks);
		// Extract id of the SM this thread block got scheduled on
		uint32_t SM_id;
		// %smid is a predefined PTX identifier
		asm volatile("mov.u32 %0, %%smid;" : "=r"(SM_id));
		processor_ids[block_idx_in_grid] = SM_id;
		// Use the current cycle counter value of this SM as 'time zero'
		start_clock = clock64();
		// Offset write buffer of accesses and SM cycle clocks to correct position for measurements from this thread block
		buffer_idx = thread_idx_in_block + block_idx_in_grid * accesses_per_block;
#if (PR_VERBOSITY > 1)
		fprintf(stderr,
			"%p entered kernel, bufidx %d, start_clock %llu on sm %u, t %d b %d\n",
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
		fprintf(stderr,
			"%p accessed %d bufferidx offset %d->%d, SM cycle %llu, t %d b %d\n",
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

	__host__
	inline void dump_thread_block_SM_schedules(std::ostream& out=std::cout, char sep='\t', bool no_header=false) const {
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		if (!no_header) {
			out << "Thread block" << sep << "SM id\n";
		}
		//TODO block index mapping to actual blockIdx
		// processor_ids was allocated as Unified Memory so there is no need to explicitly memcpy from device to host
		for (int b = 0; b < num_blocks; ++b) {
			out << b << sep << processor_ids[b] << "\n";
		}
	}

	__host__
	inline void dump_access_clocks(std::ostream& out=std::cout, char sep='\t', bool no_header=false) const {
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		if (!no_header) {
			out << "Memory index" << sep << "Accessed at cycle\n";
		}
		const int num_accesses = num_blocks * accesses_per_block;
		for (int i = 0; i < num_accesses; ++i)
		{
			unsigned c = clocks[i];
			int a = accesses[i];
			// index a was never accessed <=> timestamp c is empty
			assert((a != -1 || c == ~0u) && (a == -1 || c != ~0u));
			if (c != ~0u || a != -1) {
				// Index a was accessed at clock cycle c
				out << a << sep << c << "\n";
			}
		}
	}
};


// Wrapper around an array of type KernelDataType for recording amount of memory accesses to that array.
// Class members are same as in PatternRecorder
template <typename KernelDataType>
class AccessCounter {
	const KernelDataType* data;
	// Amount of memory accesses performed by threads in a thread block.
	unsigned long long* access_counters;

	const unsigned num_blocks;
	const bool is_master;

	int block_idx_in_grid;

public:
	__host__
	AccessCounter(const KernelDataType* device_data, unsigned nb) :
		data(device_data),
		access_counters(nullptr),
		num_blocks(nb),
		block_idx_in_grid(0),
		is_master(true)
	{
#if (PR_VERBOSITY > 0)
		std::cerr << "Constructing AccessCounter " << this << " for device data " << device_data << "\n";
#endif
		const size_t access_counters_size = num_blocks * sizeof(unsigned long long);
		CHECK_CUDA_ERROR(cudaMallocManaged(&access_counters, access_counters_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
		CHECK_CUDA_ERROR(cudaMemset(access_counters, 0, access_counters_size));
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
#if (PR_VERBOSITY > 0)
		std::cerr << "AccessCounter " << this << " allocated " << access_counters_size << " bytes on the device\n";
#endif
	}

	__host__
	AccessCounter(const AccessCounter& other) :
		num_blocks(other.num_blocks),
		is_master(false)
	{
		data = other.data;
		access_counters = other.access_counters;
		block_idx_in_grid = 0;
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
	}

	__device__ __forceinline__
	KernelDataType operator[](int idx) {
#if (PR_VERBOSITY > 1)
		fprintf(stderr,
			"%p accessed %d, t %d b %d\n",
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
	inline void dump_num_accesses(std::ostream& out=std::cout, bool max_only=false) const {
		CHECK_CUDA_ERROR(cudaDeviceSynchronize());
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
		const auto max_access_count = *std::max_element(
			non_zeros.begin(),
			non_zeros.end()
		);
		if (max_only) {
			out << max_access_count << "\n";
			return;
		}
		const auto min_access_count = *std::min_element(
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
			num_blocks * sizeof(uint32_t)
		);
		// Required PatternRecorder size might be long, add some commas
		std::string req_size_str = std::to_string(req_access_pattern_size);
		for (int pos = req_size_str.size() - 3; pos > 0; pos -= 3) {
			req_size_str.insert(pos, 1, ',');
		}
		out << "Recorded all memory accesses for all thread blocks:\n"
			<< num_blocks << '\t' << "the grid was specified to contain this many thread blocks\n"
			<< num_zero_elements << '\t' << "blocks did not perform any memory accesses\n"
			<< num_non_zero_elements << '\t' << "blocks performed at least 1 memory access\n"
			<< "Statistics on number of accesses per thread block:\n"
			<< min_access_count << '\t' << "minimum amount of accesses\n"
			<< median << '\t' << "median amount of accesses\n"
			<< max_access_count << '\t' << "maximum amount of accesses\n"
			<< "The amount of device memory required by an PatternRecorder instance would be:\n"
			<< req_size_str << '\t' << "bytes\n";
	}
};

} // namespace pr
#endif // PATTERN_RECORDER_H
