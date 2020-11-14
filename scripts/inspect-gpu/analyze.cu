#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

struct DeviceStat {
    const std::string label;
    const std::string attr_unit;
    const cudaDeviceAttr attr_enum;
    int device;
};

class DeviceAnalyzer {
    const std::vector<DeviceStat> device_stats;
    public:
        DeviceAnalyzer(int);
        void print_stats() const;
};

DeviceAnalyzer::DeviceAnalyzer(int device_num) : device_stats({
    {
        "compute capability",
        "",
        cudaDevAttrComputeCapabilityMajor,
        device_num
    },
    {
        "global memory bus width",
        "bits",
        cudaDevAttrGlobalMemoryBusWidth,
        device_num
    },
    {
        "streaming multiprocessors",
        "",
        cudaDevAttrMultiProcessorCount,
        device_num
    },
    {
        "maximum threads per SM",
        "",
        cudaDevAttrMaxThreadsPerMultiProcessor,
        device_num
    },
    {
        "L2 cache size",
        "bytes",
        cudaDevAttrL2CacheSize,
        device_num
    },
}) { };

void DeviceAnalyzer::print_stats() const {
    for (const auto& stat : device_stats) {
        int attr_value;
        CHECK(cudaDeviceGetAttribute(&attr_value, stat.attr_enum, stat.device));
        std::cout << std::left << std::setw(30)
                  << stat.label + ": " << attr_value
                  << ' ' << stat.attr_unit << std::endl;
    }
}


int main(int argc, char** argv) {
    if (argc > 2 || (argc >= 2 && std::strcmp(argv[1], "-h") == 0)) {
        std::cerr << "usage: ./analyze [-h] [device_num]\n";
        std::exit(2);
    }

    int device_num = 0;
    if (argc == 2) {
        device_num = std::stoi(argv[1]);
    }

    const DeviceAnalyzer analyzer(device_num);
    std::cout << "Device " << device_num << " stats:\n";
    analyzer.print_stats();
}
