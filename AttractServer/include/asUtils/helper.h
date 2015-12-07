
#ifndef HELPER_H_
#define HELPER_H_

#include "asUtils/asUtilsTypes.h"
#include "asUtils/macros.h"

namespace asUtils {

inline double sizeConvert(unsigned long long sizeInByte, sizeType_t type = byte) {
	return double(sizeInByte) / type;
}

inline __host__ __device__ bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

inline __host__ __device__ unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

inline __host__ __device__ void getNumBlocksAndThreads(const int& n, const int& maxBlocks, const int& maxThreads,
		int &blocks, int &threads)
{
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);

}

inline __host__ __device__ unsigned pow2(unsigned n) {
	return 1 << n;
}

}  // namespace asUtils


#endif /* HELPER_H_ */
