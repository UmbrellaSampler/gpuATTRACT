
#ifndef UTILS_H_
#define UTILS_H_

#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cuda_runtime.h>



#define PROFILE




#ifndef NDEBUG
#define cudaVerify(x) do { 																				\
		cudaError_t __cu_result = x; 																	\
		if (__cu_result!=cudaSuccess) { 																\
			fprintf(stderr, "%s:%i: error: cuda function call failed:\n" 								\
					"%s;\nmessage: %s\n", 																\
					__FILE__, __LINE__, #x, cudaGetErrorString(__cu_result));							\
			exit(1);																					\
		} 																								\
	} while(0)
#define cudaVerifyKernel(x) do {																		\
		x;																								\
		cudaError_t __cu_result = cudaGetLastError();													\
		if (__cu_result!=cudaSuccess) { 																\
			fprintf(stderr, "%s:%i: error: cuda function call failed:\n" 								\
					"%s;\nmessage: %s\n", 																\
					__FILE__, __LINE__, #x, cudaGetErrorString(__cu_result));							\
			exit(1);																					\
		} 																								\
	} while(0)
#else
#define cudaVerify(x) do {																				\
		x;																								\
	} while(0)
#define cudaVerifyKernel(x) do {																		\
		x;																								\
	} while(0)
#endif

#ifdef PROFILE
#define profile(str,x) 		\
	nvtxRangePushA(str);	\
	x; 						\
	nvtxRangePop();         \

#else
#define profile(str,x)      \
	x;                      \

#endif

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x < y) ? y : x)
#endif


#endif /* UTILS_H_ */
