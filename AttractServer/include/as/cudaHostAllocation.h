
#ifndef CUDAHOSTALLOCATION_H_
#define CUDAHOSTALLOCATION_H_

#include <cstring>
#include "cuda_runtime.h"

#include "asUtils/macros.h"

namespace as {

class HOST_PINNED {
public:
	template<class T>
	inline static void malloc(T* &ptr, size_t size) {
		cudaVerify(cudaMallocHost(&ptr, size) );
	}
	template<class T>
	inline static void free(T* &ptr) {
		cudaVerify(cudaFreeHost(ptr));
	}
	template<class T>
	inline static void memset(T* &ptr, int value, size_t size) {
		std::memset(ptr, value, size);
	}

	const static bool HostAlloc = true;
	const static bool DeviceAlloc = true;
};

class HOST {
public:
	template<class T>
	inline static void malloc(T* &ptr, size_t size) {
		ptr = (T*)std::malloc(size);
	}
	template<class T>
	inline static void free(T* &ptr) {
		std::free(ptr);
	}
	template<class T>
	inline static void memset(T* &ptr, int value, size_t size) {
		std::memset(ptr, value, size);
	}

	const static bool HostAlloc = true;
	const static bool DeviceAlloc = true;
};

class DEVONLY {
public:
	template<class T>
	inline static void malloc(T* &ptr, size_t size) {	}
	template<class T>
	inline static void free(T* &ptr) { }
	template<class T>
	inline static void memset(T* &ptr, int value, size_t size) { }

	const static bool HostAlloc = false;
	const static bool DeviceAlloc = true;
};

class HOSTONLY {
public:
	template<class T>
	inline static void malloc(T* &ptr, size_t size) {
		HOST::malloc(ptr, size);
	}
	template<class T>
	inline static void free(T* &ptr) {
		HOST::free(ptr);
	}
	template<class T>
	inline static void memset(T* &ptr, const int &value, const size_t& size) {
		HOST::memset(ptr, value, size);
	}

	const static bool HostAlloc = true;
	const static bool DeviceAlloc = false;
};

}


#endif /* CUDAHOSTALLOCATION_H_ */
