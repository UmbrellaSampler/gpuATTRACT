
#ifndef GPUWORKER_H_
#define GPUWORKER_H_

#include "as/Worker.h"

namespace as {


class GPUWorker : public as::Worker {
public:
	/* Constructor */
	GPUWorker(ServerManagement& S_mngt,
		int deviceId, unsigned atomBufferSize, unsigned dofBufferSize);

	/* Destructor */
	virtual ~GPUWorker();

	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	void run();

	/****************************
	 * public member variables
	 ****************************/

protected:
	/****************************
	 * protected member functions
	 ****************************/

	/****************************
	 * protected member variables
	 ****************************/

private:
	/****************************
	 * private member functions
	 ****************************/
	inline int deviceLocalProteinID(const int& globalId);


	inline int deviceLocalGridID(const int& globalId);

	/****************************
	 * private member variables
	 ****************************/
	int _deviceId;
	unsigned _atomBufferSize;
	unsigned _dofBufferSize;
};

}
#endif /* GPUWORKER_H_ */
