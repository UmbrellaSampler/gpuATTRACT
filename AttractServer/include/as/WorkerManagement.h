/*
 * WorkerManagement.h
 *
 *  Created on: Jan 20, 2015
 *      Author: uwe
 */

#ifndef WORKERMANAGEMENT_H_
#define WORKERMANAGEMENT_H_

#include "as/SmartVector.h"
#include "as/GPUWorker.h"
#include "as/CPUWorker.h"
#include "as/WorkItem.h"
#include <iostream>
#include <set>
#include <vector>
#include <mutex>

namespace as {

class ServerManagement;


class WorkerManagement {
public:
	/* Constructor */
	WorkerManagement(ServerManagement& S_mngt, unsigned numDOFsPerItem, unsigned GPU_atomBufferSize, unsigned CPU_atomBufferSize);


	/* Destructor */
	~WorkerManagement();

	/***************
	* G E T T E R
	***************/
//	inline unsigned numCPUWorkers() const {
//		return _CPUWorkers.realSize();
//	}

	unsigned numCPUWorkers();

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	void addGPUWorker(int deviceId);

	void addCPUWorker();

	void removeGPUWorker(int deviceId);


	void removeCPUWorker();

	unsigned deviceFillLevel(const int& deviceId) const {
		return _GPUWorkers[deviceId]->fillLevel();
	}

	unsigned hostFillLevel(const int& id) const {
		return _CPUWorkers[id]->fillLevel();
	}

	void pushItemToGPUWorker(WorkerItem* const item, const int& deviceId) const {
		_GPUWorkers[deviceId]->addItem(item);
	}

	void pushItemToCPUWorker(WorkerItem* const item, const int& id) const {
		_CPUWorkers[id]->addItem(item);
	}

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

	/****************************
	 * private member variables
	 ****************************/
	unsigned _GPU_AtomBufferSize;
	unsigned _CPU_AtomBufferSize;
	unsigned _dofBufferSize;

//	SmartVector<CPUWorker*> _CPUWorkers;	/** for fast access */
	std::vector<CPUWorker*> _CPUWorkers;	/** for fast access */

	SmartVector<GPUWorker*> _GPUWorkers;	/** for fast access */
	std::set<int> _GPUWorkerIds; 			/** freeing purposes */

	ServerManagement& _S_mngt;

	std::mutex _m_CPUWorker;
};

} // namespace

#endif /* WORKERMANAGEMENT_H_ */
