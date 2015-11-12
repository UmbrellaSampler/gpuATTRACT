/*
 * CPUWorker.h
 *
 *  Created on: Feb 5, 2015
 *      Author: uwe
 */

#ifndef CPUWORKER_H_
#define CPUWORKER_H_

#include "as/Worker.h"

namespace as {

class CPUWorker : public as::Worker {
public:
	/* Constructor */
	CPUWorker(ServerManagement& S_mngt,
			unsigned atomBufferSize);

	/* Destructor */
	virtual ~CPUWorker();

	/***************
	* G E T T E R
	***************/
	int id() const {
		return _id;
	}

	/***************
	* S E T T E R
	***************/
	void set(int id) {
		_id = id;
	}

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
	int _id;

	unsigned _atomBufferSize;
};

} // namespace


#endif /* CPUWORKER_H_ */
