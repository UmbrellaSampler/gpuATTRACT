/*
 * Worker.h
 *
 *  Created on: Jan 20, 2015
 *      Author: uwe
 */

#ifndef WORKER_H_
#define WORKER_H_

#include "asUtils/Thread.h"
#include "as/Protein.h"
#include "as/GridUnion.h"
#include "as/ParamTable.h"
#include "as/SimParam.h"
#include "as/WorkItem.h"
#include <as/SaveWorkQueue.h>


namespace as {

class ServerManagement;

class Worker : public asUtils::Thread {
public:
	/* Constructor */
	Worker(ServerManagement& S_mngt) :
		_S_mngt(S_mngt) {}

	/* Destructor */
	virtual ~Worker() {}

	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	inline bool isActive() {
		return (!_queue.sleeps());
	}

	inline void signalTerminate() {
		_queue.signalTerminate();
	}

	inline void addItem(WorkerItem* const item) {
		_queue.add(item);
	}

	inline unsigned fillLevel() const {
		return _queue.size();
	}

	virtual void run() = 0;

	/****************************
	 * public member variables
	 ****************************/

protected:
	/****************************
	 * protected member functions
	 ****************************/

	Protein* getProtein(const unsigned& globId);

	GridUnion* getGridUnion(const unsigned& globId);

	AttrParamTable* getParamTable();

	SimParam* getSimParam();

	/****************************
	 * protected member variables
	 ****************************/
	SafeWorkQueue<WorkerItem*> _queue;
	ServerManagement& _S_mngt;

private:
	/****************************
	 * private member functions
	 ****************************/

	/****************************
	 * private member variables
	 ****************************/
};

} // namespace

#endif /* WORKER_H_ */
