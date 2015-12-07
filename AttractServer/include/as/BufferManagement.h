
#ifndef BUFFERMANAGEMENT_H_
#define BUFFERMANAGEMENT_H_

#include <list>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <as/WorkItem.h>

namespace as {


class BufferManagement {
public:
	/* Constructor */
	BufferManagement(unsigned numItems, unsigned numDOFsPerItem);

	/* Destructor */
	~BufferManagement();

	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	WorkerItem* removeItem();

	void addItem(WorkerItem* const item);

	/*
	 ** @brief: Indicates if all items are back
	 ** in the queue. The object can now safely be deleted.
	 */
	bool complete();

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

	unsigned _numItems;

	DOF* _contDOFBuffer;		/** Page-Locked allocation */
	EnGrad* _contEnGradBuffer;	/** Page-Locked allocation */
	WorkerItem* _contItemBuffer;/** Ptr. to all allocated WorkerItems */

	std::queue<WorkerItem*, std::list<WorkerItem*> > _itemQueue;

	std::mutex _mutex;
};

} // namespace

#endif /* BUFFERMANAGEMENT_H_ */
