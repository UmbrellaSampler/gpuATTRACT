
#include "cuda_runtime.h"
#include <cassert>
#include <mutex>

#include "as/BufferManagement.h"
#include "asUtils/macros.h"
#include "asUtils/Logger.h"

using namespace Log;


/* Constructor */
as::BufferManagement::BufferManagement(unsigned numItems, unsigned numDOFsPerItem) :
		_numItems(numItems), _itemQueue()
{
	assert(numItems > 0);
	assert(numDOFsPerItem > 0);
	assert(numItems*numDOFsPerItem*(sizeof(DOF)+sizeof(EnGrad)) <
			1000*1000*10*(sizeof(DOF)+sizeof(EnGrad))); // < 540 MByte == 10 mio DOFs

	cudaVerify(cudaMallocHost((void**)&_contDOFBuffer, numItems*numDOFsPerItem*sizeof(DOF)));
	cudaVerify(cudaMallocHost((void**)&_contEnGradBuffer, numItems*numDOFsPerItem*sizeof(EnGrad)));
//	_contDOFBuffer = new DOF[numItems*numDOFsPerItem];
//	_contEnGradBuffer = new EnGrad[numItems*numDOFsPerItem];
	//DEBUG
	_contItemBuffer = new WorkerItem[numItems];

	/* Cut large buffer into small pieces to create work items
	 * Store the Item in the queue container */
	for (unsigned i = 0; i < numItems; ++i) {
		as::WorkerItem* item = _contItemBuffer + i;
		item->setDOFBuffer(_contDOFBuffer + i*numDOFsPerItem);
		item->setEnGradBuffer(_contEnGradBuffer + i*numDOFsPerItem);
		_itemQueue.push(item);
	}
}

/* Destructor
 * Make sure the no items are currently processed
 * or have not yet been fetched by the client.
 * For that call the complete() method */
as::BufferManagement::~BufferManagement() {
	global_log->info() << "Shutdown Buffer-Management:" << std::endl;
	if (!complete()) {
		global_log->warning() << "Warning: Item buffers are getting freed "
				<< "while computations might still be in progress." << std::endl;
	} else {
		global_log->info() << std::setw(5) << " " << "Ok" << std::endl;
	}
	cudaVerify(cudaFreeHost(_contDOFBuffer));
	cudaVerify(cudaFreeHost(_contEnGradBuffer));
//	delete[] _contDOFBuffer;
//	delete[] _contEnGradBuffer;
	delete[] _contItemBuffer;
}



/****************************
 * public member functions
 ****************************/
as::WorkerItem* as::BufferManagement::removeItem()
{
	std::lock_guard<std::mutex> guard(_mutex);
	if (_itemQueue.size() == 0) {
		return NULL;
	}

	WorkerItem* item = _itemQueue.front();
	_itemQueue.pop();
	return item;
}

void as::BufferManagement::addItem(WorkerItem* const item)
{
	std::lock_guard<std::mutex> guard(_mutex);
	assert(item->isReady() == false);
	_itemQueue.push(item);
}

/*
 ** @brief: Checks if all items are back to the buffer management.
 */
bool as::BufferManagement::complete()
{
	// maybe protection is unnecessary
	std::lock_guard<std::mutex> guard(_mutex);
	return _itemQueue.size() == _numItems;
}



/****************************
 * protected member functions
 ****************************/

/****************************
 * private member functions
 ****************************/


