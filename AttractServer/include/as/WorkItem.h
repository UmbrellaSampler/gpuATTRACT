/*
 * WorkItem.h
 *
 *  Created on: Sep 1, 2015
 *      Author: uwe
 */

#ifndef WORKITEM_H_
#define WORKITEM_H_

#include <atomic>

#include "as/asTypes.h"

namespace as {

/*
 ** @brief: Represents a basic unit request that is processed by a worker
 ** (either CPU or GPU).
 ** Provides buffers and relevant information for a worker.
 ** Elements of the DOF input buffer result in DOF gradients and energy
 ** components, which are stored in the EnGrad ouput buffer accordingly.
 */
class WorkerItem {
public:
	/* Constructor */
	WorkerItem(DOF* dofs, EnGrad* enGrads, unsigned numDOFs, unsigned globGridId, unsigned globRecId, unsigned globLigId,
			unsigned devLocGridId, unsigned devLocRecId, unsigned devLocLigId) :
		_DOFs(dofs),
		_EnGrads(enGrads),
		_numDOFs(numDOFs),
		_globGridId(globGridId),
		_globRecId(globRecId),
		_globLigId(globLigId),
		_devLocGridId(devLocGridId),
		_devLocRecId(devLocRecId),
		_devLocLigId(devLocLigId),
		_ready(false){}

	WorkerItem(DOF* dofs, EnGrad* enGrads) :
		_DOFs(dofs),
		_EnGrads(enGrads),
		_numDOFs(0),
		_globGridId(9999),
		_globRecId(9999),
		_globLigId(9999),
		_devLocGridId(9999),
		_devLocRecId(9999),
		_devLocLigId(9999),
		_ready(false){}

	WorkerItem() :
		_DOFs(NULL),
		_EnGrads(NULL),
		_numDOFs(0),
		_globGridId(9999),
		_globRecId(9999),
		_globLigId(9999),
		_devLocGridId(9999),
		_devLocRecId(9999),
		_devLocLigId(9999),
		_ready(false){}

	/* Destructor */

	~WorkerItem() {}

	/***************
	* G E T T E R
	***************/
	inline bool isReady() const {
		bool ready = _ready.load(std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_acquire);
		return ready;
	}

	inline unsigned size() const {
		return _numDOFs;
	}

	inline unsigned globGridId() const {
		return _globGridId;
	}

	inline unsigned globRecId() const {
		return _globRecId;
	}

	inline unsigned globLigId() const {
		return _globLigId;
	}

	inline unsigned devLocRecId() const {
		return _devLocRecId;
	}

	inline unsigned devLocLigId() const {
		return _devLocLigId;
	}

	inline unsigned devLocGridId() const {
		return _devLocGridId;
	}


	inline DOF* DOFBuffer() const {
		return _DOFs;
	}

	inline EnGrad* EnGradBuffer () const {
		return _EnGrads;
	}

	/***************
	* S E T T E R
	***************/
	inline void setReady() {
		std::atomic_thread_fence(std::memory_order_release);
		_ready.store(true,std::memory_order_relaxed);
	}

	inline void reset() {
		_ready = false;
	}

	inline void setNumDOFs(const unsigned& numDOFs) {
		_numDOFs = numDOFs;
	}

	inline void setGlobGridId(const unsigned& id) {
		_globGridId = id;
	}

	inline void setGlobRecId(const unsigned& id) {
		_globRecId = id;
	}

	inline void setGlobLigId(const unsigned& id) {
		_globLigId = id;
	}

	inline void setDevLocGridId(const unsigned& id) {
		_devLocGridId = id;
	}

	inline void setDevLocRecId(const unsigned& id) {
		_devLocRecId = id;
	}

	inline void setDevLocLigId(const unsigned& id) {
		_devLocLigId = id;
	}

	inline void setDOFBuffer(DOF* const buf) {
		_DOFs = buf;
	}

	inline void setEnGradBuffer(EnGrad* const buf) {
		_EnGrads = buf;
	}



	/****************************
	 * public member functions
	 ****************************/

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
	DOF* _DOFs;				/** Page-Locked Buffer (for input)*/
	EnGrad* _EnGrads;		/** Page-Locked Buffer (for output)*/
	unsigned _numDOFs;		/** Number of elements in buffers */

	unsigned _globGridId;	/** Global grid device ID */
	unsigned _globRecId;	/** Global receptor protein device ID */
	unsigned _globLigId;	/** Global ligand protein device ID */


	/* For device calcualtions */
	unsigned _devLocGridId;	/** Device local grid ID */
	unsigned _devLocRecId;	/** Device local receptor protein device ID */
	unsigned _devLocLigId;	/** Device local ligand protein ID */

	std::atomic<bool> _ready;
};

} // namespace


#endif /* WORKITEM_H_ */
