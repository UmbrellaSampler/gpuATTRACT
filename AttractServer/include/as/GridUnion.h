/*
 * GridUnion.h
 *
 *  Created on: Jan 13, 2015
 *      Author: uwe
 */

#ifndef GRIDUNION_H_
#define GRIDUNION_H_

#include <string>
#include "as/IntrplGrid.h"
#include "as/NLGrid.h"

namespace as {


/*
 ** @brief: Composition of Grids needed of core calculations
 ** Grid object allocation happens outside the scope of this
 ** class on the heap. Only freeing is performed.
 */
class GridUnion {
public:
	/* Constructor */
	GridUnion() : _innerGrid(nullptr), _outerGrid(nullptr),
            	  _NLGrid(nullptr) {}

	GridUnion(as::IntrplGrid* innerGrid, as::IntrplGrid* outerGrid,
              as::NLGrid* NLGrid) :
            	  _innerGrid(innerGrid), _outerGrid(outerGrid),
            	  _NLGrid(NLGrid) {}
	/* Destructor */
	~GridUnion() {
		if(_innerGrid!=nullptr) delete _innerGrid;
		if(_outerGrid!=nullptr) delete _outerGrid;
		if(_NLGrid   !=nullptr) delete _NLGrid;
	}

	/* == operator */
	bool operator ==( const GridUnion& rightSide) const {
			return _tag == rightSide._tag;
	}
	/***************
	* G E T T E R
	***************/
	inline IntrplGrid* innerGrid() const {
		return _innerGrid;
	}

	inline IntrplGrid* outerGrid() const {
		return _outerGrid;
	}

	inline NLGrid* NLgrid() const {
		return _NLGrid;
	}

	std::string tag() const {
		return _tag;
	}

	void freeHost() {

	}

	/***************
	* S E T T E R
	***************/

	void setTag(std::string tag) {
		_tag = tag;
	}

	void setInnerGrid(IntrplGrid* grid) {
		/* delete if already allocated to avoid memory leak */
		if(_innerGrid!=nullptr) delete _innerGrid;
		_innerGrid = grid;
	}

	void setOuterGrid(IntrplGrid* grid) {
		/* delete if already allocated to avoid memory leak */
		if(_outerGrid!=nullptr) delete _outerGrid;
		_outerGrid = grid;
	}

	void setNLGrid(NLGrid* grid) {
		/* delete if already allocated to avoid memory leak */
		if(_NLGrid!=nullptr) delete _NLGrid;
		_NLGrid = grid;
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

	IntrplGrid* _innerGrid;
	IntrplGrid* _outerGrid;
	NLGrid* _NLGrid;
	std::string _tag;
};

}


#endif /* GRIDUNION_H_ */
