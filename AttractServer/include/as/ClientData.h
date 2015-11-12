/*
 * ClientData.h
 *
 *  Created on: Jan 13, 2015
 *      Author: uwe
 */

#ifndef CLIENTDATA_H_
#define CLIENTDATA_H_

#include <vector>
#include <cassert>

namespace as {

class ClientData {
public:
	/* Constructor */
	ClientData() : _proteins(0), _grids(0), _valid(false) {}

	/* Destructor */
	~ClientData() {}

	/***************
	* G E T T E R
	***************/
	inline int globalProteinId(const int &clientLocalId) const {
		assert(clientLocalId < static_cast<int>(_proteins.size()));
		return _proteins[clientLocalId];
	}

	inline int globalGridId(const int &clientLocalId) const {
		assert(clientLocalId < static_cast<int>(_grids.size()));
		return _grids[clientLocalId];
	}

	bool isValid() const {
		return _valid;
	}

	std::vector<int>& proteins() {
		return _proteins;
	}

	std::vector<int>& grids() {
		return _grids;
	}

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/

	void addProtein(int protId) {
		assert(protId >= 0);
		_proteins.push_back(protId);
		if (_valid == false) {
			_valid = true;
		}
	}

	void addGrid(int gridId) {
		assert(gridId >= 0);
		_grids.push_back(gridId);
		if (_valid == false) {
			_valid = true;
		}
	}

	void reset() {
		_proteins.clear();
		_grids.clear();
		_valid = false;
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

	std::vector<int> _proteins; /** Global protein Ids */
	std::vector<int> _grids; 	/** Global grid Ids */

	bool _valid;
};

} //namespace


#endif /* CLIENTDATA_H_ */
