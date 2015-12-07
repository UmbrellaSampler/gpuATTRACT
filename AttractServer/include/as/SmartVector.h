
#ifndef SMARTVECTOR_H_
#define SMARTVECTOR_H_

#include <deque>
#include <mutex>
#include <cassert>
#include <cstring>
#include <iostream>

namespace as {


template<class T>
class SmartVector {
public:
	/* Constructor */
	SmartVector() : _vec() {}
	SmartVector(unsigned size) : _vec(size)
	{
		for(auto i : _vec) {
			setInvalid(i);
		}
	}
	SmartVector(unsigned size, T value) : _vec(size, value) {}

	/* Destructor */

	inline T& operator[] (const int& loc) {
		assert(loc < _vec.size());
		assert(isValid(_vec[loc]));
		return _vec[loc];
	}

	inline const T& operator[] (const int& loc) const {
		assert(loc < _vec.size());
		assert(isValid(_vec[loc]));
		return _vec[loc];
	}

	/***************
	* G E T T E R
	***************/
	inline unsigned realSize() const {
		return _vec.size();
	}

	inline unsigned size() {
		unsigned c = 0;
		for (auto i : _vec) {
			if (isValid(i))
				++c;
		}
		return c;
	}



	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	int getFirstEmptyLoc( ) {
		int loc = -1;
		if (_vec.size() == 0) {
			loc = 0;
		} else {
			int i;
			for (i = 0; i < _vec.size(); ++i) {
				if (!(isValid(_vec[i])) ) {
					loc = i;
					break;
				}
			}
			if (loc == -1) {
				loc = _vec.size();
			}
		}
		assert(loc >= 0);
		return loc;
	}

	int getLastValidLoc() {
		for(int i = 1; i <= _vec.size(); ++i) {
			if (isValid(_vec[_vec.size() - i ])) {
				return _vec.size() - i;
			}
		}

		return -1;
	}

	int getFirstValidLoc() {
		for(int i = 0; i < _vec.size(); ++i) {
			if (isValid(_vec[i])) {
				return i;
			}
		}

		return -1;
	}

	int placeAtFirstEmptyLoc( const T& obj) {
		int loc = -1;
		if (_vec.size() == 0) {
			_vec.push_back(obj);
			loc = 0;
		} else {
			int i;
			for (i = 0; i < _vec.size(); ++i) {
				if (!(isValid(_vec[i])) ) {
					_vec[i] = obj;
					loc = i;
					break;
				}
			}
			if (loc == -1) {
				_vec.push_back(obj);
				loc = _vec.size()-1;
			}
		}
		assert(loc >= 0);
		return loc;
	}
	
	/* overwrites existing object at location */
	void placeAtLoc( const T& obj, const unsigned& loc) {
		unsigned currentSize = _vec.size();
		if ( loc >= _vec.size() ) {
			_vec.resize(loc + 1);
			for(unsigned i = currentSize; i < _vec.size(); ++i) {
				setInvalid(_vec[i]);
			}
		}
		_vec[loc] = obj;
	}

	T getAndRemove(const int& loc) {
		assert(loc < _vec.size());
		assert(isValid(_vec[loc]));
		T obj = _vec[loc];
		setInvalid(_vec[loc]);
		return obj;
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

	bool isValid(const T& obj) const {
		assert(sizeof(T) >= 4*sizeof(char));
		return memcmp(&obj, &_valid, 4*sizeof(char)) != 0;
	}

	void setInvalid(T& obj) {
		memset(&obj, 0, sizeof(T));
		memcpy(&obj, &_valid, 4*sizeof(char));
	}

	std::deque<T> _vec;
	std::mutex _m;

	const char _valid[4] = {'k', 'x', 'p', 'w'};

};

}

#endif /* SMARTVECTOR_H_ */
