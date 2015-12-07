
#ifndef GRID_H_
#define GRID_H_
 
#include "cuda_runtime.h"
#include "asUtils/asUtilsTypes.h"

namespace as{


class Grid {
public:
	/* Constructor */
	Grid(unsigned width, unsigned height, unsigned depth,
			float3 pos, float dVox) :
				_width(width), _height(height), _depth(depth),
				_pos(pos), _dVox(dVox) {}

	/* Destructor */
	virtual ~Grid() {}

	/***************
	* G E T T E R
	***************/

	inline unsigned width() const {
		return _width;
	}

	inline unsigned height() const {
		return _height;
	}

	inline unsigned depth() const {
		return _depth;
	}

	inline float dVox() const {
		return _dVox;
	}

	inline float3 pos() const {
		return _pos;
	}

	/***************
	* S E T T E R
	***************/
	virtual void setPos(float3 pos) {
		_pos = pos;
	}

	/****************************
	 * public member functions
	 ****************************/

	/*
	 ** @brief:returns the size in bytes (default)
	 */
	virtual double mySize(asUtils::sizeType_t = asUtils::byte) const = 0;

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

	unsigned _width; 	/** width of each grid in elements */
	unsigned _height; 	/** height of each grid in elements */
	unsigned _depth; 	/** depth of each grid in elements */

	float3 _pos;  	/** lower bound of grid coordinates */

	float _dVox;	/** voxel distance; grid spacing */

private:
	/****************************
	 * private member functions
	 ****************************/

	/****************************
	 * private member variables
	 ****************************/
};

} // namespace



#endif /* GRID_H_ */
