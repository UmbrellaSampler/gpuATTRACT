/*
 * NLGrid.cpp
 *
 *  Created on: Jan 14, 2015
 *      Author: uwe
 */

#include <cmath>

#include "as/NLGrid.h"
#include "asUtils/helper.h"

/* Constructor */
as::NLGrid::NLGrid(NLGridDesc desc):
		Grid::Grid(desc.width, desc.height,	desc.depth,
				make_float3(desc.posMin[0], desc.posMin[1], desc.posMin[2]),
				desc.gridSpacing),
		_grid(desc.grid),
		_numElInLists(desc.numEl),
		_neighborList(desc.neighborArray),
		_dPlateau2(desc.dPlateau*desc.dPlateau),
		_dPlateau2_inv(1/_dPlateau2),
		_dVox_inv(1/_dVox)
{
	_maxDim.x = pos().x + (_width  - 1) * _dVox;
	_maxDim.y = pos().y + (_height - 1) * _dVox;
	_maxDim.z = pos().z + (_depth  - 1) * _dVox;

	/* init ratios (taken from the original attract code) */
	int size  = int(10000*_dPlateau2);
	_ratio = new float[size+1];

	for (int n = 0; n <= size; n++) {
		double d2 = ((n + 0.5) / 10000);
		_ratio[n] = sqrt(d2 / _dPlateau2);
	}
}

/* Destructor */
as::NLGrid::~NLGrid() {
	freeHost();
}


/****************************
 * public member functions
 ****************************/
void as::NLGrid::freeHost() {
	delete[] _neighborList;
	delete[] _grid;
	delete[] _ratio;
}

double as::NLGrid::mySize(asUtils::sizeType_t type) const {

	unsigned long long sizeInByte;
	sizeInByte = (unsigned long long) (_numElInLists*sizeof(uint) +
			_width*_height*_depth*sizeof(NeighbourDesc));

	return sizeConvert(sizeInByte, type);
}

/****************************
 * protected member functions
 ****************************/

/****************************
 * private member functions
 ****************************/


