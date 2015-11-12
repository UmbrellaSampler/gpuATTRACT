/*
 * core_helpers.h
 *
 *  Created on: Jan 13, 2015
 *      Author: uwe
 */

#ifndef CORE_HELPERS_H_
#define CORE_HELPERS_H_

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "as/asTypes.h"

namespace asCore {
/*
 ** @brief: Prints the voxel coordinates and the respective coordinates.
 ** For debugging purposes mainly.
 */
inline void print(as::VoxelOctet oct) {
	using namespace std;
	int precisionSetting = cout.precision( );
	ios::fmtflags flagSettings = cout.flags();

	cout.setf(ios::fixed | ios::showpos | ios::showpoint);
	cout.precision(5);

	std::cout << "min " << oct.min.x << " " << oct.min.y << " " << oct.min.z << std::endl;
	std::cout << "max " << oct.max.x << " " << oct.max.y << " " << oct.max.z << std::endl;
	std::cout << std::endl;
	std::cout << setw(10) << oct.data[0][0][0].w << setw(10) << oct.data[1][0][0].w << std::endl;
	std::cout << std::endl;
	std::cout << setw(10) << oct.data[0][1][0].w << setw(10) << oct.data[1][1][0].w << std::endl;
	std::cout << "#####z++#####\n";
	std::cout << setw(10) << oct.data[0][0][1].w << setw(10) << oct.data[1][0][1].w << std::endl;
	std::cout << std::endl;
	std::cout << setw(10) << oct.data[0][1][1].w << setw(10) << oct.data[1][1][1].w << std::endl;

	// Restore io flags
	cout.precision(precisionSetting);
	cout.flags(flagSettings);
}

inline __device__ void cudaPrint(as::VoxelOctet oct, int i, float x, float y, float z) {
	printf("%i\n%.5f %.5f %.5f\nmin %.2f %.2f %.2f\nmax %.2f %.2f %.2f\n%.5f %.5f \n%.5f %.5f\n%.5f %.5f\n%.5f %.5f\n\n",
			i,
			x, y, z,
			oct.min.x, oct.min.y, oct.min.z,
			oct.max.x, oct.max.y, oct.max.z,
			oct.data[0][0][0].w, oct.data[1][0][0].w,oct.data[0][1][0].w, oct.data[1][1][0].w,
			oct.data[0][0][1].w, oct.data[1][0][1].w,oct.data[0][1][1].w, oct.data[1][1][1].w);

}


}// namespace asCore





#endif /* CORE_HELPERS_H_ */
