/*
 * interpolation.h
 *
 *  Created on: Jan 22, 2015
 *      Author: uwe
 */

#ifndef INTERPOLATION_H_
#define INTERPOLATION_H_

#include "cuda_runtime.h"

#include "as/IntrplGrid.h"
#include "as/Protein.h"


namespace asCore {

/*
 ** @brief: Interpolation type.
 ** built_in: 	use cuda built_in interpolation routine
 ** manual: 	use "home-made" routine
 */
enum IntrplType {
	built_in,
	manual
};

template<IntrplType T>
__global__ void d_InnerPotForce(
		const unsigned gridId, const unsigned protId,
		const unsigned numDOFs,
		const float* data_in_x, const float* data_in_y, const float* data_in_z,
		float* data_out_x, float* data_out_y, float* data_out_z,
		float* data_out_eEl, float* data_out_eVdw);

template<IntrplType T>
__global__ void d_OuterPotForce(
		const unsigned gridId, const unsigned protId,
		const unsigned numDOFs,
		const float* data_in_x, const float* data_in_y, const float* data_in_z,
		float* data_out_x, float* data_out_y, float* data_out_z,
		float* data_out_eEl, float* data_out_eVdw);


void h_PotForce(const as::IntrplGrid* innerGrid,
		const as::IntrplGrid* outerGrid, const as::Protein* prot,
		const float* LigPosX,
		const float* LigPosY,
		const float* LigPosZ,
		float* data_out_x, float* data_out_y, float* data_out_z,
		float* data_out_eEl, float* data_out_eVdw);



}  // namespace asCore


#endif /* INTERPOLATION_H_ */
