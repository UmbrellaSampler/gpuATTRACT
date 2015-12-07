
#ifndef DOFTRANSFORM_H_
#define DOFTRANSFORM_H_

#include <vector>

#include "as/asTypes.h"
#include "asUtils/Vec3.h"
#include "asUtils/RotMat.h"

namespace asClient {

	/*
	 ** @brief: This function performs an inplace transformation of the ligand coordinates assuming
	 ** that the receptor is always centered at the origin.
	 *
	 ** @input: [in] 	dof_rec: 	Vector of DOF-vectors of the receptor
	 ** 		[in+out]dof_lig: 	Vector of DOF-vectors of the ligand
	 ** 		[in]	pivot_rec:	Pivot of the receptor
	 ** 		[in]	pivot_lig:	Pivot of the ligand
	 ** 		[in]	center_rec: Are receptor coords centered?
	 ** 		[in]	center_lig: Are ligand coords centered?
	 **
	 ** 		pivots:	Vector of pivots of the receptor and the ligands
	 ** 				Can be obtained by calling asDB::readDOFHeader(...)
	 */
	void transformDOF_glob2rec(const std::vector<as::DOF>& dof_rec, std::vector<as::DOF>& dof_lig,
			const asUtils::Vec3f& pivot_rec, const asUtils::Vec3f& pivot_lig,
			bool centered_rec, bool centered_lig );

	void transformEnGrad_rec2glob(const std::vector<as::DOF>& dof_rec, std::vector<as::EnGrad>& enGrad_lig);

	/*
	 ** @brief: Get Euler angles from a rotation matrix.
	 */
	template<typename T>
	void rotmat2euler(const asUtils::RotMat<T>& rotmat, T& phi, T& ssi, T& rot);

	/*
	 ** @brief: get rotation matrix from Euler angles
	 */
	template<typename T>
	void euler2rotmat(const T& phi, const T& ssi, const T& rot, asUtils::RotMat<T>& rotmat);
} // namespace


#endif /* DOFTRANSFORM_H_ */
