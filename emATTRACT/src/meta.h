/*
 * meta.h
 *
 *  Created on: Sep 24, 2015
 *      Author: uwe
 */

#ifndef META_H_
#define META_H_

#include <Eigen/Core>
#include <as/asTypes.h>
#include "as/ServerManagement.h"


#define OBJGRAD(dof, energy)	\
	do { 						\
		/*std::cout << "\t" << "request=" << Vector2extDOF(dof) << std::endl;*/ \
		state = dof;			\
		ca();					\
		energy = objective;		\
		/*std::cout << "\t" << "result=" << ObjGrad2extEnGrad(energy) << std::endl;*/						\
	} while(0)


#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

#ifndef MAX
#define MAX(x,y) ((x < y) ? y : x)
#endif

namespace ema {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Scalar = Eigen::VectorXd::Scalar;

using extDOF = as::DOF;
using extEnGrad = as::EnGrad;
using extServer = as::ServerManagement;
//class TestServer;
//using extServer = TestServer;


struct ObjGrad {
	double obj; // function value
	Vector grad; // gradients

	ObjGrad() = default;

	ObjGrad(const ObjGrad&) = default;
	ObjGrad& operator=(const ObjGrad&) = default;

	ObjGrad(ObjGrad&&) = default;
	ObjGrad& operator=(ObjGrad&&) = default;
};

inline Vector extDOF2Vector(const extDOF& dof) {
	Vector vec(6);
	vec  << dof.ang.x, dof.ang.y, dof.ang.z,
			dof.pos.x, dof.pos.y , dof.pos.z;
	return vec;
}

inline extDOF Vector2extDOF(const Vector& vec) {
	extDOF dof;
	dof.ang.x = vec(0);
	dof.ang.y = vec(1);
	dof.ang.z = vec(2);
	dof.pos.x = vec(3);
	dof.pos.y = vec(4);
	dof.pos.z = vec(5);
	return dof;
}

//inline ObjGrad extEnGrad2ObjGrad (const extEnGrad enGrad) {
//	ObjGrad objGrad;
//	objGrad.obj = enGrad.E_VdW + enGrad.E_El;
//	objGrad.grad = Vector(6);
//	objGrad.grad  << enGrad.ang.x, enGrad.ang.y, enGrad.ang.z,
//			enGrad.pos.x, enGrad.pos.y , enGrad.pos.z;
//	return objGrad;
//}

// for ATTRACT multiply gradients by -1.0
inline ObjGrad extEnGrad2ObjGrad (const extEnGrad enGrad) {
	ObjGrad objGrad;
	objGrad.obj = enGrad.E_VdW + enGrad.E_El;
	objGrad.grad = Vector(6);
	objGrad.grad  << -enGrad.ang.x, -enGrad.ang.y,  -enGrad.ang.z,
					 -enGrad.pos.x, -enGrad.pos.y , -enGrad.pos.z;
	return objGrad;
}

inline extEnGrad ObjGrad2extEnGrad(const ObjGrad& objGrad) {
	extEnGrad enGrad;
	enGrad.E_VdW = objGrad.obj;
	enGrad.E_El = 0.0; // TODO This is not true, we just lost the information
	enGrad.ang.x = objGrad.grad(0);
	enGrad.ang.y = objGrad.grad(1);
	enGrad.ang.z = objGrad.grad(2);
	enGrad.pos.x = objGrad.grad(3);
	enGrad.pos.y = objGrad.grad(4);
	enGrad.pos.z = objGrad.grad(5);
	return enGrad;
}

} //namespace

#endif /* META_H_ */
