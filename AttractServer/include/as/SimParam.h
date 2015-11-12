/*
 * SimParam.h
 *
 *  Created on: Sep 1, 2015
 *      Author: uwe
 */

#ifndef SIMPARAM_H_
#define SIMPARAM_H_

#include "config.h"

namespace as {

class SimParam {
public:
	dielec_t dielec = variable;		/** type of dielectric constant */
	float epsilon = 15;				/** dielectric constant */
	float ffelec = FELEC/epsilon;	/** precomputed factor felec/epsilon */
//	bool  useSwi;					/** using switching potential */
//	float swiOn;					/** min. switching potential distance */
//	float swiOff;					/** max. switching potential distance */
	bool useRecGrad = false;		/** using Receptor gradients */
	bool usePot = true;				/** use Potential grid */
};

}  // namespace AServ





#endif /* SIMPARAM_H_ */
