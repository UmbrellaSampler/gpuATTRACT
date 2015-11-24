/*
 * SolverFactory.h
 *
 *  Created on: Nov 24, 2015
 *      Author: uwe
 */

#ifndef SOLVERFACTORY_H_
#define SOLVERFACTORY_H_

#include <string>
#include <memory>

#include "SolverBase.h"

namespace ema {

class SolverFactory {
public:
	virtual ~SolverFactory(){};

	std::unique_ptr<SolverBase> createSolverByName (const std::string& name) {
		return createSolverByNameImpl(name);
	}

private:
	virtual std::unique_ptr<SolverBase> createSolverByNameImpl(const std::string& name) = 0;
};

}  // namespace ema



#endif /* SOLVERFACTORY_H_ */
