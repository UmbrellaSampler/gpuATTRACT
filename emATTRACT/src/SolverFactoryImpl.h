/*
 * SolverFactoryImpl.h
 *
 *  Created on: Nov 24, 2015
 *      Author: uwe
 */

#ifndef SOLVERFACTORYIMPL_H_
#define SOLVERFACTORYIMPL_H_

#include <map>

#include "SolverFactory.h"

namespace ema {

class SolverFactoryImpl : public SolverFactory {
private:
	std::unique_ptr<SolverBase> createSolverByNameImpl(const std::string& name) override;


	enum SolverType {
		VA13,
		BFGS,
		LBFGS_B,
		unspecified
	};
	static std::map<std::string, SolverType> _solverTypeMap;
};

}  // namespace ema




#endif /* SOLVERFACTORYIMPL_H_ */
