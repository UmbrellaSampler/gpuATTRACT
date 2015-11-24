/*
 * SolverFactoryImpl.cpp
 *
 *  Created on: Nov 24, 2015
 *      Author: uwe
 */

#include <iostream>
#include <cassert>

#include "SolverFactoryImpl.h"
#include "VA13Solver.h"
#include "BFGSSolver.h"




std::map<std::string, ema::SolverFactoryImpl::SolverType> ema::SolverFactoryImpl::_solverTypeMap =
	{
			{"VA13", SolverType::VA13},
			{"BFGS", SolverType::BFGS}
	};

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::unique_ptr<ema::SolverBase> ema::SolverFactoryImpl::createSolverByNameImpl(const std::string& name) {
	using namespace std;
	/* assert that map contains element of name */
	assert(_solverTypeMap.find(name) != _solverTypeMap.end());
	switch (_solverTypeMap[name] ) {
	case VA13:
		return make_unique<VA13Solver>();
		break;
	case BFGS:
		return make_unique<BFGSSolver>();
		break;
	default:
		cerr << "Error: " << "Unknown solver specification." << endl;
		exit(EXIT_FAILURE);
		break;
	}
}


