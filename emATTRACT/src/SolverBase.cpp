/*
 * SolverBase.cpp
 *
 *  Created on: Sep 28, 2015
 *      Author: uwe
 */

#include <cassert>

#include "SolverBase.h"
#include "BFGSSolver.h"


void ema::SolverBase::start() {
	assert(state.rows() > 0);
//	coro = std::move(coro_t(std::bind(&SolverBase::run, this, std::placeholders::_1)));
	coro =  new coro_t(std::bind(&SolverBase::run, this, std::placeholders::_1));
}

void ema::SolverBase::step() {
	assert(this->converged() == false);
	/* make sure that objective is already set !!! */
	(*coro)();
	if (stats && !converged()) {
		if (BFGSStatistic* stats = dynamic_cast<BFGSStatistic*>(internal_getStats())) {
			++stats->numRequests;
		}
	}
}

void ema::SolverBase::finalize() {
	if (coro) {
		delete coro;
		coro = nullptr;
	}
}

bool ema::SolverBase::stats = false;
