/*
 * RequestHandler.cpp
 *
 *  Created on: Oct 2, 2015
 *      Author: uwe
 */

#include <iostream>
#include <cassert>
#include <algorithm>

#include <AttractServer>
#include "nvToolsExt.h"

#include "RequestHandler.h"
#include "Chunk.h"

using std::cerr;
using std::cout;
using std::endl;


void ema::RequestHandler::init(extServer& server, SolverType solverType, std::vector<extDOF>& dofs) {
	_server = &server;
	_solverType = solverType;


	/* Fill the object array */
	for (unsigned i = 0; i < dofs.size(); ++i) {
		SharedSolver ptr;

		switch (_solverType) {
		case SolverType::BFGS:
			ptr = std::make_shared<BFGSSolver>();
			break;
		case SolverType::unspecified:
			cerr << "Error: " << "SolverType is unspecified." << endl;
			exit(EXIT_FAILURE);
			break;
		default:
			cerr << "Error: " << "Unknown solver specification." << endl;
			exit(EXIT_FAILURE);
			break;
		}

		ptr->setState(dofs[i]);
		_objects.emplace_hint(_objects.end(), i, ptr);
	}

	/* set number of recieved objects */
	_numObjects = _objects.size();

	/*
	 * initialize chunk list based on the number of available structures
	 */


	/* shrink the number of chunks to fit the minimal chunkSize */
	while (_numObjects < _numChunks*_minChunkSize && _numChunks > 1) {
		 --_numChunks;
	}
	assert(_numChunks >= 1);

	/* calculate chunk sizes */
	unsigned base_numPerChunk = MIN(_numObjects,_numConcurrentObjects) / _numChunks;
	unsigned rest = MIN(_numObjects,_numConcurrentObjects) % _numChunks;

	unsigned chunkSizes[_numChunks];
	std::fill(chunkSizes, chunkSizes + _numChunks, base_numPerChunk);

	assert(rest < _numChunks);
	for (unsigned i = 0; i < rest; ++i) {
		++chunkSizes[i];
	}

	/* setup chunks and put them into chunk list */
	for (unsigned i = 0; i < _numChunks; ++i) {
		_chunkList.emplace_back();
	}

	unsigned count = 0;
	for (auto& chunk : _chunkList) {
		ObjMapIter mapIter = _objects.begin();
		for (unsigned i = 0; i < chunkSizes[count]; ++i, ++mapIter) {
			chunk.getContainer().push_back(std::move(*mapIter));
			_objects.erase(mapIter);
			assert(mapIter != _objects.end());
		}
		assert(count < _numChunks);
		++count;

	}

	//Debug
//	cerr << "_objects.size()=" << _numObjects << endl;
//	cerr << "_numChunks=" << _numChunks << endl;
//	unsigned i = 0;
//	for (auto& chunk : _chunkList) {
//		cerr << "chunk #" << i++ << "; size=" << chunk.size() << endl;
//		unsigned j = 0;
//		for (auto& obj : chunk.getContainer()) {
//			cerr << "obj#" << j++ << " "<<  obj.second->getState().transpose() << endl;
//		}
//	}
//	cerr << "_objects.size()=" << _objects.size() << endl;

}

//#define H_IO

void ema::RequestHandler::run() {

	_collectedRequests.reserve(_chunkList.begin()->size());
	_collectedResults.reserve(_chunkList.begin()->size());

//	RingArray<int> reqIds;

	/* initial loop: start solvers and collect first requests and submit*/
	for (auto& chunk : _chunkList) {
		nvtxRangePushA("Processing");
		for (auto& obj : chunk.getContainer()) {
			SharedSolver& solver = obj.second;
			solver->start();
			_collectedRequests.push_back(Vector2extDOF(solver->getState()));
		}
		nvtxRangePop();

		nvtxRangePushA("Submit");
//		int reqId = ema::server_submit(*_server, _collectedRequests.data(), _collectedRequests.size(),
//				_serverOpt.gridId, _serverOpt.recId, _serverOpt.ligId, _serverOpt.useMode);
		int reqId = asClient::server_submit(*_server, _collectedRequests.data(), _collectedRequests.size(),
				_serverOpt.gridId, _serverOpt.recId, _serverOpt.ligId, _serverOpt.useMode);
		nvtxRangePop();
		chunk.setFetchSize(chunk.size());

		if (reqId == -1) {
			cerr << "Error: Submitting request." << std::endl;
			std::exit(EXIT_FAILURE);
		}
//		reqIds.push_back(reqId);
		chunk.setReqId(reqId);

		_collectedRequests.resize(0);
	}


	unsigned count = 1;
	while(_finishedObjects.size () < _numObjects && count < 100000) {

//		auto ringIter =  reqIds.begin();
#ifdef H_IO
		cerr << endl;
		cerr << endl;
		cerr << "new Round #" << count << endl;
		cerr << "chunkSizes: ";
		for (auto& chunk : _chunkList) {
			cerr << chunk.size() << " ";
		}
		cerr << endl;
		cerr << "_finishedObjects.size()=" << _finishedObjects.size() << " _objects.size()=" << _objects.size() << " _numObjects="<< _numObjects <<  endl;

		// Debug
		cerr << endl;
		cerr << endl;
		cerr << "Chunk sizes initial, _chunkList.size()=" << _chunkList.size() << endl;
		for (auto& chunk : _chunkList) {
			cerr << chunk.size() << " ";
		}
		cerr << endl;
#endif


		/* load balancing */

		/* Adjust chunk sizes to balance the workload of each chunk.
		 * This happens in case that the global object list is empty and
		 * the chunks cannot be refilled by new initial configurations.
		 * Do it not each iteration */



		if (count%4 == 0 && true) {
			double ratio = chunkSizeRatio(_chunkList);
			if(_objects.empty() && ratio > 1.5) {
#ifdef H_IO
				cerr << "Load balance at a ratio of " << ratio << endl;

				cerr << "Chunk sizes before balancing" << endl;
				cerr << "_chunkList.size()="<< _chunkList.size() << endl;
				for (auto& chunk : _chunkList) {
					cerr << chunk.size() << endl;
				}

//				char dummy; std::cin >> dummy;
//				cerr << endl;
#endif

				loadBalanceChunks(_chunkList);

#ifdef H_IO
				cerr << "Chunk sizes after balancing" << endl;
				for (auto& chunk : _chunkList) {
					cerr << chunk.overAllSize() << endl;
				}

//				std::cin >> dummy;
				cerr << endl;
#endif
			} // if
		}


			for (auto chunkListIter = _chunkList.begin(); chunkListIter != _chunkList.end(); ) {
				auto& chunk = *chunkListIter;
#ifdef H_IO
				cerr << "chunk.fetchSize()="<< chunk.fetchSize() << endl;
#endif
	//			assert(chunk.fetchSize() > 0);

				_collectedRequests.resize(0);
				if (chunk.fetchSize() > 0) {
					_collectedResults.resize(chunk.fetchSize());


					/* Wait for requests */
					nvtxRangePushA("Waiting");
	//				unsigned count = ema::server_pull(*_server, chunk.reqId(), _collectedResults.data());
					unsigned count = asClient::server_pull(*_server, chunk.reqId(), _collectedResults.data());
					nvtxRangePop();

		//			cerr << endl;
		//			cerr << "\t" << "_collectedResults.size()=" << _collectedResults.size() << " chunk.size()=" << chunk.size() << endl;

					if (count >= 10000) {
						cerr << "Error: pulling for Request." << std::endl;
						std::exit(EXIT_FAILURE);
					}

					/* Assigne results */
					chunk.setResults(_collectedResults);
				}

			/* Check if other chunks assigned results (after loadbalancing)*/
			chunk.checkLBconts();

			/* Process chunk and remove converged structures */
			nvtxRangePushA("Processing");
			auto iter = chunk.getContainer().begin();
			iter = chunk.getContainer().begin();

			int takenObj = 0; // Debug
			int convergedObj = 0;


			while (iter != chunk.getContainer().end()) {
				SharedSolver& solver = iter->second;
				solver->step();

				/* test for convergence */
				if(solver->converged()) {
					++convergedObj; // Debug

					/* destroy coroutine context by calling finalize */
					solver->finalize();
					/* move structure/object in finished object container */
					_finishedObjects.insert(move(*iter)); // no copy-construction
					chunk.getContainer().erase(iter++);

					/* move new structure/solver from object map if any left*/
					if (!_objects.empty()) {
						++takenObj; // Debug

						ObjMapIter objIter = _objects.begin();
						iter = chunk.getContainer().insert(iter, std::move(*objIter));
						_objects.erase(objIter);

						/* prepare new solver */
						SharedSolver& newSolver = iter->second;
						newSolver->start();
						/* collect new request */
						_collectedRequests.push_back(Vector2extDOF(newSolver->getState()));
						++iter;
					}
				} else {
					/* collect new request */
					_collectedRequests.push_back(Vector2extDOF(solver->getState()));
					++iter;
				}

			}
			nvtxRangePop();
			assert(iter == chunk.getContainer().end());

			chunk.setFetchSize(chunk.size());

#ifdef H_IO
			cerr << "\t" << "convergedObj=" << convergedObj << " takenObj=" << takenObj << endl;
//			cerr << "\t" << "_collectedRequests.size()=" << _collectedRequests.size() << " chunk.size()=" << chunk.size() << endl;
#endif

			/* submit request */
			if (_collectedRequests.size() > 0) { // there is still something to submit
				nvtxRangePushA("Submit");
//				int reqId = ema::server_submit(*_server, _collectedRequests.data(), chunk.size(),
//						_serverOpt.gridId, _serverOpt.recId, _serverOpt.ligId, _serverOpt.useMode);
				int reqId = asClient::server_submit(*_server, _collectedRequests.data(), chunk.size(),
						_serverOpt.gridId, _serverOpt.recId, _serverOpt.ligId, _serverOpt.useMode);
				nvtxRangePop();

				if (reqId == -1) {
					cerr << "Error: Submitting request." << std::endl;
					std::exit(EXIT_FAILURE);
				}
				chunk.setReqId(reqId);
//				*ringIter = reqId;
//				++chunkListIter;
//				++ringIter;

			}

			++chunkListIter;
			// do not remove since objects might still reside in LBconts waiting for results from other chunks
			// do it in function load balance instead.
//			else { // remove the chunk
//
////				cerr << "\t" << "removing this chunk" << endl;
//				_chunkList.erase(chunkListIter++);
//			} // if

		} // for each chunk

		++count;



		// Debug

	} // while
	assert(_finishedObjects.size () == _numObjects);

}

void ema::RequestHandler::getResult(std::vector<extDOF>& dofs, std::vector<extEnGrad>& results) {
	// cerr << dofs.size() << " " << finishedObjects.size() << endl;
	dofs.resize(_finishedObjects.size());
	results.resize(_finishedObjects.size());
	for (unsigned i = 0; i < _finishedObjects.size(); ++i) {
		dofs[i] = Vector2extDOF(_finishedObjects[i]->getState());
		results[i] = ObjGrad2extEnGrad(_finishedObjects[i]->getObjective());
	}
}

void ema::RequestHandler::getResult(std::vector<extDOF>& dofs, std::vector<extEnGrad>& results, std::vector<std::unique_ptr<Statistic>>& stats) {
	getResult(dofs, results);
	stats.resize(_finishedObjects.size());
	for (unsigned i = 0; i < _finishedObjects.size(); ++i) {
		stats[i] = _finishedObjects[i]->getStats();
	}
}
