/*
 * RequestHandler.h
 *
 *  Created on: Oct 1, 2015
 *      Author: uwe
 */

#ifndef REQUESTHANDLER_H_
#define REQUESTHANDLER_H_

#include <vector>
#include <map>
#include <list>
#include <memory>
#include "SolverBase.h"
#include "BFGSSolver.h"
#include "Chunk.h"

#include "TestServer.h"

#include <AttractServer>

namespace ema {

constexpr unsigned maxConcurrentObjects = 20000; // default (16000) maximum number of running coroutines that may exist at the same time.
constexpr unsigned numChunks = 2; // default number of chunks running at the same time. Each chunk maintains maxConcurrentObjects/numChunks objects.
constexpr unsigned minChunkSize = 10; // minimum chunksize that is worth to work with

class RequestHandler {
public:

	RequestHandler() : _server(nullptr),
		_numObjects(0), _numConcurrentObjects(maxConcurrentObjects), _numChunks(numChunks),
		_minChunkSize(minChunkSize){};

	void setNumConcurrentObjects(unsigned value) {_numConcurrentObjects = value;}
	void setNumChunks(unsigned value) {_numChunks = value;}
	void setMinChunkSize(unsigned value) { _minChunkSize = value;}

	/*
	 ** @brief: Initializes the RequestHandler. Member run() may now be called.
	 */
	void init(extServer& server, std::string const& solverName, std::vector<extDOF>& dofs);

	void run();

	std::vector<extDOF> getResultStates();
	std::vector<extEnGrad> getResultEnGrads();
	std::vector<std::unique_ptr<Statistic>> getStatistics();

private:

	extServer* _server;

	using SharedSolver = std::shared_ptr<SolverBase>;

	using ObjMap = std::map<unsigned, SharedSolver>;
	using ObjMapIter = ObjMap::iterator;
	ObjMap _objects;
	ObjMap _finishedObjects;

	using ChunkIter = Chunk::iterator;
	std::list<Chunk> _chunkList;

	std::vector<extDOF> _collectedRequests;
	std::vector<extEnGrad> _collectedResults;

	unsigned _numObjects;
	unsigned _numConcurrentObjects;
	unsigned _numChunks;
	unsigned _minChunkSize;

public:
	struct ServerOptions {
		int gridId;
		int recId;
		int ligId;
		as::Request::useMode_t useMode;

		ServerOptions () :
			gridId(-1), recId(-1), ligId(-1), useMode(as::Request::useMode_t::unspecified) {}

		ServerOptions (int _gridId, int _recId, int _ligId, as::Request::useMode_t _useMode) :
			gridId(_gridId), recId(_recId), ligId(_ligId), useMode(_useMode) {}
	};

	void setServerOptions (ServerOptions opt) { _serverOpt = opt; }

protected:
	ServerOptions _serverOpt;
};

} // namespace


#endif /* REQUESTHANDLER_H_ */
