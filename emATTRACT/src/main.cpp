
#include <iostream>
#include <sstream>
#include <list>
#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <cuda_runtime.h>

#include <AttractServer>
#include "asUtils/Logger.h"
#include "asClient/DOFTransform.h"
#include "asUtils/timer.h"

#include "tclap/CmdLine.h"

#include "RequestHandler.h"
#include "SolverBase.h"
#include "VA13Solver.h"

using namespace std;

/* global logger */
namespace ema {
std::unique_ptr<Log::Logger> log;
}

void init_logger( bool use_file = true) {
	using namespace Log;
#ifndef NDEBUG
	logLevel level = Debug;
#else
	logLevel level = Info;
#endif
	if (use_file) {
		string filename = "emATTRACT.log";
		ema::log.reset(new Logger(level, filename.substr(0,filename.size()-4)));
	} else {
		ema::log.reset(new Logger(level, &(std::cerr)));
	}
}


/* printing results to stdout */
void printResultsOutput(unsigned numDofs, as::DOF* dofs, as::EnGrad* enGrads, std::vector<asUtils::Vec3f>& pivots);

int main (int argc, char *argv[]) {
	using namespace std;
	using ema::log;

	/* initialize Logger */
	bool use_file = true;
	init_logger(use_file);

	/* required variables */
	string dofName;

	/* optional variables */
	string gridFileName;
	string ligFileName;
	string recFileName;
	string paramsFileName;

	string solverName;

	unsigned numCPUs;
	unsigned chunkSize;
	vector<int> devices;

	/* for request Handler */
	unsigned rh_maxNumConcurrentObjects;
	unsigned rh_numChunks;
//	unsigned minChunkSize;

	int numToConsider;
	int whichToTrack;

	/* catch command line exceptions */
	try {

		/* print argv */
		log->info() << "Client starts with command: ";
		std::vector<std::string> arguments(argv , argv + argc);
		for (auto string : arguments) { *log << string << " ";}
		*log << endl;

		/* description of the application */
		stringstream desc;
		desc << "An ATTRACT client that performs energy minimization using a BFGS method quasi-Newton method. "
				<< "It is not yet optimized for multi-GPU Systems";
		TCLAP::CmdLine cmd(desc.str(), ' ', "1.1");

		/* define required arguments */
		TCLAP::ValueArg<string> dofArg("","dof","",true,"Structure (DOF) file.","*.dat", cmd);


		/* define optional arguments */
		TCLAP::ValueArg<string> recArg("r","receptor-pdb","pdb-file name of receptor. (Default: receptorr.pdb)", false,"receptorr.pdb","*.pdb", cmd);
		TCLAP::ValueArg<string> ligArg("l","ligand-pdb","pdb-file name of ligand. (Default: ligandr.pdb)", false, "ligandr.pdb","*.pdb", cmd);
		TCLAP::ValueArg<string> gridArg("g","grid","Receptor grid file. (Default: receptorgrid.grid)",false, "receptorgrid.grid","*.grid", cmd);
		TCLAP::ValueArg<string> paramArg("p","par","Attract parameter file. (Default: attract.par)",false,"attract.par","*.par", cmd);

		vector<string> allowedSolvers = {"VA13", "BFGS", "LBFGS-B"};
		TCLAP::ValuesConstraint<string> vc_solvers(allowedSolvers);
		TCLAP::ValueArg<string> solverTypeArg("s","solverType","Solver type. Available solvers: VA13|BFGS|LBFGS-B (Default: VA13)",false,"VA13",&vc_solvers, cmd);

		TCLAP::ValueArg<unsigned> cpusArg("c","cpus","Number of CPU threads to be used. (Default: 0)", false, 0, "uint");

		int numDevicesAvailable; cudaVerify(cudaGetDeviceCount(&numDevicesAvailable));
		vector<int> allowedDevices(numDevicesAvailable); iota(allowedDevices.begin(), allowedDevices.end(), 0);
		TCLAP::ValuesConstraint<int> vc(allowedDevices);
		TCLAP::MultiArg<int> deviceArg("d","device","Device ID of serverMode to be used. Must be between 0 and the number of available GPUs minus one.", false, &vc);

		TCLAP::ValueArg<unsigned> chunkSizeArg("","chunkSize", "Number of concurrently processed structures at the server. (Default: 5000)", false, 5000, "uint", cmd);

		TCLAP::ValueArg<unsigned> rq_maxConcObjsArg("","maxConcurrency", "Max. number of concurrent structures that may be processed at the same time. (Default: 16000)", false, 16000, "uint", cmd);
		TCLAP::ValueArg<unsigned> rq_numChunksArg("","numChunks", "Number of request chunks. (Default: 2)", false, 2, "uint", cmd);
		TCLAP::ValueArg<int> num2ConsiderArg("","num", "Number of configurations to consider (1 - num). (Default: All)", false, -1, "int", cmd);
		TCLAP::ValueArg<int> which2TrackArg("","focusOn", "Condider only this configuration. (Default: -1)", false, -1, "int", cmd);


		cmd.xorAdd(cpusArg, deviceArg);

		// parse cmd-line input
		cmd.parse(argc, argv);

		/* Assigne parsed values */
		recFileName 	= recArg.getValue();
		ligFileName 	= ligArg.getValue();
		gridFileName 	= gridArg.getValue();
		paramsFileName 	= paramArg.getValue();
		dofName 	= dofArg.getValue();
		devices 	= deviceArg.getValue();
		numCPUs 	= cpusArg.getValue();
		chunkSize 	= chunkSizeArg.getValue();
		rh_maxNumConcurrentObjects = rq_maxConcObjsArg.getValue();
		rh_numChunks = rq_numChunksArg.getValue();
		numToConsider = num2ConsiderArg.getValue();
		whichToTrack = which2TrackArg.getValue();
		solverName = solverTypeArg.getValue();

	} catch (TCLAP::ArgException &e){
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
	}

	log->info() << "recName=" << recFileName 		<< endl;
	log->info() << "ligName=" << ligFileName 		<< endl;
	log->info() << "gridName=" << gridFileName 		<< endl;
	log->info() << "parName=" << paramsFileName 	<< endl;
	log->info() << "dofName=" << dofName	 	<< endl;
	log->info() << "numCPUs=" << numCPUs 		<< endl;
	log->info() << "devices=[ "; for (auto device : devices) *log << device << " "; *log << "]"<<  endl;
	log->info() << "chunkSize=" << chunkSize 	<< endl;
	log->info() << "rh_maxNumConcurrentObjects=" << rh_maxNumConcurrentObjects << endl;
	log->info() << "rh_numChunk=" << rh_numChunks << endl;
	log->info() << "numToConsider=" << numToConsider << endl;
	log->info() << "whichToTrack=" << whichToTrack << endl;
	log->info() << "solverName=" << solverName << endl;

	/* check if cpu or gpu is used */
	as::Request::useMode_t serverMode = as::Request::unspecified;
	if(numCPUs > 0) {
		serverMode = as::Request::CPU;
	} else if (devices.size() > 0) {
		serverMode = as::Request::GPU;
	} else {
		log->error() << "Neither CPU nor GPU is specified. This state should not happen." << endl;
		exit(EXIT_FAILURE);
	}


	/* read dof header */
	std::vector<asUtils::Vec3f> pivots;
	bool autoPivot;
	bool centered_receptor, centered_ligands;
	asDB::readDOFHeader(dofName, pivots, autoPivot, centered_receptor, centered_ligands);

	/* check file. only a receptor-ligand pair (no multi-bodies!) is allowed */
	if(!autoPivot && pivots.size() > 2) {
		log->error() << "DOF-file contains defintions for more than two molecules. Multi-body docking is not supported." << endl;
		exit(EXIT_FAILURE);
	}

	/* read dofs */
	std::vector<std::vector<as::DOF>> DOF_molecules;
	asDB::readDOFFromFile(dofName, DOF_molecules);

	/* check file. only one receptor-ligand pair (no multi-bodies!) is allowed */
	if(DOF_molecules.size() != 2) {
		log->error() << "DOF-file contains defintions for more than two molecules. Multi-body docking is not supported." << endl;
		exit(EXIT_FAILURE);
	}

	/* shrink number of dofs artificially */
	if (numToConsider >= 0 || whichToTrack >= 0) {
		if (whichToTrack >= 0) {
			DOF_molecules[1][0] = DOF_molecules[1][whichToTrack];
			DOF_molecules[1].resize(1);
			DOF_molecules[0].resize(1);
		} else {
			DOF_molecules[1].resize(numToConsider);
			DOF_molecules[0].resize(numToConsider);
		}
	}

	/*
	 * initialize the scoring server: mngt(numItems, chunkSize, deviceBufferSize )
	 * only a maximum number of items per request is allowed since too many items introduce a large overhead
	 */

	unsigned ligandSize = asDB::readProteinSizeFromPDB(ligFileName);
	unsigned deviceBufferSize = ligandSize*chunkSize;
	unsigned numDofs = DOF_molecules[0].size();
	const unsigned numItems = (rh_maxNumConcurrentObjects + chunkSize - 1) / chunkSize;

	log->info() << "ligandSize=" 		<< ligandSize 		<< endl;
	log->info() << "deviceBufferSize=" 	<< deviceBufferSize << endl;
	log->info() << "numDofs=" 			<< numDofs 			<< endl;
	log->info() << "numItems=" 			<< numItems 		<< endl;

	/* check if there are too two many items per request */
	constexpr unsigned maxItemsPerSubmit = 10000;
	if ((unsigned)ceil((double)numItems/rh_numChunks) > maxItemsPerSubmit) {
		log->error() << "Too many items per request. Increase chunkSize" << endl;
		exit(EXIT_FAILURE);
	}

	as::ServerManagement server(numItems, chunkSize, deviceBufferSize);

	/* load proteins and grid, and get a handle to it*/
	const int clientId = 0; // by specifing a client id, we may remove all added data by the by a call to removeClient(id)
	int ligId, recId, gridId;
	recId = server.addProtein(clientId, recFileName);
	ligId = server.addProtein(clientId ,ligFileName);
	gridId = server.addGridUnion(clientId, gridFileName);
	server.addParamTable(paramsFileName);

	/* parse or get pivots. only two pivots/molecules are allowed */
	if(autoPivot) {
		if (!pivots.empty()) {
			log->error() << "Auto pivot specified, but explicitly definined pivots available. (File "<< dofName << ")" << endl;
			exit(EXIT_FAILURE);
		}
		pivots.push_back(server.getProtein(recId)->pivot());
		pivots.push_back(server.getProtein(ligId)->pivot());
	} else {
		if (pivots.size() != 2) {
			log->error() << "No auto pivot specified, but number of definined pivots is incorrect. (File "<< dofName << ")" << endl;
			exit(EXIT_FAILURE);
		}
		server.getProtein(recId)->pivotize(pivots[0]);
		server.getProtein(ligId)->pivotize(pivots[1]);
	}
	log->info() << "pivots= "; for (auto pivot : pivots) *log << pivot << ", "; *log << endl;

	/* transform ligand dofs assuming that the receptor is always centered in the origin */
	asClient::transformDOF_glob2rec(DOF_molecules[0], DOF_molecules[1], pivots[0], pivots[1], centered_receptor, centered_ligands);

	/* adapt grid locations according to receptor pivot (actually, I don't get why we need this) */
	as::GridUnion* grid = server.getGridUnion(gridId);
	asUtils::Vec3f& pivot = pivots[0];
	float3 pos;

	pos = grid->innerGrid()->pos();
	pos.x -= pivot[0]; pos.y -= pivot[1]; pos.z -= pivot[2];
	grid->innerGrid()->setPos(pos);
//		as::print(grid->innerGrid(), 0, 0 , 0,0, 25, 30, 25, 30, 25, 30);

	pos = grid->outerGrid()->pos();
	pos.x -= pivot[0]; pos.y -= pivot[1]; pos.z -= pivot[2];
	grid->outerGrid()->setPos(pos);

	pos = grid->NLgrid()->pos();
	pos.x -= pivot[0]; pos.y -= pivot[1]; pos.z -= pivot[2];
	grid->NLgrid()->setPos(pos);

	/* initialize CPU workers if any*/
	for(unsigned i = 0; i < numCPUs; ++i) {
		server.addCPUWorker();
	}

	/* initialize devices and GPU workers if any */
	for (unsigned i = 0; i < devices.size(); ++i) {
		server.attachProteinToDevice(recId, devices[i]);
		server.attachProteinToDevice(ligId, devices[i]);
		server.attachGridUnionToDevice(gridId, devices[i]);
		server.attachParamTableToDevice(devices[i]);
		server.addGPUWorker(devices[i]);
	}

	/* Finalize server initialization */
	if (devices.size() > 0) {
		server.updateDeviceIDLookup();
	}

	/*
	 ** The server and the data is now initialized. We are ready to use it.
	 ** Next we need to initialize the client.
	 */

	ema::RequestHandler reqHandler;
	reqHandler.setNumChunks(rh_numChunks);
	reqHandler.setNumConcurrentObjects(rh_maxNumConcurrentObjects);
	reqHandler.setServerOptions({gridId, recId, ligId, serverMode});
	reqHandler.init(server, solverName, DOF_molecules[1]);
	ema::SolverBase::enableStats();
	reqHandler.run();

	vector<as::EnGrad> enGrads(numDofs);
	vector<std::unique_ptr<ema::Statistic>> stats;
	reqHandler.getResult(DOF_molecules[0], enGrads, stats);

	unsigned num_objEval = 0;
	for(unsigned i = 0; i < DOF_molecules[0].size(); ++i) {
		ema::Statistic* stat = stats[i].get();
		if(stat) num_objEval += stat->numRequests;
	}

	cerr << "Total Statistic" << endl;
	cerr << "num_objEval=" << num_objEval << "(av " << double(num_objEval) / DOF_molecules[0].size() << ")"  << endl;
//	cerr << "num_gradTolerance=" << num_gradTolerance << endl;
//	cerr << "num_finitePrec=" << num_finitePrec << endl;
//	cerr << "num_maxIter=" << num_maxIter << endl;



	/* print results to stdout*/
	printResultsOutput(numDofs, DOF_molecules[0].data(), enGrads.data(), pivots);


	/* remove all data from host and devices that correspond to the client id */
	server.removeClient(clientId);


	log->info() << "exit" << endl;

	return 0;
}

/* printing results to stdout */
void printResultsOutput(unsigned numDofs, as::DOF* dofs, as::EnGrad* enGrads, std::vector<asUtils::Vec3f>& pivots)
{
	using namespace std;

	int precisionSetting = cout.precision( );
	ios::fmtflags flagSettings = cout.flags();
	cout.setf(ios::showpoint);
	cout.precision(6);

	/* print header */
	cout << "#pivot 1 " << pivots[0][0] << " " << pivots[0][1] << " " << pivots[0][2] << " " << endl;
	cout << "#pivot 2 " << pivots[1][0] << " " << pivots[1][1] << " " << pivots[1][2] << " " << endl;
	cout << "#centered receptor: true" << endl;
	cout << "#centered ligands: true" << endl;
	for (unsigned i = 0; i < numDofs; ++i) {
		const as::EnGrad& enGrad = enGrads[i];
		const as::DOF& dof = dofs[i];
		cout << "#"<< i+1 << endl;
		cout << "## Energy: " << enGrad.E_VdW + enGrad.E_El << endl;
		cout << "## " << enGrad.E_VdW << " " << enGrad.E_El << endl;
		cout << 0.0 << " " << 0.0 << " " << 0.0 << " "
			 << 0.0 << " " << 0.0 << " " << 0.0 << endl;
		cout << dof.ang.x << " " << dof.ang.y << " " << dof.ang.z << " "
			 << dof.pos.x << " " << dof.pos.y << " " << dof.pos.z << endl;
	}

	cout.precision(precisionSetting);
	cout.flags(flagSettings);
}


