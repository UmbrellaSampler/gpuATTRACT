/*******************************************************************************
 * gpuATTRACT framework
 * Copyright (C) 2015 Uwe Ehmann
 *
 * This file is part of the gpuATTRACT framework.
 *
 * The gpuATTRACT framework is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * The gpuATTRACT framework is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *******************************************************************************/

#include <iostream>
#include <sstream>
#include <list>
#include <cassert>
#include <random>
#include <cmath>
#include <memory>
#include <string>
#include <nvToolsExt.h>
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <AttractServer>
#include "asUtils/Logger.h"
#include "asClient/DOFTransform.h"
#include "asUtils/timer.h"

#include "ensembleWeightTable.h"
#include "RNG.h"
#include "tclap/CmdLine.h"

using namespace std;

namespace mca {
Log::Logger* _log;
}


void init_logger( bool use_file = true) {
	using namespace Log;
#ifndef NDEBUG
	logLevel level = Debug;
#else
	logLevel level = Info;
#endif
	if (use_file) {
		string filename = "mcATTRACT.log";
		mca::_log = new Logger(level, filename.substr(0,filename.size()-4));
	} else {
		mca::_log = new Logger(level, &(std::cerr));
	}
}


static double maxDist;
static double maxAng;
static double kT;
static double probApplyEnsembleMove;
static double cumulativeSamplingWeights[3] = {5.0, 5.0, 10.0};
static int ensembleSizes[2];
static double ub;
static double k;
static unsigned seed;
static vector<double> dseeds;

extern "C" void mc_ensemble_move_(int const& cartstatehandle, int const& nlig, int const& fixre,
		int const& iori,int const& itra, int* ens, int* nrens,
		double const& ensprob,
		double* phi, double* ssi, double* rot, double*xa, double* ya, double* za,
		double const& scalecenter,double const& scalerot,
		double* cumu_sws, int& mover, double& dseed);

//      nlig: number of binding partners, it is 2 in 2-body docking
//      fixre: if receptor is fixed
//      itra: if translation is applied
//      iori: if orientation is changed, or rotation is applied
//      mover: applied mover index, needed later to count how many times rigid-body-mover applied
//      mover: 1 - rigid-body-mover alone, 2 - enstrans only, 3 - rigid-body-mover and enstrans simutaneously
//      ens: size maxlig, model index for each ligand(binding partner)
//      nrens: size maxlig, model number for each binding partner
//      scalecenter: magnitude for rigid-body translation
//      scalerot: magnitude for rigid-body rotation
//      cumu_sws: cumulative sampling weights for movers, options in the main code to define the sampling weights for each 3 movers
//      dseed: random number seed, it is neccessary to maintail the dseed in the main code, otherwise it will be needed to initialize each time here

inline void randomStep (const as::DOF& oldDOF, as::DOF& newDOF)
{
	constexpr int cartstatehandle = -99; // dummy
	constexpr int nlig = 2;
	constexpr int fixre = 1;
	constexpr int iori = 1;
	constexpr int itra = 1;
	int ens[nlig] = {oldDOF.recId + 1, oldDOF.ligId - (ensembleSizes[0] - 1)};
	int nrens[nlig] = {ensembleSizes[0], ensembleSizes[1]};
	double ensprob = probApplyEnsembleMove;
	double phi[nlig] = {0.0, oldDOF.ang.x};
	double ssi[nlig] = {0.0, oldDOF.ang.y};
	double rot[nlig] = {0.0, oldDOF.ang.z};
	double xa[nlig]  = {0.0, oldDOF.pos.x};
	double ya[nlig]  = {0.0, oldDOF.pos.y};
	double za[nlig]  = {0.0, oldDOF.pos.z};
	double scalecenter = maxDist;
	double scalerot = maxAng;
	double* cumu_sws = cumulativeSamplingWeights;
	int mover = 0;

#ifdef _OPENMP
	int tid = omp_get_thread_num();
#else
	constexpr int tid = 0;
#endif
	double& dseed = dseeds[tid];

//	cout << "OLD" << endl;
//	cout << oldDOF << endl;

	mc_ensemble_move_(cartstatehandle, nlig, fixre, iori, itra, ens,
			nrens, ensprob, phi, ssi, rot, xa, ya, za,
			scalecenter, scalerot, cumu_sws, mover, dseed);

	/* TODO: count mover values */


	newDOF.ang.x = phi[1];
	newDOF.ang.y = ssi[1];
	newDOF.ang.z = rot[1];
	newDOF.pos.x = xa[1];
	newDOF.pos.y = ya[1];
	newDOF.pos.z = za[1];
	newDOF.recId = ens[0] - 1;
	newDOF.ligId = ens[1] + (ensembleSizes[0] - 1);

//	cout << "NEW" << endl;
//	cout << newDOF << endl;

}

inline void applyConstraints(const as::DOF& dof, as::EnGrad& enGrad) {
	double dx2 = dof.pos.x*dof.pos.x + dof.pos.y*dof.pos.y + dof.pos.z*dof.pos.z;
	double dx = std::sqrt(dx2);
	if (dx > ub) {
		double d = dx - ub;
		enGrad.E_El += k*d*d;
	}
}

/* Initialize random number generators */
static std::default_random_engine generator;
static std::uniform_real_distribution<double> distribution(0.0, 1.0);

static std::vector<mca::default_RNG> RNGs;

void MC_accept(as::DOF& oldDOF, as::EnGrad& oldEnGrad, as::DOF &newDOF, as::EnGrad& newEnGrad) {
	float newEnergy = newEnGrad.E_El + newEnGrad.E_VdW;
	float oldEnergy = oldEnGrad.E_El + oldEnGrad.E_VdW;

	/* Metropolis Criterion */
	if (newEnergy <= oldEnergy) {
		oldEnGrad = newEnGrad;
		oldDOF = newDOF;
//		cout << oldEnGrad.E_El + oldEnGrad.E_VdW << endl;
	} else {
#ifdef _OPENMP
		int tid = omp_get_thread_num();
		double r = RNGs[tid]();
#else
		double r = RNGs[0]();
#endif
		if (r < std::exp(-(newEnergy - oldEnergy)/kT)) {
			oldEnGrad = newEnGrad;
			oldDOF = newDOF;
//			cout << oldEnGrad.E_El + oldEnGrad.E_VdW << " by luck"<< endl;
		}
		else {
//			cout << oldEnGrad.E_El + oldEnGrad.E_VdW << " not accepted" << endl;
		}
	}

//	static int count = 0;
//	count++;
//	if (count % 2 == 0) {
//		cout << endl;
//	}
}

/* printing results to stderr */
void printResultsOutput(unsigned numDofs, as::DOF* dofs, as::EnGrad* enGrads, std::vector<asUtils::Vec3f>& pivots)
{
	using namespace std;

	int precisionSetting = cout.precision( );
	ios::fmtflags flagSettings = cout.flags();
	cout.setf(ios::showpoint);
	cout.precision(6);
	asUtils::Vec3f pivot_diff = pivots[0] - pivots[1];

	/* print header */
	cout << "#pivot 1 " << pivots[0][0] << " " << pivots[0][1] << " " << pivots[0][2] << " " << endl;
	cout << "#pivot 2 " << pivots[1][0] << " " << pivots[1][1] << " " << pivots[1][2] << " " << endl;
	cout << "#centered receptor: false" << endl;
	cout << "#centered ligands: false" << endl;
	for (unsigned i = 0; i < numDofs; ++i) {
		const as::EnGrad& enGrad = enGrads[i];
		const as::DOF& dof = dofs[i];
		cout << "#"<< i+1 << endl;
		cout << "## Energy: " << enGrad.E_VdW + enGrad.E_El << endl;
		cout << "## " << enGrad.E_VdW << " " << enGrad.E_El << endl;
		cout << dof.recId << " "
			 << 0.0 << " " << 0.0 << " " << 0.0 << " "
			 << 0.0 << " " << 0.0 << " " << 0.0 << endl;
		cout << dof.ligId << " "
			 <<	dof.ang.x << " " << dof.ang.y << " " << dof.ang.z << " "
			 << dof.pos.x + pivot_diff[0]<< " " << dof.pos.y  + pivot_diff[1] << " " << dof.pos.z + pivot_diff[2] << endl;
	}

	cout.precision(precisionSetting);
	cout.flags(flagSettings);
}


void copyRecIds2LigDof(std::vector<std::vector<as::DOF>>& DOF_molecules);
void applyIdMapping(std::vector<as::DOF>& dofs, int shift);
void applyInverseIdMapping(std::vector<as::DOF>& dofs, int shift);
void setupEnsembleWeigths(as::ServerManagement const& server, std::vector<int> recIds, std::vector<int> ligIds);

int main (int argc, char *argv[]) {
	using namespace std;

	/* initialize Logger */
	bool use_file = true;
	init_logger(use_file);
	unique_ptr<Log::Logger> log(mca::_log);


	/* required variables */
	string recListName;
	string ligListName;
	string gridName;
	string paramsName;
	string dofName;
	string recGridAlphabetName;

	/* optional variables */
	unsigned numCPUs;
	unsigned numIter;
	unsigned chunkSize;
	vector<int> devices;
	vector<double> samplingWeights;

	int numToConsider;
	int whichToTrack;
#ifdef _OPENMP
	unsigned numOMPThreads;
#endif

	/* catch command line exceptions */
	try {

		/* print argv */
		log->info() << "Client starts with command: ";
		std::vector<std::string> arguments(argv , argv + argc);
		for (auto string : arguments) { *log << string << " ";}
		*log << endl;

		/* description of the application */
		TCLAP::CmdLine cmd("An ATTRACT client that performs energy minimization by a Monte Carlo search.", ' ', "1.1");

		/* define required arguments */
		TCLAP::ValueArg<string> dofArg("","dof","",true,"Structure (DOF) file","*.dat", cmd);

		/* define optional arguments */
		TCLAP::ValueArg<string> recArg("r","receptor-list","list of pdb-files of receptor file names. (Default: partner1-ensemble.list)", false,"partner1-ensemble.list","*.list", cmd);
		TCLAP::ValueArg<string> ligArg("l","ligand-list","list of pdb-files of ligand file names. (Default: partner2-ensemble.list)", false,"partner2-ensemble.list","*.list", cmd);
		TCLAP::ValueArg<string> gridArg("g","grid","Receptor grid file. (Default: receptorgrid.grid)",false, "receptorgrid.grid","*.grid", cmd);
		TCLAP::ValueArg<string> paramArg("p","par","Attract parameter file. (Default: attract.par)",false,"attract.par","*.par", cmd);
		TCLAP::ValueArg<string> gridAlphabet("a","receptor-alphabet","Receptor grid alphabet file.",false,"","*.alphabet", cmd);

		/* define optional arguments */
		TCLAP::ValueArg<unsigned> cpusArg("c","cpus","Number of CPU threads to be used. (Default: 0)", false, 0, "uint");
		TCLAP::ValueArg<unsigned> numIterArg("","iter","Number Monte Carlo iterations. (Default: 50)", false, 50, "uint", cmd);
		TCLAP::ValueArg<double> maxDistArg("","maxDist","Maximum translational displacement (A). (Default: 1.0A)", false, 1.0, "int", cmd);
		TCLAP::ValueArg<double> maxAngArg("","maxAng","Maximum rotational displacement (deg). (Default: 3.0deg)", false, 3.0, "int", cmd);
		TCLAP::ValueArg<double> kTArg("","kT","Monte Carlo temperature. (Default: 10.0)", false, 10.0, "double", cmd);
		TCLAP::ValueArg<double> probEnsembleMoveArg("","ensProb","Probability of ensemble move. (Default: 1.0)", false, 1.0, "double", cmd);
		TCLAP::MultiArg<double> samplingWeightsArg("w","samplingWeight","Sampling weights for Monte Carlo move. (Default: 5 5 10)", false,"double", cmd);
		TCLAP::ValueArg<double> ubArg("","ub","Restraint distance. (Default: 100); ", false, 100.0, "double", cmd);
		TCLAP::ValueArg<double> kArg("","rstk","Force constant for restraints. (Default: 0.02); ", false, 0.02, "double", cmd);
		TCLAP::ValueArg<unsigned> seedArg("","seed","Random number generator seed. (Default: 1); ", false, 1, "uint", cmd);

		int numDevicesAvailable; cudaVerify(cudaGetDeviceCount(&numDevicesAvailable));
//		numDevicesAvailable = 0;
		vector<int> allowedValues(numDevicesAvailable); iota(allowedValues.begin(), allowedValues.end(), 0);
		TCLAP::ValuesConstraint<int> vc(allowedValues);
		TCLAP::MultiArg<int> deviceArg("d","device","Device ID of serverMode to be used.", false, &vc);
		TCLAP::ValueArg<unsigned> chunkSizeArg("","chunkSize", "Number of concurrently processed structures", false, 5000, "uint", cmd);

		TCLAP::ValueArg<int> num2ConsiderArg("","num", "Number of configurations to consider (1 - num). (Default: All)", false, -1, "int", cmd);
		TCLAP::ValueArg<int> which2TrackArg("","focusOn", "Condider only this configuration. (Default: -1)", false, -1, "int", cmd);

#ifdef _OPENMP
		TCLAP::ValueArg<unsigned> numOMPThreadsArg("","ompThreads", "Number of OpenMP threads. (Default: 1)", false, 1, "uint", cmd);
#endif

		cmd.xorAdd(cpusArg, deviceArg);

		// parse cmd-line input
		cmd.parse(argc, argv);

		/* Assigne parsed values */
		recListName 	= recArg.getValue();
		ligListName 	= ligArg.getValue();
		gridName 	= gridArg.getValue();
		paramsName 	= paramArg.getValue();
		dofName 	= dofArg.getValue();
		devices 	= deviceArg.getValue();
		numCPUs 	= cpusArg.getValue();
		chunkSize 	= chunkSizeArg.getValue();
		numToConsider = num2ConsiderArg.getValue();
		whichToTrack = which2TrackArg.getValue();
		recGridAlphabetName = gridAlphabet.getValue();


		numIter		= numIterArg.getValue();
		maxDist		= maxDistArg.getValue();
		maxAng		= maxAngArg.getValue();
		kT 			= kTArg.getValue();
		probApplyEnsembleMove = probEnsembleMoveArg.getValue();
		samplingWeights = samplingWeightsArg.getValue();
		ub = ubArg.getValue();
		k = kArg.getValue();
		seed = seedArg.getValue();
#ifdef _OPENMP
		numOMPThreads = numOMPThreadsArg.getValue();
#endif


	} catch (TCLAP::ArgException &e){
		cerr << "error: " << e.error() << " for arg " << e.argId() << endl;
	}

	log->info() << "recListName=" << recListName 		<< endl;
	log->info() << "ligListName=" << ligListName 		<< endl;
	log->info() << "gridName=" << gridName 		<< endl;
	log->info() << "parName=" << paramsName 	<< endl;
	log->info() << "dofName=" << dofName	 	<< endl;
	log->info() << "numCPUs=" << numCPUs 		<< endl;
	log->info() << "devices=[ "; for (auto device : devices) *log << device << " "; *log << "]"<<  endl;
	log->info() << "numIter=" << numIter 		<< endl;
	log->info() << "maxDist=" << maxDist		<< endl;
	log->info() << "maxAng="  << maxAng			<< endl;
	log->info() << "kT="  	  << kT				<< endl;
	log->info() << "ub="  	  << ub				<< endl;
	log->info() << "k="		  << k				<< endl;
	log->info() << "chunkSize=" << chunkSize 	<< endl;
	log->info() << "numToConsider=" << numToConsider << endl;
	log->info() << "whichToTrack=" << whichToTrack << endl;
#ifdef _OPENMP
	log->info() << "numOMPThreads=" << numOMPThreads << endl;
#endif


	/* convert degrees to rad */
	maxAng = maxAng * M_PI / 180.0;

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

	/* convert cumulative weights */
	{
		double sum = 0;
		for(int i = 0; i < 3; ++i) {
			sum += samplingWeights[i];
		}
		cumulativeSamplingWeights[0] = samplingWeights[0]/sum;
		for (int i = 1; i < 3; ++i) {
			cumulativeSamplingWeights[i] = cumulativeSamplingWeights[i-1] + samplingWeights[i]/sum;
		}
		log->info() << "samplingWeigths=[ ";
		for (int i=0; i<3;++i) *log << samplingWeights[i] << " "; *log << "]";
		*log << " --> ";
		for (int i=0; i<3;++i) *log << cumulativeSamplingWeights[i] << " "; *log << "]" << endl;
	}

	/* init omp related stuff */
#ifdef _OPENMP
	for(int i = 0; i<numOMPThreads; ++i) {
		dseeds.push_back(static_cast<double>(seed + i));
		RNGs.push_back(mca::default_RNG(seed + numOMPThreads + i));
	}
#else
	dseeds.push_back(static_cast<double>(seed));
	RNGs.push_back(mca::default_RNG(seed + 1));
#endif

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
	asDB::readEnsembleDOFFromFile(dofName, DOF_molecules);
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

	std::vector<string> ligEnsembleFileNames = asDB::readFileNamesFromEnsembleList(ligListName);
	int ligandSize = asDB::readProteinSizeFromPDB(ligEnsembleFileNames[0]);
	unsigned deviceBufferSize = ligandSize*chunkSize;
	unsigned numDofs = DOF_molecules[0].size();
	const unsigned numItems = 4*(((unsigned)ceil((double)numDofs/2.0) + chunkSize - 1) / chunkSize);

	log->info() << "ligandSize=" 		<< ligandSize 		<< endl;
	log->info() << "deviceBufferSize=" 	<< deviceBufferSize << endl;
	log->info() << "numDofs=" 			<< numDofs 			<< endl;
	log->info() << "numItems=" 			<< numItems 		<< endl;

	/* check if there are too two many items per request */
	const int maxItemsPerSubmit = 10000;
	if (numItems/4 > maxItemsPerSubmit) {
		log->error() << "Too many items per request. Increase chunkSize" << endl;
		exit(EXIT_FAILURE);
	}

	as::ServerManagement server(numItems, chunkSize, deviceBufferSize);

	/* load proteins and grid, and get a handle to it*/
	const int clientId = 0; // by specifing a client id, we may remove all added data by the by a call to removeClient(id)

	std::vector<int> recIds = server.addProteinEnsemble(clientId, recListName);
	std::vector<int> ligIds = server.addProteinEnsemble(clientId ,ligListName);
	int gridId = server.addGridUnion(clientId, gridName);
	server.addParamTable(paramsName);
	ensembleSizes[0] = recIds.size();
	ensembleSizes[1] = ligIds.size();

	/* parse or get pivots. only two pivots/molecules are allowed */
	if(autoPivot) {
		if (!pivots.empty()) {
			log->error() << "Auto pivot specified, but explicitly definined pivots available. (File "<< dofName << ")" << endl;
			exit(EXIT_FAILURE);
		}
		pivots.push_back(server.getProtein(recIds[0])->pivot());
		pivots.push_back(server.getProtein(ligIds[0])->pivot());
	} else {
		if (pivots.size() != 2) {
			log->error() << "No auto pivot specified, but number of definined pivots is incorrect. (File "<< dofName << ")" << endl;
			exit(EXIT_FAILURE);
		}
		for(auto recId: recIds) {
			server.getProtein(recId)->pivotize(pivots[0]);
		}
		for(auto ligId: ligIds) {
			server.getProtein(ligId)->pivotize(pivots[1]);
		}
	}

	/* apply receptor grid mapping for ligands */
	if (!recGridAlphabetName.empty()) {
		std::vector<unsigned> mapVec = asDB::readGridAlphabetFromFile(recGridAlphabetName);
		as::TypeMap typeMap = as::createTypeMapFromVector(mapVec);
		for(auto ligId: ligIds) {
			as::Protein* prot = server.getProtein(ligId);
			as::applyDefaultMapping(prot->numAtoms(), prot->type(), prot->type());
			as::applyMapping(typeMap, prot->numAtoms(), prot->type(), prot->mappedTypes());
		}
	} else {
		log->warning() << "No grid alphabet specified. Applying default mapping." << endl;
		for(auto ligId: ligIds) {
			as::Protein* prot = server.getProtein(ligId);
			as::applyDefaultMapping(prot->numAtoms(), prot->type(), prot->type());
			as::applyDefaultMapping(prot->numAtoms(), prot->type(), prot->mappedTypes());
		}
	}

	log->info() << "pivots= "; for (auto pivot : pivots) *log << pivot << ", "; *log << endl;

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


	/* transform ligand dofs assuming that the receptor is always centered in the origin */
	asClient::transformDOF_glob2rec(DOF_molecules[0], DOF_molecules[1], pivots[0], pivots[1], centered_receptor, centered_ligands);

	/* copy receptor ids to ligand dofs*/
	copyRecIds2LigDof(DOF_molecules);

	applyIdMapping(DOF_molecules[1], recIds.size());

	/* setup ensemble weights table */
	setupEnsembleWeigths(server, recIds, ligIds);


	/* initialize CPU workers if any*/
	for(unsigned i = 0; i < numCPUs; ++i) {
		server.addCPUWorker();
	}

	/* initialize devices and GPU workers if any */
	for (auto deviceId : devices) {
		server.addGPUWorker(deviceId);
		for (auto recId : recIds) {
			server.attachProteinToDevice(recId, deviceId);
		}
		for (auto ligId : ligIds) {
			server.attachProteinToDevice(ligId, deviceId);
		}
		server.attachGridUnionToDevice(gridId, deviceId);
		server.attachParamTableToDevice(deviceId);
	}

	/* Finalize server initialization */
	if (devices.size() > 0) {
		server.updateDeviceIDLookup();
	}

	/*
	 ** The server and the data is now initialized. We are ready to use it.
	 ** Next we need to initialize the client.
	 */

	/* Allocate result buffer and declare dof buffer */
	vector<as::EnGrad> enGradOld(numDofs);
	as::EnGrad* enGradBuffer = enGradOld.data();

	as::DOF* dofBuffer = DOF_molecules[1].data();


	unsigned idx[2] = {0, 1};
	int reqIds[2] = {-1, -1};
	int reqIdsFirstIter[2] = {-1, -1}; // only used in first iteration
	unsigned DOFSize[2] = { (unsigned)ceil((double)numDofs/2.0), (unsigned)floor((double)numDofs/2.0) };

	assert(DOFSize[0]+DOFSize[1] == numDofs);
	assert(DOFSize[0] != 0);
	assert(DOFSize[1] != 0);
//	cout << "size0 " << DOFSize[0] << " size1 " <<  DOFSize[1] << endl;

	/* Devide initial DOF Buffer in two parts */
	vector<as::DOF> dof(numDofs);
	as::DOF* newDOF = dof.data();
	as::DOF* newDOFs[2] = {newDOF , newDOF + DOFSize[0]};
	as::DOF* oldDOFs[2] = {dofBuffer, dofBuffer + DOFSize[0]};

	vector<as::EnGrad> enGradNew(numDofs);
	as::EnGrad* newEnGrad = enGradNew.data();
	as::EnGrad* newEnGrads[2] = {newEnGrad, newEnGrad + DOFSize[0]};
	as::EnGrad* oldEnGrads[2] = {enGradBuffer, enGradBuffer + DOFSize[0]};



	/******** Main Loop ********/

	/* First iteration is unique since we cannot process anything meanwhile.
	 * The loop is of size 2 since we have two half buffers. */
	nvtxRangePushA("Processing");
	for (int i = 0; i < 2; ++i) {
		/* Submit Request for first half of DOF Buffer to the GPU Server.
		 * This is a non-blocking call */
		reqIdsFirstIter[idx[0]] = asClient::server_submit(server, oldDOFs[idx[0]], DOFSize[idx[0]], gridId, serverMode);

		/* while energy is evaluated by the server, go ahead with calculating new configurations
		 * for first half of DOF Buffer */
#ifdef _OPENMP
		#pragma omp parallel for num_threads(numOMPThreads)
#endif
		for(unsigned j = 0; j < DOFSize[idx[0]]; ++j) {
			const as::DOF& oldDOF = oldDOFs[idx[0]][j];
			as::DOF& newDOF = newDOFs[idx[0]][j];
			randomStep(oldDOF, newDOF);
		}

		/* Submit Request for first half of new DOF Buffer to the GPU Server.
		 * This is a non-blocking call */
		reqIds[idx[0]] = asClient::server_submit(server, newDOFs[idx[0]], DOFSize[idx[0]], gridId, serverMode);

		/* Swap Buffers.
		 * The next use of idx[0] has the value of idx[1] and vice versa */
		std::swap(idx[0], idx[1]);
	}
	nvtxRangePop();

	/* Enter main loop */
	for(unsigned i = 0; i < (numIter-1)*2; ++i) {

		/* if we process a half buffer the first time, we need to wait for energies
		 * of the very first submission */
		nvtxRangePushA("Waiting");
		if (i == 0 || i == 1) {
			asClient::server_pull(server, reqIdsFirstIter[idx[0]], oldEnGrads[idx[0]]);
#ifdef _OPENMP
			#pragma omp parallel for num_threads(numOMPThreads)
#endif
			for(unsigned j = 0; j < DOFSize[idx[0]]; ++j) {
				as::DOF& oldDOF = oldDOFs[idx[0]][j];
				as::EnGrad& oldEnGrad = oldEnGrads[idx[0]][j];
				applyConstraints(oldDOF, oldEnGrad);
			}
		}

		/* pull (wait) for request that was submitted two iterations ago
		 * the buffers */
		asClient::server_pull(server, reqIds[idx[0]], newEnGrads[idx[0]]);
		nvtxRangePop();

		/* Accept new positions according to Metropolis criterion */
		nvtxRangePushA("Processing");
#ifdef _OPENMP
		#pragma omp parallel for num_threads(numOMPThreads)
#endif
		for(unsigned j = 0; j < DOFSize[idx[0]]; ++j) {
			as::DOF& oldDOF = oldDOFs[idx[0]][j];
			as::EnGrad& oldEnGrad = oldEnGrads[idx[0]][j];

			as::DOF& newDOF = newDOFs[idx[0]][j];
			as::EnGrad& newEnGrad = newEnGrads[idx[0]][j];

			applyConstraints(newDOF, newEnGrad);

			/* the accepted values are stored in old variables !!! */
			MC_accept(oldDOF, oldEnGrad, newDOF, newEnGrad);

			/* calulate new trial configuration */
			randomStep(oldDOF, newDOF);
		}


		/* Submit Request to the GPU Server.
		 * This is a non-blocking call */
		reqIds[idx[0]] = asClient::server_submit(server, newDOFs[idx[0]], DOFSize[idx[0]], gridId, serverMode);


		/* Swap Buffers */
		std::swap(idx[0], idx[1]);
		nvtxRangePop();

	}

	/* Finish last iteration */

	/* pull (wait) for request that was submitted two iterations ago
	 * the buffers */
	for (int i = 0; i < 2; ++i) {
		nvtxRangePushA("Waiting");
		if (numIter == 1) {
			asClient::server_pull(server, reqIdsFirstIter[idx[0]], oldEnGrads[idx[0]]);
#ifdef _OPENMP
			#pragma omp parallel for num_threads(numOMPThreads)
#endif
			for(unsigned j = 0; j < DOFSize[idx[0]]; ++j) {
				as::DOF& oldDOF = oldDOFs[idx[0]][j];
				as::EnGrad& oldEnGrad = oldEnGrads[idx[0]][j];
				applyConstraints(oldDOF, oldEnGrad);
			}
		}

		asClient::server_pull(server, reqIds[idx[0]], newEnGrads[idx[0]]);
		nvtxRangePop();

		/* Accept new positions according to Metropolis criterion */
		nvtxRangePushA("Processing");
#ifdef _OPENMP
		#pragma omp parallel for num_threads(numOMPThreads)
#endif
		for(unsigned j = 0; j < DOFSize[idx[0]]; ++j) {
			as::DOF& oldDOF = oldDOFs[idx[0]][j];
			as::EnGrad& oldEnGrad = oldEnGrads[idx[0]][j];

			as::DOF& newDOF = newDOFs[idx[0]][j];
			as::EnGrad& newEnGrad = newEnGrads[idx[0]][j];

			applyConstraints(newDOF, newEnGrad);

			/* the accepted values are stored in old variables !!! */
			MC_accept(oldDOF, oldEnGrad, newDOF, newEnGrad);

		}
		nvtxRangePop();
		std::swap(idx[0], idx[1]);
	}


	/******** End Main Loop ********/

	applyInverseIdMapping(DOF_molecules[1], recIds.size());

	/* print results to stderr */
	printResultsOutput(numDofs, dofBuffer, enGradBuffer, pivots);


	/* remove all data from host and devices */
	server.removeClient(clientId);


	log->info() << "exit" << endl;

	return 0;
}

void copyRecIds2LigDof(std::vector<std::vector<as::DOF>>& DOF_molecules) {
	for (unsigned i = 0; i < DOF_molecules[0].size(); ++i) {
		DOF_molecules[1][i].recId = DOF_molecules[0][i].ligId;
	}
}

void applyIdMapping(std::vector<as::DOF>& dofs, int shift) {
	for (unsigned i = 0; i < dofs.size(); ++i) {
		dofs[i].recId -= 1;
		dofs[i].ligId += shift - 1;
	}
}

void applyInverseIdMapping(std::vector<as::DOF>& dofs, int shift) {
	for (unsigned i = 0; i < dofs.size(); ++i) {
		dofs[i].recId += 1;
		dofs[i].ligId -= shift - 1;
	}
}

void setupEnsembleWeigths(as::ServerManagement const& server, std::vector<int> recIds, std::vector<int> ligIds) {
	std::vector<std::vector<as::Protein*>> ensProts(2);
	for(auto recId: recIds) {
		as::Protein* prot = server.getProtein(recId);
		ensProts[0].push_back(prot);
	}
	for(auto ligId: ligIds) {
		as::Protein* prot = server.getProtein(ligId);
		ensProts[1].push_back(prot);
	}

	EnsembleWeightTable::globTable.setEnsembleProteins(ensProts);
	EnsembleWeightTable::globTable.init();

}
