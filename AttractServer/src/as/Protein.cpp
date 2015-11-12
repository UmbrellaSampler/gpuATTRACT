
#include <iostream>
#include <iomanip>

#include "as/Protein.h"

/* Constructor */
as::Protein::Protein() :
	_tag(),
	_numAtoms(0), _pivot(0,0,0),
	_pos(nullptr),
	_type(nullptr), _charge(nullptr),
	_numModes(0), _modes(nullptr) {};

as::Protein::Protein(ProteinDesc desc) :
	_tag(desc.id),
	_numAtoms(desc.numAtoms),
	_pivot(0,0,0),
	_type(desc.type),
	_charge(desc.charge),
	_numModes(desc.numModes)

{
	if (desc.pos != NULL) {
		_pos = desc.pos;
	} else {
		std::cerr << "Error: Protein description: invalid position pointer"
				<< std::endl;
		exit(EXIT_FAILURE);
	}

	if (_numModes > 0 ) {
		if (desc.modes == nullptr) {
			std::cerr << "Error: Protein description: invalid mode pointer"
					<< std::endl;
			exit(EXIT_FAILURE);
		}
		_modes = desc.modes;
	} else { // _xModes is going to be explicitly deleted!
		_modes = nullptr;
	}

	for (unsigned i = 0; i < _numAtoms; ++i) {
		_type[i] = _type[i] == 0 ? 31 : _type[i]-1;
	}

}

/* Destructor */
as::Protein::~Protein() {
	if (_pos     != nullptr) delete[] _pos;
	if (_charge  != nullptr) delete[] _charge;
	if (_type    != nullptr) delete[] _type;
	if (_modes   != nullptr) delete[] _modes;
}


/****************************
 * public member functions
 ****************************/

float* as::Protein::getOrCreatePosPtr() {
	if (_pos == nullptr) {
		if (_numAtoms == 0) {
			std::cerr << "Error: getOrCreatePosPtr(): the number of atoms must be set before." << std::endl;
			exit(EXIT_FAILURE);
		}
		_pos = new float[3*_numAtoms];
	}
	return _pos;
}

unsigned* as::Protein::getOrCreateTypePtr() {
	if (_type == nullptr) {
		if (_numAtoms == 0) {
			std::cerr << "Error: getOrCreateTypePtr(): the number of atoms must be set before" << std::endl;
			exit(EXIT_FAILURE);
		}
		_type = new unsigned[_numAtoms];
	}
	return _type;
}

float* as::Protein::getOrCreateChargePtr() {
	if (_charge == nullptr) {
		if (_numAtoms == 0) {
			std::cerr << "Error: getOrCreateChargePtr(): the number of atoms must be set before" << std::endl;
			exit(EXIT_FAILURE);
		}
		_charge = new float[_numAtoms];
	}
	return _charge;
}

float* as::Protein::getOrCreateModePtr() {
	if (_modes == nullptr) {
		if (_numAtoms == 0) {
			std::cerr << "Error: getOrCreateModePtr(): the number of atoms must be set before" << std::endl;
			exit(EXIT_FAILURE);
		}

		if (_numModes == 0) {
			std::cerr << "Error: getOrCreateModePtr(): the number of modes must be set before" << std::endl;
			exit(EXIT_FAILURE);
		}
		_modes = new float[3*_numAtoms*_numModes];
	}
	return _modes;
}

void as::Protein::pivotize(asUtils::Vec3f pivot) {
	/* undo preceding pivotizing if necessary */
	if (!(_pivot == asUtils::Vec3f(0,0,0))) {
		for (unsigned i = 0; i < _numAtoms; ++i) {
			xPos()[i] += _pivot[0];
			yPos()[i] += _pivot[1];
			zPos()[i] += _pivot[2];
		}
	}
	_pivot = pivot;
	if (!(_pivot == asUtils::Vec3f(0,0,0))) {
		for (unsigned i = 0; i < _numAtoms; ++i) {
			xPos()[i] -= _pivot[0];
			yPos()[i] -= _pivot[1];
			zPos()[i] -= _pivot[2];
		}
	}
}

void as::Protein::auto_pivotize() {
	/* undo preceding pivotizing if necessary */
	if (!(_pivot == asUtils::Vec3f(0,0,0))) {
		for (unsigned i = 0; i < _numAtoms; ++i) {
			xPos()[i] += _pivot[0];
			yPos()[i] += _pivot[1];
			zPos()[i] += _pivot[2];
		}
		_pivot = asUtils::Vec3f(0,0,0);
	}
	/* calculate pivot by center of mass coordinates */
	asUtils::Vec3f pivot(0,0,0);
	for (unsigned i = 0; i < _numAtoms; ++i) {
		pivot[0] += xPos()[i];
		pivot[1] += yPos()[i];
		pivot[2] += zPos()[i];
	}
	pivot /= static_cast<double>(_numAtoms);
	pivotize(pivot);
}

void as::Protein::print() {
	using namespace std;
	int precisionSetting = cout.precision( );
	ios::fmtflags flagSettings = cout.flags();
	cout.setf(ios::dec | ios::showpoint | ios::showpos);
	cout.precision(6);

	int w = 13;
//	outStream 	<< setw(w) << "DOF"
//				<< setw(w) << dof.pos.x << setw(w) << dof.pos.y << setw(w) << dof.pos.z
//				<< setw(w) << dof.ang.x << setw(w) << dof.ang.y << setw(w) << dof.ang.z;

	cout << setw(5) << "#" << setw(w) << "X" << setw(w) << "Y" << setw(w) << "Z" << setw(6) << "TYPE" << setw(w) << "CHARGE" << endl;
	for(unsigned i = 0; i < _numAtoms; ++i) {
		cout << setw(5) << i+1
			 << setw(w) << xPos()[i]
		     << setw(w) << yPos()[i]
		     << setw(w) << zPos()[i]
		     << setw(6) << type()[i]
			 << setw(w) << charge()[i]
			 << endl;
	}

	cout.precision(precisionSetting);
	cout.flags(flagSettings);
}






