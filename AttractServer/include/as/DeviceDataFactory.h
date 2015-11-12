/*
 * DeviceDataFactory.h
 *
 *  Created on: Jan 13, 2015
 *      Author: uwe
 */

#ifndef DEVICEDATAFACTORY_H_
#define DEVICEDATAFACTORY_H_

#include <string>
#include "as/IntrplGrid.h"
#include "as/NLGrid.h"
#include "as/GridUnion.h"
#include "as/Protein.h"
#include "as/ParamTable.h"
#include "as/asTypes.h"

namespace as {

/*
 ** @brief: Performs necessary steps to initialize a gridUnion,
 ** a Protein or a ParamTable object for use on the GPU.
 **
 */
class DeviceDataFactory {
public:
	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	static cudaGridUnionDesc initDeviceGridUnion(const GridUnion* gridUnion, int deviceId);
	static cudaProteinDesc initDeviceProtein(const Protein* protein, int deviceId);
	static cudaParamTableDesc initDeviceParamTable (const AttrParamTable* table, int deviceId);

	static void disposeDeviceGridUnion(hostGridUnionResource desc, int deviceId);
	static void disposeDeviceProtein(hostProteinResource desc, int deviceId);
	static void disposeDeviceParamTable(hostParamTableResource desc, int deviceId);
	

	/****************************
	 * public member variables
	 ****************************/

protected:
	/****************************
	 * protected member functions
	 ****************************/

	/****************************
	 * protected member variables
	 ****************************/

private:
	/****************************
	 * private member functions
	 ****************************/

	/* Constructor */
	DeviceDataFactory() {}; // object instantiation is private!

	static cudaIntrplGridDesc initIntrpl(const IntrplGrid* grid);
	static cudaNLGridDesc initNL(const NLGrid* grid);

	/****************************
	 * private member variables
	 ****************************/

};

} // namespace


#endif /* DEVICEDATAFACTORY_H__ */
