/*
 * DeviceDataInterface.h
 *
 *  Created on: Jan 18, 2015
 *      Author: uwe
 */

#ifndef DEVICEDATAINTERFACE_H_
#define DEVICEDATAINTERFACE_H_

#include "as/asTypes.h"
#include "as/SimParam.h"

namespace as {
	void setDeviceGridUnion(const deviceGridUnionDesc &desc, unsigned deviceId, unsigned localDeviceID);
	void unsetDeviceGridUnion(unsigned deviceId, unsigned localDeviceID);
	void setDeviceProtein(const deviceProteinDesc &desc, unsigned deviceId, unsigned localDeviceID);
	void unsetDeviceProtein(unsigned deviceId, unsigned localDeviceID);
	void setDeviceParamTable(const deviceParamTableDesc& desc, unsigned deviceId);
	void unsetDeviceParamTable(unsigned deviceId);
	void setDeviceSimParam(const SimParam& desc, unsigned deviceId);
	void unsetDeviceSimParam(unsigned deviceId);
}


#endif /* DEVICEDATAINTERFACE_H_ */
