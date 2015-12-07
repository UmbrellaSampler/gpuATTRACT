#include "as/DeviceDataInterface.h"
#include "asUtils/macros.h"
#include "config.h"



__constant__ as::deviceGridUnionDesc c_Grids[DEVICE_MAXGRIDS];
__constant__ as::deviceProteinDesc c_Proteins[DEVICE_MAXPROTEINS];
__constant__ as::deviceParamTableDesc c_ParamTable;
__constant__ as::deviceSimParam c_SimParam;

void as::setDeviceGridUnion(const deviceGridUnionDesc &desc,
		unsigned deviceId, unsigned localDeviceID)
{
	cudaVerify(cudaSetDevice(deviceId));
	cudaVerify(cudaMemcpyToSymbol(c_Grids, &desc,
			sizeof(deviceGridUnionDesc),
			localDeviceID*sizeof(deviceGridUnionDesc), cudaMemcpyHostToDevice));
}

void as::unsetDeviceGridUnion(
		unsigned deviceId, unsigned localDeviceID)
{
	cudaVerify(cudaSetDevice(deviceId));
	deviceGridUnionDesc desc;
	memset(&desc, 0, sizeof(deviceGridUnionDesc));
	cudaVerify(cudaMemcpyToSymbol(c_Grids, &desc,
			sizeof(deviceGridUnionDesc),
			localDeviceID*sizeof(deviceGridUnionDesc), cudaMemcpyHostToDevice));
}

void as::setDeviceProtein(const deviceProteinDesc &desc,
		unsigned deviceId, unsigned localDeviceID)
{
	cudaVerify(cudaSetDevice(deviceId));
	cudaVerify(cudaMemcpyToSymbol(c_Proteins, &desc,
			sizeof(deviceProteinDesc),
			localDeviceID*sizeof(deviceProteinDesc), cudaMemcpyHostToDevice));
}

void as::unsetDeviceProtein(
		unsigned deviceId, unsigned localDeviceID)
{
	cudaVerify(cudaSetDevice(deviceId));
	deviceProteinDesc desc;
	memset(&desc, 0, sizeof(deviceProteinDesc));
	cudaVerify(cudaMemcpyToSymbol(c_Proteins, &desc,
			sizeof(deviceProteinDesc),
			localDeviceID*sizeof(deviceProteinDesc), cudaMemcpyHostToDevice));
}

void as::setDeviceParamTable(const deviceParamTableDesc& desc,
		unsigned deviceId)
{
	cudaVerify(cudaSetDevice(deviceId));
	cudaVerify(cudaMemcpyToSymbol(c_ParamTable, &desc,
			sizeof(deviceParamTableDesc), 0, cudaMemcpyHostToDevice));
}

void as::unsetDeviceParamTable(unsigned deviceId)
{
	cudaVerify(cudaSetDevice(deviceId));
	deviceParamTableDesc desc;
	memset(&desc, 0, sizeof(deviceParamTableDesc));
	cudaVerify(cudaMemcpyToSymbol(c_ParamTable, &desc,
			sizeof(deviceParamTableDesc), 0, cudaMemcpyHostToDevice));
}

void as::setDeviceSimParam(const SimParam& simPar,
		unsigned deviceId)
{
	cudaVerify(cudaSetDevice(deviceId));
	deviceSimParam desc;
	desc.dielec 	= simPar.dielec;
	desc.epsilon	= simPar.epsilon;
	desc.ffelec		= simPar.ffelec;
	desc.useRecGrad	= simPar.useRecGrad;
	desc.usePot		= simPar.usePot;
	cudaVerify(cudaMemcpyToSymbol(c_SimParam, &desc,
			sizeof(c_SimParam), 0, cudaMemcpyHostToDevice));
}

void as::unsetDeviceSimParam(unsigned deviceId)
{
	cudaVerify(cudaSetDevice(deviceId));
	deviceSimParam desc;
	memset(&desc, 0, sizeof(deviceSimParam));
	cudaVerify(cudaMemcpyToSymbol(c_SimParam, &desc,
			sizeof(c_SimParam), 0, cudaMemcpyHostToDevice));
}

