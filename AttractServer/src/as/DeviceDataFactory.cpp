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

#include "cuda_runtime.h"
#include <cstring>
#include <cassert>

#include "as/DeviceDataFactory.h"
#include "asUtils/macros.h"


/****************************
 * public member functions
 ****************************/
as::cudaGridUnionDesc as::DeviceDataFactory::initDeviceGridUnion(const GridUnion* gridUnion, int deviceId)
{
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));


	cudaIntrplGridDesc inner = DeviceDataFactory::initIntrpl(gridUnion->innerGrid());
	cudaIntrplGridDesc outer = DeviceDataFactory::initIntrpl(gridUnion->outerGrid());
	cudaNLGridDesc NL = DeviceDataFactory::initNL(gridUnion->NLgrid());

	deviceGridUnionDesc deviceDesc;
	deviceDesc.inner = inner.deviceDesc;
	deviceDesc.outer = outer.deviceDesc;
	deviceDesc.NL = NL.deviceDesc;

	hostGridUnionResource hostResc;
	hostResc.inner = inner.hostResc;
	hostResc.outer = outer.hostResc;
	hostResc.NL = NL.hostResc;

	cudaGridUnionDesc cudaDesc;
	cudaDesc.deviceDesc = deviceDesc;
	cudaDesc.hostResc = hostResc;
	return cudaDesc;
}

as::cudaProteinDesc as::DeviceDataFactory::initDeviceProtein(const Protein* protein, int deviceId)
{
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));

	deviceProteinDesc deviceDesc;
	unsigned numAtoms = protein->numAtoms();

	float *d_xPos;
	cudaVerify(cudaMalloc((void**) &d_xPos, numAtoms * sizeof(float)));
	cudaVerify(cudaMemcpy(d_xPos, protein->xPos(), numAtoms * sizeof(float), cudaMemcpyHostToDevice));
	float *d_yPos;
	cudaVerify(cudaMalloc((void**) &d_yPos, numAtoms * sizeof(float)));
	cudaVerify(cudaMemcpy(d_yPos, protein->yPos(), numAtoms * sizeof(float), cudaMemcpyHostToDevice));
	float *d_zPos;
	cudaVerify(cudaMalloc((void**) &d_zPos, numAtoms * sizeof(float)));
	cudaVerify(cudaMemcpy(d_zPos, protein->zPos(), numAtoms * sizeof(float), cudaMemcpyHostToDevice));

	unsigned* d_type;
	cudaVerify(cudaMalloc((void**) &d_type, numAtoms * sizeof(unsigned)));
	cudaVerify(cudaMemcpy(d_type, protein->type(), numAtoms * sizeof(unsigned), cudaMemcpyHostToDevice));
	unsigned* d_mappedType;
	cudaVerify(cudaMalloc((void**) &d_mappedType, numAtoms * sizeof(unsigned)));
	cudaVerify(cudaMemcpy(d_mappedType, protein->mappedType(), numAtoms * sizeof(unsigned), cudaMemcpyHostToDevice));
	float* d_charge;
	cudaVerify(cudaMalloc((void**) &d_charge, numAtoms * sizeof(float)));
	cudaVerify(cudaMemcpy(d_charge, protein->charge(), numAtoms * sizeof(float), cudaMemcpyHostToDevice));

	unsigned numModes = protein->numModes();
	float* d_xModes = NULL;
	float* d_yModes = NULL;
	float* d_zModes = NULL;
	if (numModes != 0) {
		cudaVerify(cudaMalloc((void**) d_xModes, numAtoms * numModes * sizeof(float)));
		cudaVerify(cudaMemcpy(d_xModes, protein->xModes(), numAtoms * numModes * sizeof(float), cudaMemcpyHostToDevice));
		cudaVerify(cudaMalloc((void**) d_yModes, numAtoms * numModes * sizeof(float)));
		cudaVerify(cudaMemcpy(d_yModes, protein->yModes(), numAtoms * numModes * sizeof(float), cudaMemcpyHostToDevice));
		cudaVerify(cudaMalloc((void**) d_zModes, numAtoms * numModes * sizeof(float)));
		cudaVerify(cudaMemcpy(d_zModes, protein->zModes(), numAtoms * numModes * sizeof(float), cudaMemcpyHostToDevice));
	}

	deviceDesc.numAtoms = numAtoms;
	deviceDesc.xPos = d_xPos;
	deviceDesc.yPos = d_yPos;
	deviceDesc.zPos = d_zPos;
	deviceDesc.type = d_type;
	deviceDesc.mappedType = d_mappedType;
	deviceDesc.charge = d_charge;
	deviceDesc.numModes = numModes;
	deviceDesc.xModes = d_xModes;
	deviceDesc.yModes = d_yModes;
	deviceDesc.zModes = d_zModes;

	cudaProteinDesc cudaDesc;
	cudaDesc.deviceDesc = deviceDesc;
	cudaDesc.hostResc = deviceDesc;

	return cudaDesc;
}

as::cudaParamTableDesc as::DeviceDataFactory::initDeviceParamTable (
		const AttrParamTable* table, int deviceId)
{
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));

	deviceParamTableDesc deviceDesc;

	const unsigned numTypes = table->numTypes();

	AttrParamTable::type* d_table;
	cudaVerify(cudaMalloc((void**)&d_table, numTypes*numTypes*sizeof(AttrParamTable::type)));
	cudaVerify(cudaMemcpy(d_table, table->table(), numTypes*numTypes*sizeof(AttrParamTable::type), cudaMemcpyHostToDevice));

	deviceDesc.numTypes = numTypes;
	deviceDesc.shape = table->potShape();
	deviceDesc.paramTable = d_table;
	cudaParamTableDesc cudaDesc;
	cudaDesc.deviceDesc = deviceDesc;
	cudaDesc.hostResc = deviceDesc;

	return cudaDesc;
}

void as::DeviceDataFactory::disposeDeviceGridUnion(hostGridUnionResource resc, int deviceId)
{
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));

	hostIntrplGridResource& inner = resc.inner;
	hostIntrplGridResource& outer = resc.outer;
	hostNLGridResource& NL = resc.NL;

	/* Free interpolation grid resources */
	for (uint i = 0; i<inner.numArrays; i++) {
		cudaVerify(cudaFreeArray(inner.cuArrayPtr[i]));
		cudaVerify(cudaDestroyTextureObject(inner.h_texArrayLin[i]));
		cudaVerify(cudaDestroyTextureObject(inner.h_texArrayPt[i]));
	}
	delete[] inner.cuArrayPtr;
	delete[] inner.h_texArrayLin;
	delete[] inner.h_texArrayPt;
	cudaVerify(cudaFree(inner.d_texArrayLin));
	cudaVerify(cudaFree(inner.d_texArrayPt));

	for (uint i = 0; i<outer.numArrays; i++) {
		cudaVerify(cudaFreeArray(outer.cuArrayPtr[i]));
		cudaVerify(cudaDestroyTextureObject(outer.h_texArrayLin[i]));
		cudaVerify(cudaDestroyTextureObject(outer.h_texArrayPt[i]));
	}
	delete[] outer.cuArrayPtr;
	delete[] outer.h_texArrayLin;
	delete[] outer.h_texArrayPt;
	cudaVerify(cudaFree(outer.d_texArrayLin));
	cudaVerify(cudaFree(outer.d_texArrayPt));

	/* Free NL grid resources */
	cudaVerify(cudaFreeArray(NL.cuArray));
	cudaVerify(cudaDestroyTextureObject(NL.tex));
}
void as::DeviceDataFactory::disposeDeviceProtein(hostProteinResource resc, int deviceId)
{
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));

	cudaVerify(cudaFree(resc.xPos));
	cudaVerify(cudaFree(resc.yPos));
	cudaVerify(cudaFree(resc.zPos));
	cudaVerify(cudaFree(resc.charge));
	cudaVerify(cudaFree(resc.type));
	cudaVerify(cudaFree(resc.mappedType));
	if (resc.numModes != 0) {
		cudaVerify(cudaFree(resc.xModes));
		cudaVerify(cudaFree(resc.yModes));
		cudaVerify(cudaFree(resc.zModes));
	}
}

void as::DeviceDataFactory::disposeDeviceParamTable(hostParamTableResource resc, int deviceId) {
	assert(deviceId >= 0);
	cudaVerify(cudaSetDevice(deviceId));

	cudaVerify(cudaFree(resc.paramTable));

}

/****************************
 * protected member functions
 ****************************/

/****************************
 * private member functions
 ****************************/
as::cudaIntrplGridDesc as::DeviceDataFactory::initIntrpl(const IntrplGrid* grid)
{
	/** array of CUDA texture objects for built-in interpolation */
	cudaTextureObject_t* h_texArrayLin; /** located on host */
	cudaTextureObject_t* d_texArrayLin; /** located on device */
	/** array of CUDA texture objects for manual interpolation */
	cudaTextureObject_t* h_texArrayPt; 	/** located on host */
	cudaTextureObject_t* d_texArrayPt; 	/** located on device */
	cudaArray** h_cuArrayPtr;

	/** The following pointers are deleted when the grid gets detached from the device.
	 * They are needed to destroy device recourses (textures) and are kept in the hostResc object
	 * below.*/
	h_cuArrayPtr = new cudaArray*[grid->numTypes()];
	h_texArrayLin = new cudaTextureObject_t[grid->numTypes()];
	h_texArrayPt = new cudaTextureObject_t[grid->numTypes()];

	// Allocate CUDA array in device memory
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	struct cudaExtent cuExtent = make_cudaExtent(grid->width(), grid->height(), grid->depth());
	// For cudaMalloc3DArray: width range in elements.
	// For cudaMalloc3D(...): width range in bytes.

	// Specify texture object parameters
	cudaTextureDesc texDescLin; // for built in device interpolation
	memset(&texDescLin, 0, sizeof(cudaTextureDesc));
	texDescLin.addressMode[0] = cudaAddressModeBorder; // return 0 if out of bounds
	texDescLin.addressMode[1] = cudaAddressModeBorder;
	texDescLin.addressMode[2] = cudaAddressModeBorder;
	texDescLin.filterMode = cudaFilterModeLinear;
	texDescLin.readMode = cudaReadModeElementType;
	texDescLin.normalizedCoords = false;

	cudaTextureDesc texDescPt; // for manual interpolation kernel
	memset(&texDescPt, 0, sizeof(cudaTextureDesc));
	texDescPt.addressMode[0] = cudaAddressModeBorder; // return 0 if out of bounds
	texDescPt.addressMode[1] = cudaAddressModeBorder;
	texDescPt.addressMode[2] = cudaAddressModeBorder;
	texDescPt.filterMode = cudaFilterModePoint;
	texDescPt.readMode = cudaReadModeElementType;
	texDescPt.normalizedCoords = false;

	for (unsigned i = 0; i<grid->numTypes(); i++) {
		cudaArray* &cuArray = h_cuArrayPtr[i];
		cudaVerify(cudaMalloc3DArray(&cuArray, &channelDesc, cuExtent, cudaChannelFormatKindFloat));

		// copy data to 3D array
		cudaMemcpy3DParms copyParams;
		memset(&copyParams, 0, sizeof(copyParams));
		void* gridPtr = (void*)grid->getHostGridPtr(i);
		copyParams.srcPtr   = make_cudaPitchedPtr(gridPtr, cuExtent.width*sizeof(float4), cuExtent.width, cuExtent.height);
		copyParams.dstArray = cuArray;
		copyParams.extent   = cuExtent;
		copyParams.kind     = cudaMemcpyHostToDevice;
		cudaVerify(cudaMemcpy3D(&copyParams));

		// Specify resource
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		// Create texture objects
		cudaTextureObject_t &texObjLin = h_texArrayLin[i];
		texObjLin = (long long)NULL;
		cudaVerify(cudaCreateTextureObject(&texObjLin, &resDesc, &texDescLin, NULL));

		cudaTextureObject_t &texObjPt = h_texArrayPt[i];
		texObjPt = (long long)NULL;
		cudaVerify(cudaCreateTextureObject(&texObjPt, &resDesc, &texDescPt, NULL));
	}

	cudaVerify(cudaMalloc((void**)&d_texArrayLin, grid->numTypes()*sizeof(cudaTextureObject_t)));
	cudaVerify(cudaMemcpy(d_texArrayLin, h_texArrayLin,grid->numTypes()*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	cudaVerify(cudaMalloc((void**)&d_texArrayPt, grid->numTypes()*sizeof(cudaTextureObject_t)));
	cudaVerify(cudaMemcpy(d_texArrayPt, h_texArrayPt,grid->numTypes()*sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	/* create deviceIntrplGridDesc */
	deviceIntrplGridDesc deviceDesc;
	deviceDesc.width = grid->width();
	deviceDesc.height = grid->height();
	deviceDesc.depth = grid->depth();
	deviceDesc.dVox = grid->dVox();
	deviceDesc.dVox_inv = grid->dVox_inv();
	deviceDesc.voxelVol_inv = grid->voxelVol_inv();
 	deviceDesc.minDim = grid->pos();
 	deviceDesc.maxDim = grid->maxDim();
 	deviceDesc.texArrayLin = d_texArrayLin;
 	deviceDesc.texArrayPt = d_texArrayPt;

 	/* create hostResc (for device resource deletion) */
 	hostIntrplGridResource hostResc;
 	hostResc.numArrays = grid->numTypes();
 	hostResc.cuArrayPtr = h_cuArrayPtr;
 	hostResc.h_texArrayLin = h_texArrayLin;
    hostResc.h_texArrayPt = h_texArrayPt;
    hostResc.d_texArrayLin = d_texArrayLin;
    hostResc.d_texArrayPt = d_texArrayPt;

    /* create cudaIntrplGridDesc */
    cudaIntrplGridDesc cudaDesc;
    cudaDesc.deviceDesc = deviceDesc;
    cudaDesc.hostResc = hostResc;

    return cudaDesc;
}

as::cudaNLGridDesc as::DeviceDataFactory::initNL(const NLGrid* grid) {

	cudaChannelFormatDesc channelDesc =  cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindUnsigned);

	struct cudaExtent cuExtent = make_cudaExtent(grid->width(), grid->height(), grid->depth());
	// For cudaMalloc3DArray: width range in elements.
	// For cudaMalloc3D(...): width range in bytes.

	// Specify texture object parameters
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder; // return 0 if out of bounds
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = false;

	cudaArray* cuArray;
	cudaVerify(cudaMalloc3DArray(&cuArray, &channelDesc, cuExtent));

	// copy data to 3D array
	cudaMemcpy3DParms copyParams;
	memset(&copyParams, 0, sizeof(copyParams));
	copyParams.srcPtr = make_cudaPitchedPtr((void*) grid->grid(), cuExtent.width*sizeof(uint2), cuExtent.width, cuExtent.height);
	copyParams.dstArray = cuArray;
	copyParams.extent   = cuExtent;
	copyParams.kind     = cudaMemcpyHostToDevice;
	cudaVerify(cudaMemcpy3D(&copyParams));

	// Specify resource
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cuArray;

	// Create texture object
	cudaTextureObject_t texObj;
	texObj = (long long)NULL;
	cudaVerify(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

	// Create device neighbor list
	unsigned* d_neighborList;
	cudaVerify(cudaMalloc((void**)&d_neighborList, grid->neighborListSize()*sizeof(unsigned)));
	cudaVerify(cudaMemcpy(d_neighborList, grid->neighborList(),grid->neighborListSize()*sizeof(unsigned), cudaMemcpyHostToDevice));

	/* create deviceNLGridDesc */
	deviceNLGridDesc deviceDesc;
	deviceDesc.width 	= grid->width();
	deviceDesc.height = grid->height();
	deviceDesc.depth	= grid->depth();
	deviceDesc.dVox_inv  = grid->dVox_inv();
	deviceDesc.dPlateau2 = grid->dPlateau2();
	deviceDesc.minDim	= grid->minDim();
	deviceDesc.maxDim	= grid->maxDim();
	deviceDesc.tex = texObj;
	deviceDesc.neighborList = d_neighborList;

	hostNLGridResource hostResc;
	hostResc.tex = texObj;
	hostResc.cuArray = cuArray;

	cudaNLGridDesc cudaDesc;
	cudaDesc.deviceDesc = deviceDesc;
    cudaDesc.hostResc = hostResc;

	return cudaDesc;

}




