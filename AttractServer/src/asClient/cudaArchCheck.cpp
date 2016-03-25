/*******************************************************************************
 * gpuATTRACT framework
 * Copyright (C) 2016 Uwe Ehmann
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
#include "asUtils/macros.h"
#include "asClient/cudaArchCheck.h"


namespace asClient {

bool checkComputeCapability(int minMajorCC, int minMinorCC, const std::vector<int>& devices) {
	int nDevices = devices.size();

	bool proper = true;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

		int majorCC = prop.major;
		int minorCC = prop.minor;

		int CC = 10*majorCC + minorCC;
		int minCC = 10*minMajorCC + minMinorCC;

		proper &=  minCC <= CC;
	}

	return proper;
}

}


