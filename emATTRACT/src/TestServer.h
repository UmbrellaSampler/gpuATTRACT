/*
 * TestServer.h
 *
 *  Created on: Oct 6, 2015
 *      Author: uwe
 */

#ifndef TESTSERVER_H_
#define TESTSERVER_H_

#include "meta.h"

namespace ema {

class TestServer {

public:
	int submit(extDOF* states, unsigned num) {
		std::vector<extEnGrad> objectives(num);
//		static int count = 0;
		for (unsigned i = 0; i < num; ++i) {
			float x,y,z,a,b,c;
			x = states[i].pos.x;
			y = states[i].pos.y;
			z = states[i].pos.z;
			a = states[i].ang.x;
			b = states[i].ang.y;
			c = states[i].ang.z;

//			objectives[i].E_VdW =0.5f*( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].E_VdW =0.25f*( a*a*a*a + b*b*b*b + c*c*c*c + x*x*x*x + y*y*y*y + z*z*z*z) - 1.0f;
			objectives[i].E_VdW =0.5* sin( a*a + b*b + c*c + x*x + y*y + z*z);
			objectives[i].E_VdW =0.5* sin( a*a + b*b + c*c + x*x + y*y + z*z) + a*a + b*b + c*c + x*x + y*y + z*z ;
			objectives[i].E_El = 0.0f;

//			objectives[i].pos.x = +x;
//			objectives[i].pos.y = +y;
//			objectives[i].pos.z = +z;
//			objectives[i].ang.x = +a;
//			objectives[i].ang.y = +b;
//			objectives[i].ang.z = +c;

//			objectives[i].pos.x = +x*x*x;
//			objectives[i].pos.y = +y*y*y;
//			objectives[i].pos.z = +z*z*z;
//			objectives[i].ang.x = +a*a*a;
//			objectives[i].ang.y = +b*b*b;
//			objectives[i].ang.z = +c*c*c;

//			objectives[i].pos.x = x*cos( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].pos.y = y*cos( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].pos.z = z*cos( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].ang.x = a*cos( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].ang.y = b*cos( a*a + b*b + c*c + x*x + y*y + z*z);
//			objectives[i].ang.z = c*cos( a*a + b*b + c*c + x*x + y*y + z*z);

			objectives[i].pos.x = x*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*x;
			objectives[i].pos.y = y*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*y;
			objectives[i].pos.z = z*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*z;
			objectives[i].ang.x = a*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*a;
			objectives[i].ang.y = b*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*b;
			objectives[i].ang.z = c*cos( a*a + b*b + c*c + x*x + y*y + z*z) + 2*c;


//			if (count < 1000) {
//				std::cout << "dof=" << states[i] << std::endl;
//				std::cout << "engrad=" << objectives[i] << std::endl;
////				std::cout << "ENERGY=" << objectives[i].E_VdW << std::endl;
//			}

		}
//		count++;

		assert(results.find(id) == results.end());
		results[id] = std::move(objectives);
		return id++;

	}

	std::vector<extEnGrad> getResult(int reqId) {
		assert(results.find(reqId) != results.end());
		auto tmp = std::move(results[reqId]);
		results.erase(reqId);
		return tmp;
	}

	std::map<int,std::vector<extEnGrad>> results;
	int id = 0;
};


inline int server_submit(TestServer& mngt, extDOF* dofs, const unsigned& numDofs,
			const int& gridId, const int& recId, const int& ligId,
			const as::Request::useMode_t& mode)
{
	/* Submit Request */
	int reqId = mngt.submit(dofs, numDofs);
	return reqId;
}

inline unsigned server_pull(TestServer& mngt, const int& RequestId, extEnGrad* buffer)
{

	/* Pull for Request */
	unsigned count = 0;

	auto EnGradVec = mngt.getResult(RequestId);
	auto beg = EnGradVec.begin();
	auto end = EnGradVec.end();
	std::copy(beg, end, buffer);

	return count;
}

}


#endif /* TESTSERVER_H_ */
