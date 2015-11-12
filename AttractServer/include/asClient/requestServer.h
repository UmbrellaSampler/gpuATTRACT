/*
 * requestServer.h
 *
 *  Created on: Oct 5, 2015
 *      Author: uwe
 */

#ifndef REQUESTSERVER_H_
#define REQUESTSERVER_H_

#include <thread>
#include <chrono>

#include "as/ServerManagement.h"
#include "as/asTypes.h"


namespace asClient {

/*
 ** @brief: returns -1 on failure
 */

inline int server_submit(as::ServerManagement& mngt, as::DOF* dofs, const unsigned& numDofs,
			const int& gridId, const int& recId, const int& ligId,
			const as::Request::useMode_t& mode)
{
	using namespace std;

	/* Submit Request */
	unsigned count = 0;
	int reqId = -1;
	while (reqId == -1 && count < 10000) {
		reqId = mngt.submitRequest(dofs, numDofs, gridId, recId, ligId, mode);
		if (reqId != -1) {
			return reqId;
		}
		++count;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	return reqId;
}

inline unsigned server_pull(as::ServerManagement& mngt, const int& RequestId, as::EnGrad* buffer)
{
	using namespace std;

	/* Pull for Request */
	unsigned count = 0;
	while ((mngt.pullRequest(RequestId, buffer) != as::Dispatcher::ready && count < 10000)) {
		++count;
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	return count;
}

}  // namespace asClient


#endif /* REQUESTSERVER_H_ */
