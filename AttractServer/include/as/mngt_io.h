
#ifndef MNGT_IO_H_
#define MNGT_IO_H_

#include <string>

#include "as/io_config.h"
#include "as/DataManagement.h"

namespace as {

class mngt_io {
public:
	/* Constructor */
	// private

	/* Destructor */

	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	static std::string heading();
	static std::string clientInfo(DataManagement* mngt);
	static std::string dataInfo(const DataManagement* mngt);
	static std::string deviceInfo(const DataManagement* mngt);


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
	mngt_io() {}
	/****************************
	 * private member functions
	 ****************************/

	/****************************
	 * private member variables
	 ****************************/
	static const char fill  = '#';
	static const char vsepa = '|';
	static char const hsepa = '_';
	static const unsigned colWidth = COLWIDTH;
};

}
#endif /* MNGT_IO_H_ */
