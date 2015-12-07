
#include "as/Worker.h"
#include "as/ServerManagement.h"

/* Constructor */


/* Destructor */



/****************************
 * public member functions
 ****************************/

/****************************
 * protected member functions
 ****************************/
as::Protein* as::Worker::getProtein(const unsigned& globId) {
	return _S_mngt.DataMngt()->getProtein(globId);
}

as::GridUnion* as::Worker::getGridUnion(const unsigned& globId) {
	return _S_mngt.DataMngt()->getGridUnion(globId);
}

as::AttrParamTable* as::Worker::getParamTable() {
	return _S_mngt.DataMngt()->getParamTable();
}

as::SimParam* as::Worker::getSimParam() {
	return _S_mngt.DataMngt()->getSimParam();
}

/****************************
 * private member functions
 ****************************/



