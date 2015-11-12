/*
 * Thread.h
 *
 *  Created on: Jan 20, 2015
 *      Author: uwe
 */

#ifndef THREAD_H_
#define THREAD_H_

#include <thread>

namespace asUtils {

class Thread {
public:
	/* Constructor */
	Thread();

	/* Destructor */
	virtual ~Thread();
	/***************
	* G E T T E R
	***************/

	/***************
	* S E T T E R
	***************/

	/****************************
	 * public member functions
	 ****************************/
	std::thread::id id();
	void start();
	void join();
	void detach();
	void joinable();

	virtual void run() = 0;


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

	/****************************
	 * private member variables
	 ****************************/
	std::thread _thread;
};

} // namespace


#endif /* THREAD_H_ */
