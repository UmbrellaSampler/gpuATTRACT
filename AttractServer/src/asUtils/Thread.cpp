/*
 * Thread.cpp
 *
 *  Created on: Jan 20, 2015
 *      Author: uwe
 */

#include "asUtils/Thread.h"

/* Constructor */
asUtils::Thread::Thread() {}

/* Destructor */
asUtils::Thread::~Thread() {}


/****************************
 * public member functions
 ****************************/
std::thread::id asUtils::Thread::id() {
	return _thread.get_id();
}
void asUtils::Thread::start() {
	_thread = std::thread(&Thread::run, this);
}
void asUtils::Thread::join() {
	_thread.join();
}
void asUtils::Thread::detach() {
	_thread.detach();
}
void asUtils::Thread::joinable() {
	_thread.joinable();
}

/****************************
 * protected member functions
 ****************************/

/****************************
 * private member functions
 ****************************/


