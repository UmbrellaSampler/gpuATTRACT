#include <iostream>

#include "VA13Solver.h"

using std::cerr;
using std::endl;

ema::VA13Solver::Options ema::VA13Solver::settings;

extern "C" void minfor_(void* FortranSmuggler_ptr, int const& maxFunEval,
		double const* state);


void ema::VA13Solver::run(coro_t::caller_type& ca) {
	/* Create Smuggler */
	VA13Solver::FortranSmuggler smuggler(ca, state, objective);

	/* create and fill state array */
	double state_array[state.rows()];
	for (int i = 0; i < state.rows(); ++i) {
		state_array[i] = state(i);
	}

//	//Debug
//	cerr << "invoking minfor()" << endl;
//	cerr << "state=" << state.transpose() << endl;
//	cerr << "smuggler_ptr=" << &smuggler << endl;

	minfor_(&smuggler, settings.maxFunEval, state_array);
}


// Call back function for fortran to access the class.  This is a free function
// that is not a member or friend of MyClass, so it can only access public
// member variables and functions of MyClass.  BaseClass::operator() is such a
// public member function.
extern "C" void energy_for_fortran_to_call_(void* FortranSmuggler_ptr, double state_ptr[], double* energy, double grad[])
{
   // Cast to BaseClass.  If the pointer isn't a pointer to an object
   // derived from BaseClass, then the world will end.
	ema::VA13Solver::FortranSmuggler* smuggler = static_cast<ema::VA13Solver::FortranSmuggler*>(FortranSmuggler_ptr);

//	//Debug
//	cerr << "smuggler_ptr=" << smuggler << endl;

	/* set the state */
	ema::Vector& state = smuggler->state_ref();
	for (int i = 0; i < state.rows(); ++i) {
		state(i) = state_ptr[i];
	}
	/* call coroutine to break execution here until energy and gradients are available */
//	//Debug
//	cerr << "VA13: call coro" << endl;
	smuggler->call_coro();
//	cerr << "VA13: returned from coro" << endl;

	/* get the state */
	ema::ObjGrad& objGrad = smuggler->objective_ref();
	*energy = objGrad.obj;
	for (int i = 0; i < objGrad.grad.rows(); ++i) {
		grad[i] = objGrad.grad(i);
	}

}
