/*
 * params.h
 *
 *  Created on: Mar 24, 2015
 *      Author: uwe
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include <string>
#include <cstring>
#include <sstream>

namespace asClient {

// parameter processing
template<typename T>
bool getParam(std::string param, T &var, int argc, char **argv)
{
    const char *c_param = param.c_str();
    for(int i=argc-1; i>=1; i--)
    {
        if (argv[i][0]!='-') continue;
//        cout << argv[i]+1 << endl;
        if (strcmp(argv[i]+1, c_param)==0)
        {
            if (!(i+1<argc))
            	return true;
            std::stringstream ss;
            ss << argv[i+1];
            ss >> var;
            return true;
        }

    }
    return false;
}

}


#endif /* PARAMS_H_ */
