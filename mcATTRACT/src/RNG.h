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

#ifndef OMP_RNG_H_
#define OMP_RNG_H_

#include <random>

namespace mca {

template <typename Engine, typename Distribution>
class RNG
{
public:

    explicit RNG(unsigned seed = 1) :
		_engine(seed)
    {}

    typename Distribution::result_type operator()()
    {
        return _distribution(_engine);
    }

    Engine _engine;
    Distribution _distribution;
};

using default_RNG = RNG<std::default_random_engine, std::uniform_real_distribution<double>>;

} // namespace

#endif /* OMP_RNG_H_ */
