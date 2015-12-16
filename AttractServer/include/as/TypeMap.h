/*
 * TypeMap.h
 *
 *  Created on: Dec 15, 2015
 *      Author: uwe
 */

#ifndef TYPEMAP_H_
#define TYPEMAP_H_

#include <map>
#include <vector>

namespace as {

class TypeMap {
public:
	using valueType = unsigned;
	using keyType = unsigned;


	keyType getValue(valueType key) const {
		return _map.at(key);
	}

	void setKeyValuePair(keyType key, valueType value) {
		_map[key] = value;
	}

private:

	std::map<valueType, keyType> _map;
};

inline TypeMap createTypeMapFromVector(std::vector<TypeMap::keyType> vec) {
	TypeMap map;

	for (unsigned i = 0; i < vec.size(); ++i) {
		map.setKeyValuePair(vec[i], (vec[i] == 99 ? 0: i+1)); /* valid type begins at 1 */
	}
	map.setKeyValuePair(0, 0);
	return map;
}

}  // namespace as



#endif /* TYPEMAP_H_ */
