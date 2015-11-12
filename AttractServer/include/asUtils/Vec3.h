/*
 * Vector3.h
 *
 *  Created on: Nov 21, 2014
 *      Author: tchipevn
 */

#ifndef VEC3_H_
#define VEC3_H_

#include <cmath>
#include <iostream>

namespace asUtils {

template<typename type> class Vec3;

/** Global operators */

template<typename type>
std::ostream& operator<<(std::ostream& stream, const Vec3<type>& v) {

	stream << "[" << v._content[0] << "; " << v._content[1] << "; " << v._content[2] << "]";
	return stream;
}

template<typename type>
Vec3<type> operator*(double scalar, const Vec3<type>& v) {
	return v * scalar;
}

template<typename type>
class Vec3 {
public:
	Vec3() {
	}

	Vec3(type arg) {
		_content[0] = arg;
		_content[1] = arg;
		_content[2] = arg;
	}

	Vec3(type arg0, type arg1, type arg2) {
		_content[0] = arg0;
		_content[1] = arg1;
		_content[2] = arg2;
	}

	Vec3(type args[3]) {
		_content[0] = args[0];
		_content[1] = args[1];
		_content[2] = args[2];
	}

	Vec3(const Vec3& other) {
		_content[0] = other._content[0];
		_content[1] = other._content[1];
		_content[2] = other._content[2];
	}

	Vec3 operator+(const Vec3& rhs) const {
		type result[3];
		result[0] = _content[0] + rhs._content[0];
		result[1] = _content[1] + rhs._content[1];
		result[2] = _content[2] + rhs._content[2];
		return Vec3(result);
	}

	void operator+=(const Vec3& rhs) {
		_content[0] += rhs._content[0];
		_content[1] += rhs._content[1];
		_content[2] += rhs._content[2];
	}

	Vec3 operator-(const Vec3& rhs) const {
		type result[3];
		result[0] = _content[0] - rhs._content[0];
		result[1] = _content[1] - rhs._content[1];
		result[2] = _content[2] - rhs._content[2];
		return Vec3(result);
	}

	void operator-=(const Vec3& rhs) {
		_content[0] -= rhs._content[0];
		_content[1] -= rhs._content[1];
		_content[2] -= rhs._content[2];
	}

	Vec3 operator*(double scalar) const {
		type result[3];
		result[0] = _content[0] * scalar;
		result[1] = _content[1] * scalar;
		result[2] = _content[2] * scalar;
		return Vec3(result);
	}

	void operator*=(double scalar) {
		_content[0] *= scalar;
		_content[1] *= scalar;
		_content[2] *= scalar;
	}

	Vec3 operator/(double scalar) const {
		type result[3];
		result[0] = _content[0] / scalar;
		result[1] = _content[1] / scalar;
		result[2] = _content[2] / scalar;
		return Vec3(result);
	}

	void operator/=(double scalar) {
		_content[0] /= scalar;
		_content[1] /= scalar;
		_content[2] /= scalar;
	}

	type MaxNorm() const {
		type norm;
		norm = std::max(abs(_content[0]), abs(_content[1]));
		norm = std::max(norm, abs(_content[2]));
		return norm;
	}

	type L2NormSquare() const {
		return _content[0] * _content[0] + _content[1] * _content[1] + _content[2] * _content[2];
	}

	double L2Norm() const {
		return sqrt(L2NormSquare());
	}

	Vec3& operator=(const Vec3& rhs) {
		if (this != &rhs) {
			_content[0] = rhs._content[0];
			_content[1] = rhs._content[1];
			_content[2] = rhs._content[2];
		}
		return *this;
	}

	Vec3& operator=(type rhs) {
		_content[0] = rhs;
		_content[1] = rhs;
		_content[2] = rhs;
		return *this;
	}

	type& operator[](int i) {
		return _content[i];
	}

	type operator[](int i) const {
		return _content[i];
	}

	bool operator==(const Vec3& rhs) const {
		if (_content[0] != rhs._content[0])
			return false;
		if (_content[1] != rhs._content[1])
			return false;
		if (_content[2] != rhs._content[2])
			return false;
		return true;
	}

	~Vec3() {
	}

	friend std::ostream& operator<< <>(std::ostream& stream, const Vec3<type>& v);
//	friend Vector3 operator*<type>(double scalar, const Vector3& v);

	/* used in std::map, returns true if lhs < rhs */
	class compare {
	public:
		bool operator()(const Vec3& lhs, const Vec3& rhs) const {
			return lhs[0] < rhs[0]
                || ( lhs[0] == rhs[0] && ( lhs[1] < rhs[1]
                || ( lhs[1] == rhs[1] && lhs[2] < rhs[2])));
		}
	};

private:
	type _content[3];
};

typedef Vec3<float> Vec3f;
typedef Vec3<double> Vec3d;

} /* namespace asUtils */

#endif /* VEC3_H_ */
