
#ifndef ROTMAT_H_
#define ROTMAT_H_

#include <asUtils/Vec3.h>

namespace asUtils {
template <typename T> class RotMat;

template<typename type>
std::ostream& operator<<(std::ostream& stream, const RotMat<type>& v) {
	stream << "[";
	unsigned j = 0;
	for(int i = 0; i < 3; ++i) {
		stream << "[" << v.mat[j+0] << ", " << v.mat[j+1] << ", " << v.mat[j+2] << "]";
		j += 3;
	}
	stream << "]";
	return stream;
}

template <typename T>
class RotMat {
public:
	RotMat() {};
	RotMat(T value) {
		std::fill_n(mat, 9, value);
	}

	T mat[9];

	RotMat getInv() const {
		RotMat m0;
		m0.mat[0] = mat[0];
		m0.mat[1] = mat[3];
		m0.mat[2] = mat[6];
		m0.mat[3] = mat[1];
		m0.mat[4] = mat[4];
		m0.mat[5] = mat[7];
		m0.mat[6] = mat[2];
		m0.mat[7] = mat[5];
		m0.mat[8] = mat[8];
		return m0;
	}

	RotMat operator* (const RotMat& rhs) {
		RotMat matOut(0.0);
		for(unsigned i = 0; i < 3; ++i) {
			for(unsigned j = 0; j < 3; ++j) {
				for(unsigned k = 0; k < 3; ++k) {
					matOut[i*3 + j] += mat[i*3 + k]*rhs[k*3 + j];
				}
			}
		}
		return matOut;
	}

	T& operator[](const unsigned& i) {
		return mat[i];
	}

	const T& operator[](const unsigned& i) const {
		return mat[i];
	}

	void operator*= (const RotMat& rhs) {
		RotMat matOut(0.0);
		for(unsigned i = 0; i < 3; ++i) {
			for(unsigned j = 0; j < 3; ++j) {
				for(unsigned k = 0; k < 3; ++k) {
					matOut.mat[i*3 + j] += mat[i*3 + k]*rhs.mat[k*3 + j];
				}
			}
		}
		*this = matOut;
	}

	Vec3<T> operator * (const Vec3<T>& rhs) {
		Vec3<T> vecOut;
		vecOut[0] = mat[0] * rhs[0] + mat[1] * rhs[1] + mat[2] * rhs[2];
		vecOut[1] = mat[3] * rhs[0] + mat[4] * rhs[1] + mat[5] * rhs[2];
		vecOut[2] = mat[6] * rhs[0] + mat[7] * rhs[1] + mat[8] * rhs[2];
		return vecOut;
	}

	friend std::ostream& operator<< <>(std::ostream& stream, const RotMat<T>& v);


};


typedef RotMat<float> RotMatf;
typedef RotMat<double> RotMatd;



}
#endif /* ROTMAT_H_ */
