// This file is part of PoseEstimation.
// Copyright (c) 2021, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef THREE_POINT_HPP_
#define THREE_POINT_HPP_

#include <Eigen/Dense>

class ThreePointTest;

namespace ThreePoint
{
template <typename FloatType>
class ThreePoint
{
	friend ::ThreePointTest;

private:
	typedef Eigen::Matrix<FloatType, Eigen::Dynamic, 1> Matrix_Dx1;
	typedef Eigen::Matrix<FloatType, 3, 1> Matrix_3x1;
	typedef Eigen::Matrix<FloatType, 3, 3> Matrix_3x3;
	typedef Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> Matrix_3x3R;
	typedef Eigen::Matrix<FloatType, 3, Eigen::Dynamic> Matrix_3xD;
	typedef Eigen::Matrix<FloatType, 4, 4> Matrix_4x4;
	typedef Eigen::Matrix<FloatType, 4, 1> Matrix_4x1;

public:
	// q = [x, y, z, w] = w + x*i + y*j + z*k
	static inline void getRotation(const FloatType *q, FloatType *R)
	{
		const FloatType q0_2 = 2 * q[0], q1_2 = 2 * q[1], q2_2 = 2 * q[2];
		const FloatType q03_2 = q0_2 * q[3], q13_2 = q1_2 * q[3], q23_2 = q2_2 * q[3];
		const FloatType q00_2 = q0_2 * q[0], q01_2 = q0_2 * q[1], q02_2 = q0_2 * q[2];
		const FloatType q11_2 = q1_2 * q[1], q12_2 = q1_2 * q[2], q22_2 = q2_2 * q[2];
		R[0] = 1 - (q11_2 + q22_2);
		R[1] = q01_2 - q23_2;
		R[2] = q02_2 + q13_2;
		R[3] = q01_2 + q23_2;
		R[4] = 1 - (q00_2 + q22_2);
		R[5] = q12_2 - q03_2;
		R[6] = q02_2 - q13_2;
		R[7] = q12_2 + q03_2;
		R[8] = 1 - (q00_2 + q11_2);
	}

	static inline void transformPoints(int n, FloatType s, const FloatType *R, const FloatType *t, const FloatType *X1, FloatType *X2)
	{
		Eigen::Map<const Matrix_3x3R> mR(R);
		Eigen::Map<const Matrix_3x1> mt(t);
		Eigen::Map<const Matrix_3xD> mX1(X1, 3, n);
		Eigen::Map<Matrix_3xD> mX2(X2, 3, n);
		mX2 = (s * mR * mX1).colwise() + mt;
	}

	// X1 = [X11 Y11 Z11 X12 Y12 ... X1n Y1n Z1n]
	// X2 = [X21 Y21 Z21 X22 Y22 ... X2n Y2n Z2n]
	// X2 = s*R*X1 + t
	static void getRT(int n, const FloatType *X1, const FloatType *X2, FloatType *R, FloatType *t, FloatType &s)
	{
		Eigen::Map<const Matrix_3xD> mX1(X1, 3, n);
		Eigen::Map<const Matrix_3xD> mX2(X2, 3, n);
		// centroid
		Matrix_3x1 mc1 = mX1.rowwise().sum() / mX1.cols();
		Matrix_3x1 mc2 = mX2.rowwise().sum() / mX2.cols();
		// translate origin to centroid
		Matrix_3xD mX1c = mX1.colwise() - mc1;
		Matrix_3xD mX2c = mX2.colwise() - mc2;

		// estimate scale
		Matrix_Dx1 mX1n = mX1c.colwise().norm();
		Matrix_Dx1 mX2n = mX2c.colwise().norm();
		s = (mX1n.array() * mX2n.array()).sum() / (mX1n.array() * mX1n.array()).sum();
		mX2c *= s;

		// corrrelation matrix
		Matrix_3x3 mC = mX1c * mX2c.transpose();
		// system matrix
		Matrix_4x4 A;
		A(3, 3) = mC(0, 0) + mC(1, 1) + mC(2, 2);
		A(3, 0) = A(0, 3) = mC(1, 2) - mC(2, 1);
		A(3, 1) = A(1, 3) = mC(2, 0) - mC(0, 2);
		A(3, 2) = A(2, 3) = mC(0, 1) - mC(1, 0);
		A(0, 0) = mC(0, 0) - mC(1, 1) - mC(2, 2);
		A(0, 1) = A(1, 0) = mC(0, 1) + mC(1, 0);
		A(0, 2) = A(2, 0) = mC(2, 0) + mC(0, 2);
		A(1, 1) = -mC(0, 0) + mC(1, 1) - mC(2, 2);
		A(1, 2) = A(2, 1) = mC(1, 2) + mC(2, 1);
		A(2, 2) = -mC(0, 0) - mC(1, 1) + mC(2, 2);

		// estimate quaternion from eiven vector
		Eigen::SelfAdjointEigenSolver<Matrix_4x4> eig(A, Eigen::ComputeEigenvectors);
		getRotation(eig.eigenvectors().col(3).data(), R);
		Eigen::Map<Matrix_3x3R> mR(R);
		Eigen::Map<Matrix_3x1> mt(t);
		mt = -mR * mc1 * s + mc2;
	}
};

}

#endif // THREE_POINT_HPP_