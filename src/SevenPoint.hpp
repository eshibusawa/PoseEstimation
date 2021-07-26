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

#ifndef SEVEN_POINT_HPP_
#define SEVEN_POINT_HPP_

#include <Eigen/Dense>

#include <vector>

class SevenPointTest;

namespace SevenPoint
{

template <typename FloatType>
class SevenPoint
{
	friend ::SevenPointTest;
private:
	typedef Eigen::Matrix<std::complex<FloatType>, 3, 1> CMatrix_3x1;
	typedef Eigen::Matrix<FloatType, 3, 3> Matrix_3x3;
	typedef Eigen::Matrix<FloatType, 7, 9> Matrix_7x9;
	typedef Eigen::Matrix<FloatType, 9, 9> Matrix_9x9;
	typedef Eigen::Matrix<FloatType, 9, Eigen::Dynamic> Matrix_9xD;

public:
	SevenPoint() :
		m_F1(m_F)
		, m_F2(m_F1 + 9)
	{
	}

	virtual ~SevenPoint()
	{
	}

	bool getMatrix(const FloatType *pts1, const FloatType *pts2, int &nSolutions, std::vector<FloatType> &Fs)
	{
		getConstraints(pts1, pts2);
		bool ret = getRoot();
		if (!ret)
		{
			return false;
		}

		nSolutions = m_mFvec.cols();
		Fs.resize(9 * nSolutions);
		for (int i = 0; i < nSolutions; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				Fs[9 * i + j] = m_mFvec(j, i);
			}
		}

		return true;
	}

private:
	// 1st order polynomial multiplication
	// (ax + b) * (cx + d) = (acx^2 + (ad + bc)x + bd)
	// p1 = [x 1]
	// p2 = [x 1]
	// p3 = [x^2 x 1]
	static inline void polynomial_multiplication1(const FloatType *p1, const FloatType *p2, FloatType *p3)
	{
		p3[0] = (p1[0])*(p2[0]);					// x^2
		p3[1] = (p1[0])*(p2[1]) + (p1[1])*(p2[0]);	// x
		p3[2] = (p1[1])*(p2[1]);					// 1
	}

	// 2nd order polynomial multiplication
	// (ax + b) * (cx^2 + dx + e) = (acx^3 + (ad + bc)x^2 + ... + be)
	// p1 = [x 1]
	// p3 = [x^2 x 1]
	// p3 = [x^3 x^2 x 1]
	static inline void polynomial_multiplication2(const FloatType *p1, const FloatType *p2, FloatType *p3)
	{
		p3[0] = (p1[0])*(p2[0]);					// x^3
		p3[1] = (p1[0])*(p2[1]) + (p1[1])*(p2[0]);	// x^2
		p3[2] = (p1[0])*(p2[2]) + (p1[1])*(p2[1]);	// x
		p3[3] = (p1[1])*(p2[2]);					// 1
	}

	static inline void getLinearConstraints(const FloatType *pts1, const FloatType *pts2, FloatType *F1, FloatType *F2)
	{
		Matrix_7x9 mC;
		for (unsigned int i = 0; i < 7; i++){
			const FloatType *p1 = &(pts1[2*i]), *p2 = &(pts2[2*i]);
			mC(i, 0) = (p2[0])*(p1[0]);
			mC(i, 1) = (p2[0])*(p1[1]);
			mC(i, 2) = (p2[0]);
			mC(i, 3) = (p2[1])*(p1[0]);
			mC(i, 4) = (p2[1])*(p1[1]);
			mC(i, 5) = (p2[1]);
			mC(i, 6) = (p1[0]);
			mC(i, 7) = (p1[1]);
			mC(i, 8) = 1;
		}

		Eigen::JacobiSVD<Matrix_7x9> svd(mC, Eigen::ComputeFullV);
		Matrix_9x9 mV = svd.matrixV();
		for (int i = 0; i < 9; i++)
		{
			F1[i] = mV(i, 7);
			F2[i] = mV(i, 8);
		}
	}
	static inline void getDeterminantConstraints(const FloatType *F1, const FloatType *F2, FloatType *C)
	{
		C[0] = C[1] = C[2] = C[3] = 0;
		FloatType p00[] = {F1[0], F2[0]}; // (0, 0)
		FloatType p01[] = {F1[1], F2[1]}; // (0, 1)
		FloatType p02[] = {F1[2], F2[2]}; // (0, 2)
		FloatType p10[] = {F1[3], F2[3]}; // (1, 0)
		FloatType p11[] = {F1[4], F2[4]}; // (1, 1)
		FloatType p12[] = {F1[5], F2[5]}; // (1, 2)
		FloatType p20[] = {F1[6], F2[6]}; // (2, 0)
		FloatType p21[] = {F1[7], F2[7]}; // (2, 1)
		FloatType p22[] = {F1[8], F2[8]}; // (2, 2)

		FloatType c1[3], c2[3], c3[4];
		polynomial_multiplication1(p11, p22, c1);
		polynomial_multiplication1(p12, p21, c2);
		c1[0] -= c2[0]; c1[1] -= c2[1]; c1[2] -= c2[2];
		polynomial_multiplication2(p00, c1, c3);
		C[0] += c3[0]; C[1] += c3[1]; C[2] += c3[2]; C[3] += c3[3];

		polynomial_multiplication1(p01, p22, c1);
		polynomial_multiplication1(p02, p21, c2);
		c1[0] -= c2[0]; c1[1] -= c2[1]; c1[2] -= c2[2];
		polynomial_multiplication2(p10, c1, c3);
		C[0] -= c3[0]; C[1] -= c3[1]; C[2] -= c3[2]; C[3] -= c3[3];

		polynomial_multiplication1(p01, p12, c1);
		polynomial_multiplication1(p02, p11, c2);
		c1[0] -= c2[0]; c1[1] -= c2[1]; c1[2] -= c2[2];
		polynomial_multiplication2(p20, c1, c3);
		C[0] += c3[0]; C[1] += c3[1]; C[2] += c3[2]; C[3] += c3[3];
	}

protected:
	static const size_t m_minSet; // minimum data number for parameter esimation
	static const bool m_acceptArbitraryNSet; // if true, this model accepts arbitrary data number for parameter esimation

private:
	void getConstraints(const FloatType *pts1, const FloatType *pts2)
	{
		getLinearConstraints(pts1, pts2, m_F1, m_F2);
		getDeterminantConstraints(m_F1, m_F2, m_C);
	}

	bool getRoot()
	{
		// construct the companion matrix
		Matrix_3x3 mC;
		mC.setZero();
		mC(1, 0) = mC(2, 1) = 1;
		mC(0, 2) = -m_C[3] / m_C[0];
		mC(1, 2) = -m_C[2] / m_C[0];
		mC(2, 2) = -m_C[1] / m_C[0];
		Eigen::EigenSolver<Matrix_3x3> eig(mC, false);
		CMatrix_3x1 mSols = eig.eigenvalues();

		std::vector<int> idx(0);
		for(int j = 0; j < 3; j++)
		{
			if(mSols[j].imag() == 0)
			{
				idx.push_back(j);
			}
		}

		if (idx.empty())
		{
			return false; // 3rd oder polynomial has at least one real root!
		}

		m_mFvec = Matrix_9xD(9, idx.size());
		for (size_t i = 0; i < idx.size(); i++)
		{
			for (int j = 0; j < 9; j++)
			{
				m_mFvec(j, i) = (mSols[i].real()) * (m_F1[j]) + (m_F2[j]);
			}
		}
		return true;
	}

	FloatType m_F[9*2]; // orthogonal basis of null space of linear constarints
	FloatType m_C[4]; 	// deteminant constraints
	FloatType *m_F1, *m_F2; // pointer to F
	Matrix_9xD m_mFvec; // Essential matrix solutions
};
template <typename FloatType>
const size_t SevenPoint<FloatType>::m_minSet = 7;
template <typename FloatType>
const bool SevenPoint<FloatType>::m_acceptArbitraryNSet = false;

}

#endif // SEVEN_POINT_HPP_