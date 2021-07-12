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

// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
//   either expressed or implied, of the FreeBSD Project.

#ifndef FIVE_POINT_HPP_
#define FIVE_POINT_HPP_

#include <Eigen/Dense>

#include <vector>

class FivePointTest;

namespace FivePoint
{

template <typename FloatType>
class FivePoint
{
	friend ::FivePointTest;
private:
	typedef Eigen::Matrix<FloatType, 3, 3> Matrix_3x3;
	typedef Eigen::Matrix<FloatType, 5, 9> Matrix_5x9;
	typedef Eigen::Matrix<FloatType, 9, 4> Matrix_9x4;
	typedef Eigen::Matrix<FloatType, 9, 9> Matrix_9x9;
	typedef Eigen::Matrix<FloatType, 9, Eigen::Dynamic> Matrix_9xD;
	typedef Eigen::Matrix<FloatType, 10, 10> Matrix_10x10;
	typedef Eigen::Matrix<FloatType, 10, 20, Eigen::RowMajor> Matrix_10x20R;
	typedef Eigen::Matrix<std::complex<FloatType>, 4, 10> CMatrix_4x10;
	typedef Eigen::Matrix<std::complex<FloatType>, 9, 10> CMatrix_9x10;

public:
	FivePoint() :
		m_E1(m_E)
		, m_E2(m_E1 + 9)
		, m_E3(m_E2 + 9)
		, m_E4(m_E3 + 9)
		, m_trace(m_C)
		, m_determinant(m_trace + 20 * 9)
	{
	}

	virtual ~FivePoint()
	{
	}

	bool getEMatrix(const FloatType *pts1, const FloatType *pts2, int &nSolutions, std::vector<FloatType> &Es)
	{
		getGroebnerBasisMatrix(pts1, pts2);
		getActionMatrix();
		bool ret = getRoot();
		if (!ret)
		{
			return false;
		}

		nSolutions = m_mEvec.cols();
		Es.resize(9 * nSolutions);
		for (int i = 0; i < nSolutions; i++)
		{
			for (int j = 0; j < 9; j++)
			{
				Es[9 * i + j] = m_mEvec(j, i);
			}
		}

		return true;
	}

private:
	// 1st order polynomial multiplication
	// (ax + by + cz + d)*(ex + fy + gz + h)
	//   = (aex^2 + bfy^2 + .... + hd)
	// p1 = [x y z 1]
	// p2 = [x y z 1]
	// p3 = [x^2 xy xz y^2 yz z^2 x y z 1]
	static inline void polynomial_multiplication1(const FloatType *p1, const FloatType *p2, FloatType *p3)
	{
		p3[0] = (p1[0])*(p2[0]);					// x^2
		p3[1] = (p1[0])*(p2[1]) + (p1[1])*(p2[0]);	// xy
		p3[2] = (p1[0])*(p2[2]) + (p1[2])*(p2[0]);	// xz
		p3[3] = (p1[1])*(p2[1]);					// y^2
		p3[4] = (p1[1])*(p2[2]) + (p1[2])*(p2[1]);	// yz
		p3[5] = (p1[2])*(p2[2]);					// z^2
		p3[6] = (p1[0])*(p2[3]) + (p1[3])*(p2[0]);	// x
		p3[7] = (p1[1])*(p2[3]) + (p1[3])*(p2[1]);	// y
		p3[8] = (p1[2])*(p2[3]) + (p1[3])*(p2[2]);	// z
		p3[9] = (p1[3])*(p2[3]);					// 1
	}

	// 2nd order polynomial multiplication
	// p1 = [x y z 1]
	// p3 = [x^2 xy xz y^2 yz z^2 x y z 1]
	// p4 = [x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1]
	static inline void polynomial_multiplication2(const FloatType *p1, const FloatType *p3, FloatType *p4)
	{
		p4[0] = (p1[0])*(p3[0]);										// x^3
		p4[1] = (p1[1])*(p3[0]) + (p1[0])*(p3[1]);						// x^2y
		p4[2] = (p1[0])*(p3[3]) + (p1[1])*(p3[1]);						// xy^2
		p4[3] = (p1[1])*(p3[3]);										// y^3
		p4[4] = (p1[2])*(p3[0]) + (p1[0])*(p3[2]);						// x^2z
		p4[5] = (p1[0])*(p3[4]) + (p1[1])*(p3[2]) + (p1[2])*(p3[1]);	// xyz
		p4[6] = (p1[1])*(p3[4]) + (p1[2])*(p3[3]);						// y^2z
		p4[7] = (p1[0])*(p3[5]) + (p1[2])*(p3[2]);						// xz^2
		p4[8] = (p1[1])*(p3[5]) + (p1[2])*(p3[4]);						// yz^2
		p4[9] = (p1[2])*(p3[5]);										// z^3
		p4[10] = (p1[0])*(p3[6]) + (p1[3])*(p3[0]);						// x^2
		p4[11] = (p1[0])*(p3[7]) + (p1[1])*(p3[6]) + (p1[3])*(p3[1]);	// xy
		p4[12] = (p1[1])*(p3[7]) + (p1[3])*(p3[3]);						// y^2
		p4[13] = (p1[0])*(p3[8]) + (p1[2])*(p3[6]) + (p1[3])*(p3[2]);	// xz
		p4[14] = (p1[1])*(p3[8]) + (p1[2])*(p3[7]) + (p1[3])*(p3[4]);	// yz
		p4[15] = (p1[2])*(p3[8]) + (p1[3])*(p3[5]);						// z^2
		p4[16] = (p1[0])*(p3[9]) + (p1[3])*(p3[6]);						// x
		p4[17] = (p1[1])*(p3[9]) + (p1[3])*(p3[7]);						// y
		p4[18] = (p1[2])*(p3[9]) + (p1[3])*(p3[8]);						// z
		p4[19] = (p1[3])*(p3[9]);										// 1
	}

	// matrices Es are assumed row major format
	static inline void getDeterminantConstraints(const FloatType *E1, const FloatType *E2, const FloatType *E3, const FloatType *E4, FloatType *det)
	{
		FloatType p1[4], p2[4], p31[10], p32[10], p5[20];
		// O1(E12, E23) - O1(E13, E22)
		p1[0] = E1[1]; p1[1] = E2[1]; p1[2] = E3[1]; p1[3] = E4[1];
		p2[0] = E1[5]; p2[1] = E2[5]; p2[2] = E3[5]; p2[3] = E4[5];
		polynomial_multiplication1(p1, p2, p31);
		p1[0] = E1[2]; p1[1] = E2[2]; p1[2] = E3[2]; p1[3] = E4[2];
		p2[0] = E1[4]; p2[1] = E2[4]; p2[2] = E3[4]; p2[3] = E4[4];
		polynomial_multiplication1(p1, p2, p32);
		for (int k = 0; k < 10; k++)
		{
			p31[k] -= p32[k];
		}
		p1[0] = E1[6]; p1[1] = E2[6]; p1[2] = E3[6]; p1[3] = E4[6];
		// O2(O1(E12, E23) - O1(E13, E22), E31)
		polynomial_multiplication2(p1, p31, p5);
		for (int k = 0; k < 20; k++)
		{
			det[k] = p5[k];
		}
		// O1(E13, E21) - O1(E11, E23)
		p1[0] = E1[2]; p1[1] = E2[2]; p1[2] = E3[2]; p1[3] = E4[2];
		p2[0] = E1[3]; p2[1] = E2[3]; p2[2] = E3[3]; p2[3] = E4[3];
		polynomial_multiplication1(p1, p2, p31);
		p1[0] = E1[0]; p1[1] = E2[0]; p1[2] = E3[0]; p1[3] = E4[0];
		p2[0] = E1[5]; p2[1] = E2[5]; p2[2] = E3[5]; p2[3] = E4[5];
		polynomial_multiplication1(p1, p2, p32);
		for (int k = 0; k < 10; k++)
		{
			p31[k] -= p32[k];
		}
		p1[0] = E1[7]; p1[1] = E2[7]; p1[2] = E3[7]; p1[3] = E4[7];
		// O2(O1(E13, E21) - O1(E11, E23), E32)
		polynomial_multiplication2(p1, p31, p5);
		for (int k = 0; k < 20; k++)
		{
			det[k] += p5[k];
		}
		// O1(E11, E22) - O1(E12, E21)
		p1[0] = E1[0]; p1[1] = E2[0]; p1[2] = E3[0]; p1[3] = E4[0];
		p2[0] = E1[4]; p2[1] = E2[4]; p2[2] = E3[4]; p2[3] = E4[4];
		polynomial_multiplication1(p1, p2, p31);
		p1[0] = E1[1]; p1[1] = E2[1]; p1[2] = E3[1]; p1[3] = E4[1];
		p2[0] = E1[3]; p2[1] = E2[3]; p2[2] = E3[3]; p2[3] = E4[3];
		polynomial_multiplication1(p1, p2, p32);
		for (int k = 0; k < 10; k++)
		{
			p31[k] -= p32[k];
		}
		p1[0] = E1[8]; p1[1] = E2[8]; p1[2] = E3[8]; p1[3] = E4[8];
		// O2(O1(E11, E22) - O1(E12, E21), E33)
		polynomial_multiplication2(p1, p31, p5);
		for (int k = 0; k < 10; k++)
		{
			p31[k] -= p32[k];
		}
		for (int k = 0; k < 20; k++)
		{
			det[k] += p5[k];
		}
	}

	// matrices Es are assumed row major format
	static inline void getSingularValueConstraints(const FloatType *E1, const FloatType *E2, const FloatType *E3, const FloatType *E4, FloatType *trace)
	{
		FloatType p1[4], p2[4], p3[4], p31[10];
		FloatType EE11[10] = {0}, EE12[10] = {0}, EE13[10] = {0}, EE22[10] = {0}, EE23[10] = {0}, EE33[10] = {0};
		for(int k = 0; k < 3; k++)
		{
			// EE11
			p1[0] = E1[k]; p1[1] = E2[k]; p1[2] = E3[k]; p1[3] = E4[k]; // E(1, k)
			polynomial_multiplication1(p1, p1, p31);
			for (int l = 0; l < 10; l++)
			{
				EE11[l] += p31[l];
			}
			// EE12
			p2[0] = E1[3 + k]; p2[1] = E2[3 + k]; p2[2] = E3[3 + k]; p2[3] = E4[3 + k]; // E(2, k)
			polynomial_multiplication1(p1, p2, p31);
			for (int l = 0; l < 10; l++)
			{
				EE12[l] += p31[l];
			}
			// EE22
			polynomial_multiplication1(p2, p2, p31);
			for (int l = 0; l < 10; l++)
			{
				EE22[l] += p31[l];
			}
			// EE23
			p3[0] = E1[6 + k]; p3[1] = E2[6 + k]; p3[2] = E3[6 + k]; p3[3] = E4[6 + k]; // E(3, k)
			polynomial_multiplication1(p2, p3, p31);
			for (int l = 0; l < 10; l++)
			{
				EE23[l] += p31[l];
			}
			// EE13
			polynomial_multiplication1(p1, p3, p31);
			for (int l = 0; l < 10; l++)
			{
				EE13[l] += p31[l];
			}
			// EE33
			polynomial_multiplication1(p3, p3, p31);
			for (int l = 0; l < 10; l++)
			{
				EE33[l] += p31[l];
			}
		}

		FloatType L11[10] = {0}, L12[10] = {0}, L13[10] = {0}, L22[10] = {0}, L23[10] = {0}, L33[10] = {0};
		for (int l = 0; l < 10; l++)
		{
			const FloatType sum = (EE11[l] + EE22[l] + EE33[l])/2;
			L11[l] = EE11[l] - sum;
			L22[l] = EE22[l] - sum;
			L33[l] = EE33[l] - sum;
			L12[l] = EE12[l];
			L13[l] = EE13[l];
			L23[l] = EE23[l];
		}

		for(int k = 0; k < 9; k++)
		{
			for(int l = 0; l < 20; l++)
			{
				trace[k*20 + l] = 0;
			}
		}
		{
			FloatType p5[20], p6[20], p7[20];
			FloatType *LE11 = &(trace[0]),  *LE21 = &(trace[20]), *LE31 = &(trace[40]), *LE12 = &(trace[60]), *LE22 = &(trace[80]);
			FloatType *LE32 = &(trace[100]),  *LE13 = &(trace[120]), *LE23 = &(trace[140]), *LE33 = &(trace[160]);
			p1[0] = E1[0]; p1[1] = E2[0]; p1[2] = E3[0]; p1[3] = E4[0]; // E(1, 1)
			p2[0] = E1[3]; p2[1] = E2[3]; p2[2] = E3[3]; p2[3] = E4[3]; // E(2, 1)
			p3[0] = E1[6]; p3[1] = E2[6]; p3[2] = E3[6]; p3[3] = E4[6]; // E(3, 1)
			polynomial_multiplication2(p1, L11, p5); // O2(L11, E11)
			polynomial_multiplication2(p2, L12, p6); // O2(L12, E21)
			polynomial_multiplication2(p3, L13, p7); // O2(L13, E31)
			for (int l = 0; l < 20; l++)
			{
				LE11[l] += p5[l];
				LE11[l] += p6[l];
				LE11[l] += p7[l];
			}
			polynomial_multiplication2(p1, L12, p5); // O2(L21 == L12, E11)
			polynomial_multiplication2(p2, L22, p6); // O2(L22, E21)
			polynomial_multiplication2(p3, L23, p7); // O2(L23, E31)
			for (int l = 0; l < 20; l++)
			{
				LE21[l] += p5[l];
				LE21[l] += p6[l];
				LE21[l] += p7[l];
			}
			polynomial_multiplication2(p1, L13, p5); // O2(L31 == L13, E11)
			polynomial_multiplication2(p2, L23, p6); // O2(L32 == L23, E21)
			polynomial_multiplication2(p3, L33, p7); // O2(L33, E31)
			for (int l = 0; l < 20; l++)
			{
				LE31[l] += p5[l];
				LE31[l] += p6[l];
				LE31[l] += p7[l];
			}
			p1[0] = E1[1]; p1[1] = E2[1]; p1[2] = E3[1]; p1[3] = E4[1]; // E(1, 2)
			p2[0] = E1[4]; p2[1] = E2[4]; p2[2] = E3[4]; p2[3] = E4[4]; // E(2, 2)
			p3[0] = E1[7]; p3[1] = E2[7]; p3[2] = E3[7]; p3[3] = E4[7]; // E(3, 2)
			polynomial_multiplication2(p1, L11, p5); // O2(L11, E12)
			polynomial_multiplication2(p2, L12, p6); // O2(L12, E22)
			polynomial_multiplication2(p3, L13, p7); // O2(L13, E32)
			for (int l = 0; l < 20; l++)
			{
				LE12[l] += p5[l];
				LE12[l] += p6[l];
				LE12[l] += p7[l];
			}
			polynomial_multiplication2(p1, L12, p5); // O2(L21 == L12, E12)
			polynomial_multiplication2(p2, L22, p6); // O2(L22, E22)
			polynomial_multiplication2(p3, L23, p7); // O2(L23, E32)
			for (int l = 0; l < 20; l++)
			{
				LE22[l] += p5[l];
				LE22[l] += p6[l];
				LE22[l] += p7[l];
			}
			polynomial_multiplication2(p1, L13, p5); // O2(L31 == L13, E12)
			polynomial_multiplication2(p2, L23, p6); // O2(L32 == L23, E22)
			polynomial_multiplication2(p3, L33, p7); // O2(L33, E32)
			for (int l = 0; l < 20; l++)
			{
				LE32[l] += p5[l];
				LE32[l] += p6[l];
				LE32[l] += p7[l];
			}
			p1[0] = E1[2]; p1[1] = E2[2]; p1[2] = E3[2]; p1[3] = E4[2]; // E(1, 3)
			p2[0] = E1[5]; p2[1] = E2[5]; p2[2] = E3[5]; p2[3] = E4[5]; // E(2, 3)
			p3[0] = E1[8]; p3[1] = E2[8]; p3[2] = E3[8]; p3[3] = E4[8]; // E(3, 3)
			polynomial_multiplication2(p1, L11, p5); // O2(L11, E13)
			polynomial_multiplication2(p2, L12, p6); // O2(L12, E23)
			polynomial_multiplication2(p3, L13, p7); // O2(L13, E33)
			for (int l = 0; l < 20; l++)
			{
				LE13[l] += p5[l];
				LE13[l] += p6[l];
				LE13[l] += p7[l];
			}
			polynomial_multiplication2(p1, L12, p5); // O2(L21 == L12, E13)
			polynomial_multiplication2(p2, L22, p6); // O2(L22, E23)
			polynomial_multiplication2(p3, L23, p7); // O2(L23, E33)
			for (int l = 0; l < 20; l++)
			{
				LE23[l] += p5[l];
				LE23[l] += p6[l];
				LE23[l] += p7[l];
			}
			polynomial_multiplication2(p1, L13, p5); // O2(L31 == L13, E13)
			polynomial_multiplication2(p2, L23, p6); // O2(L32 == L23, E23)
			polynomial_multiplication2(p3, L33, p7); // O2(L33, E33)
			for (int l = 0; l < 20; l++)
			{
				LE33[l] += p5[l];
				LE33[l] += p6[l];
				LE33[l] += p7[l];
			}
		}
	}

	static inline void getLinearConstraints(const FloatType *pts1, const FloatType *pts2, FloatType *E1, FloatType *E2, FloatType *E3, FloatType *E4)
	{
		Matrix_5x9 mC;
		for (unsigned int i = 0; i < 5; i++){
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

		Eigen::JacobiSVD<Matrix_5x9> svd(mC, Eigen::ComputeFullV);
		Matrix_9x9 mV = svd.matrixV();
		for (int i = 0; i < 9; i++)
		{
			E1[i] = mV(i, 5);
			E2[i] = mV(i, 6);
			E3[i] = mV(i, 7);
			E4[i] = mV(i, 8);
		}
	}

private:
	void getGroebnerBasisMatrix(const FloatType *pts1, const FloatType *pts2)
	{
		getLinearConstraints(pts1, pts2, m_E1, m_E2, m_E3, m_E4);
		getSingularValueConstraints(m_E1, m_E2, m_E3, m_E4, m_trace);
		getDeterminantConstraints(m_E1, m_E2, m_E3, m_E4, m_determinant);

		Matrix_10x20R mM(m_C);
		Eigen::PartialPivLU<Matrix_10x10> lu(mM.block(0, 0, 10, 10));
		m_mB = lu.solve(mM.block(0, 10, 10, 10));
	}

	void getActionMatrix()
	{
		m_mAt.setZero();
		m_mAt.block(0, 0, 3, 10) = -m_mB.block(0, 0, 3, 10);
		m_mAt.block(3, 0, 2, 10) = -m_mB.block(4, 0, 2, 10);
		m_mAt.row(5) = -m_mB.row(7);
		m_mAt(6, 0) = m_mAt(7, 1) = m_mAt(8, 3) = m_mAt(9, 6) = 1;
	}

	bool getRoot()
	{
		Eigen::EigenSolver<Matrix_10x10> eig(m_mAt, true);
		CMatrix_4x10 mSols;
		mSols.row(0) = eig.eigenvectors().row(6).array() / eig.eigenvectors().row(9).array();
		mSols.row(1) = eig.eigenvectors().row(7).array() / eig.eigenvectors().row(9).array();
		mSols.row(2) = eig.eigenvectors().row(8).array() / eig.eigenvectors().row(9).array();
		mSols.row(3).setOnes();

		Eigen::Map<Matrix_9x4> mE(m_E);
		CMatrix_9x10 mEvec = mE*mSols;
		std::vector<int> idx(0);
		for(int j = 0; j < mEvec.cols(); j++)
		{
			if(mEvec.col(j).imag().array().all() == 0)
			{
				idx.push_back(j);
			}
		}

		if (idx.empty())
		{
			return false;
		}

		m_mEvec = Matrix_9xD(9, idx.size());
		for(unsigned int j = 0; j < idx.size(); j++){
			m_mEvec.col(j) = mEvec.col(idx[j]).real().normalized();
		}

		return true;
	}

	FloatType m_E[9*4]; // orthogonal basis of null space of linear constarints
	FloatType *m_E1, *m_E2, *m_E3, *m_E4; // pointer to E
	FloatType m_C[20*10]; // trace / determinant constraint matrix
	FloatType *m_trace, *m_determinant; // pointer to C
	Matrix_10x10 m_mB; // Groebner basis matrix
	Matrix_10x10 m_mAt; // Action matrix
	Matrix_9xD m_mEvec; // Essential matrix solutions
};

}

#endif // FIVE_POINT_HPP_