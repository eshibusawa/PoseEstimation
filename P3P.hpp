// This file is part of PoseEstimation.
// This file is a modified version of p3p.m <http://rpg.ifi.uzh.ch/software_datasets.html>,
// see 3-Clause BSD license below.
// Copyright (c) 2021, Eijiro Shibusawa <phd_kimberlite@yahoo.co.jp>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
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

// Copyright (c) 2011, Laurent Kneip, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of ETH Zurich nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL ETH ZURICH BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef P3P_HPP_
#define P3P_HPP_

#include <Eigen/Dense>

class P3PTest;

namespace P3P
{
template <typename FloatType>
class P3P {
	friend ::P3PTest;

private:
	typedef Eigen::Matrix<std::complex<FloatType>, 4, 1> CMatrix_4x1;
	typedef Eigen::Matrix<FloatType, 3, 1> Matrix_3x1;
	typedef Eigen::Matrix<FloatType, 3, 3> Matrix_3x3;
	typedef Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> Matrix_3x3R;
	typedef Eigen::Matrix<FloatType, 4, 1> Matrix_4x1;
	typedef Eigen::Matrix<FloatType, 4, 4> Matrix_4x4;
	const static FloatType m_eps;

public:
	// x = R'*(X - t) =>  x = R*X + t
	static void convertPose(FloatType *R, FloatType *t)
	{
		Eigen::Map<Matrix_3x3R> mR1(R);
		Eigen::Map<Matrix_3x1> mt1(t);
		Matrix_3x3 mR2 = mR1.transpose();
		Matrix_3x1 mt2 = -mR2*mt1;
		mR1 = mR2;
		mt1 = mt2;
	}

	// x = R'*(X - t)
	bool getMatrix(const FloatType *p2dn, const FloatType *p3d, int &nSols, std::vector<FloatType> &Ps)
	{
		// check degenerate case
		if (checkColinearity(p3d))
		{
			nSols = 0;
			Ps.resize(0);
			return false;
		}

		// select P1, f1 and compute T matrix, N matrix
		selectP1(p2dn, p3d);
		transformCoordinate();

		// compute constraints
		getConstraints();

		// compute real roots for the constraints
		getRealRoot();
		if (m_realRoots.empty())
		{
			nSols = 0;
			Ps.resize(0);
			return true;
		}

		// compute poses corresponding to the real root
		getPoses(nSols, Ps);

		return true;
	}

private:
	inline void getConstraints()
	{
		FloatType pcotn[2] = {m_p[1], m_p[0] * m_phi[0] / m_phi[1] - m_d12 * m_b}; // numerator of Eq. 9
		FloatType pcotd[2] = {m_p[1] * m_phi[0] / m_phi[1], -m_p[0] +  m_d12}; // denominator of Eq. 9
		FloatType pcotn2[3] = {}, pcotd2[3] = {}, pcotnd[3] = {}; // nume^2, denom^2, nume*denom
		polynomialMultiplication1(pcotn, pcotn, pcotn2);
		polynomialMultiplication1(pcotd, pcotd, pcotd2);
		polynomialMultiplication1(pcotn, pcotd, pcotnd);
		FloatType pcotd2n2[3] = {}; // denom^2 + nume^2
		pcotd2n2[0] = pcotd2[0] + pcotn2[0];
		pcotd2n2[1] = pcotd2[1] + pcotn2[1];
		pcotd2n2[2] = pcotd2[2] + pcotn2[2];

		FloatType pcos1[] = {-m_phi[1] * m_phi[1] * m_p[1] * m_p[1], 0, 0}; // term 1 of l.h.s of Eq.10, I think f_2^2 in original equation is phi_2^2
		pcos1[2] = -pcos1[0];
		polynomialMultiplication2(pcos1, pcotd2n2, m_as);

		FloatType p12 = m_p[0] * m_p[0];
		FloatType p122 = 2 * m_p[0] * m_p[1];
		FloatType p22 = m_p[1] * m_p[1];
		m_as[2] -= (p12 * pcotd2[0]); // term 1 of r.h.s of Eq.10
		m_as[3] -= (p12 * pcotd2[1]);
		m_as[4] -= (p12 * pcotd2[2]);
		m_as[1] += (p122 * pcotnd[0]); // term 2 of r.h.s of Eq.10
		m_as[2] += (p122 * pcotnd[1]);
		m_as[3] += (p122 * pcotnd[2]);
		m_as[0] -= (p22 * pcotn2[0]); // term 3 of r.h.s of Eq.10
		m_as[1] -= (p22 * pcotn2[1]);
		m_as[2] -= (p22 * pcotn2[2]);
	}

	void getPoses(int &nSols, std::vector<FloatType> &Ps)
	{
		nSols = 0;
		Ps.resize((3 + 9) * m_realRoots.size());
		for (size_t k = 0; k < m_realRoots.size(); k++)
		{
			nSols++;
			const FloatType cos_theta = m_realRoots[k];
			const FloatType sin_theta = std::sqrt(1 - (cos_theta * cos_theta));
			const FloatType cot_alpha =
				(m_phi[0] * m_p[0] / m_phi[1] + cos_theta * m_p[1] - m_d12 * m_b) /
				(m_phi[0] * m_p[1] * cos_theta / m_phi[1] - m_p[0] + m_d12); // Eq. 9
			const FloatType sin_alpha = std::sqrt(1 / (cot_alpha * cot_alpha + 1));
			FloatType cos_alpha = std::sqrt(1 - (sin_alpha * sin_alpha));
			if (cot_alpha < 0)
			{
				cos_alpha = -cos_alpha;
			}

			FloatType *pC = &(Ps[(3 + 9) * k]);
			Eigen::Map<Matrix_3x1> mC(pC); // Eq. 5
			mC[0] = cos_alpha;
			mC[1] = sin_alpha * cos_theta;
			mC[2] = sin_alpha * sin_theta;

			Eigen::Map<Matrix_3x3R> mR(pC + 3); // Eq. 6
			mR(0,0) = -mC[0];
			mR(0,1) = -mC[1];
			mR(0,2) = -mC[2];
			mR(1, 0) = sin_alpha;
			mR(1, 1) = -cos_alpha * cos_theta;
			mR(1, 2) = -cos_alpha * sin_theta;
			mR(2, 0) = 0;
			mR(2, 1) = -sin_theta;
			mR(2, 2) = cos_theta;

			mC *= (m_d12 * (sin_alpha * m_b + cos_alpha));
			mC = m_mP1 + m_mN.transpose() * mC; // Eq. 12
			mR = m_mN.transpose() * mR.transpose() * m_mT; // Eq. 13
		}
	}

	inline void getRealRoot()
	{
		m_realRoots.resize(0);
		Matrix_4x4 mCom; // companion matrix
		mCom.setZero();
		mCom(1, 0) = mCom(2, 1) = mCom(3, 2) = 1;
		mCom(0, 3) = -m_as[4] / m_as[0];
		mCom(1, 3) = -m_as[3] / m_as[0];
		mCom(2, 3) = -m_as[2] / m_as[0];
		mCom(3, 3) = -m_as[1] / m_as[0];
		Eigen::EigenSolver<Matrix_4x4> eig(mCom, false);
		CMatrix_4x1 mSols = eig.eigenvalues();

		for (int k = 0; k < 4; k++)
		{
			if (std::abs(mSols[k].imag()) < m_eps)
			{
				m_realRoots.push_back(mSols[k].real());
			}
		}
	}

	inline void selectP1(const FloatType *f, const FloatType *P)
	{
		const FloatType *f1 = f;
		const FloatType *f2 = f1 + 3;
		const FloatType *f3 = f2 + 3;
		// compute 3rd row of T
		FloatType T3[3] = {f1[1] * f2[2] - f1[2] * f2[1], f1[2] * f2[0] - f1[0] * f2[2], f1[0] * f2[1] - f1[1] * f2[0]};
		FloatType sign = T3[0] * f3[0] + T3[1] * f3[1] + T3[2] * f3[2];
		if (sign > 0)
		{
			m_f[0] = f + 3;
			m_f[1] = f;
			m_P[0] = P + 3;
			m_P[1] = P;
		}
		else
		{
			m_f[0] = f;
			m_f[1] = f + 3;
			m_P[0] = P;
			m_P[1] = P + 3;
		}
		m_f[2] = f + 6;
		m_P[2] = P + 6;
		m_mP1[0] = m_P[0][0];
		m_mP1[1] = m_P[0][1];
		m_mP1[2] = m_P[0][2];
	}

	void transformCoordinate()
	{
		// compute T matrix
		m_mT(0, 0) = m_f[0][0];
		m_mT(0, 1) = m_f[0][1];
		m_mT(0, 2) = m_f[0][2];
		m_mT(1, 0) = m_f[1][0];
		m_mT(1, 1) = m_f[1][1];
		m_mT(1, 2) = m_f[1][2];
		FloatType cos_beta = m_mT.row(0).dot(m_mT.row(1));
		m_mT.row(2) = m_mT.row(0).cross(m_mT.row(1));
		m_mT.row(1) = m_mT.row(2).cross(m_mT.row(0));
		m_mT.rowwise().normalize();

		// compute N matrix
		m_mN(0, 0) = m_P[1][0] - m_P[0][0];
		m_mN(0, 1) = m_P[1][1] - m_P[0][1];
		m_mN(0, 2) = m_P[1][2] - m_P[0][2];
		m_d12 = m_mN.row(0).norm();
		m_mN(1, 0) = m_P[2][0] - m_P[0][0];
		m_mN(1, 1) = m_P[2][1] - m_P[0][1];
		m_mN(1, 2) = m_P[2][2] - m_P[0][2];
		m_mN.row(2) = m_mN.row(0).cross(m_mN.row(1));
		m_mN.row(1) = m_mN.row(2).cross(m_mN.row(0));
		m_mN.rowwise().normalize();

		// compute b
		m_b = std::sqrt((1 / (1 - (cos_beta * cos_beta))) - 1); // Eq. 3
		if (cos_beta < 0)
		{
			m_b = -m_b;
		}

		// compute phis
		Matrix_3x1 tmp(m_f[2]);
		tmp = m_mT * tmp;
		m_phi[0] = tmp[0] / tmp[2]; // Eq. 8
		m_phi[1] = tmp[1] / tmp[2]; // Eq. 8

		// compute ps
		tmp[0] = m_P[2][0];
		tmp[1] = m_P[2][1];
		tmp[2] = m_P[2][2];
		tmp = m_mN * (tmp - m_mP1); // Eq. 2
		m_p[0] = tmp[0];
		m_p[1] = tmp[1];
	}

private:
	static inline bool checkColinearity(const FloatType *p)
	{
		FloatType p12[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]};
		FloatType p13[3] = {p[6] - p[0], p[7] - p[1], p[8] - p[2]};
		FloatType pcross[] = {p12[2] * p13[1] - p12[1] * p13[2], p12[0] * p13[2] - p12[2] * p13[0], p12[1] * p13[0] - p12[0] * p13[1]};
		FloatType ncross = std::sqrt(pcross[0] * pcross[0] + pcross[1] * pcross[1] + pcross[2] * pcross[2]);
		return (ncross < m_eps);
	}

	// 1st order polynomial multiplication
	// (ax + b) * (cx + d) = (acx^2 + (ad + bc)x + bd)
	// p1 = [x 1]
	// p2 = [x 1]
	// p3 = [x^2 x 1]
	static inline void polynomialMultiplication1(const FloatType *p1, const FloatType *p2, FloatType *p3)
	{
		p3[0] = (p1[0])*(p2[0]);					// x^2
		p3[1] = (p1[0])*(p2[1]) + (p1[1])*(p2[0]);	// x
		p3[2] = (p1[1])*(p2[1]);					// 1
	}

	// 2nd order polynomial multiplication
	// (ax^2 + bx + c) * (dx^2 + ex + f) = (adx^4 + (ae + bd)x^3 + (af + be + cd)x^2 + (bf + ce)x + cf
	// p1 = [x^2 x 1]
	// p2 = [x^2 x 1]
	// p3 = [x^4 x^3 x^2 x 1]
	static inline void polynomialMultiplication2(const FloatType *p1, const FloatType *p2, FloatType *p3)
	{
		p3[0] = (p1[0])*(p2[0]);										// x^4
		p3[1] = (p1[0])*(p2[1]) + (p1[1])*(p2[0]);						// x^3
		p3[2] = (p1[0])*(p2[2]) + (p1[1])*(p2[1]) + (p1[2])*(p2[0]);	// x^2
		p3[3] = (p1[1])*(p2[2]) + (p1[2])*(p2[1]);						// x
		p3[4] = (p1[2])*(p2[2]);										// 1
	}

private:
	const FloatType *m_f[3]; // pointer to 2D feature vector
	const FloatType *m_P[3]; // pointer to 3D feature vector
	Matrix_3x1 m_mP1;
	Matrix_3x3 m_mT;
	Matrix_3x3 m_mN;
	FloatType m_phi[2];
	FloatType m_p[2];
	FloatType m_b;
	FloatType m_d12;
	FloatType m_as[5]; // coefficient of 4th order polynomial
	std::vector<FloatType> m_realRoots; // real root for cos_theta polynomial
};

template <typename FloatType>
const FloatType P3P<FloatType>::m_eps = 1E-7;
}

#endif // P3P_HPP_