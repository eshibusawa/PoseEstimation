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

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

#include "FivePoint.hpp"

#include <random>

namespace
{
typedef double FloatType;
const double ut_delta = 1E-7;
}

class FivePointTest: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(FivePointTest);
CPPUNIT_TEST(polynomial1Test);
CPPUNIT_TEST(polynomial2Test);
CPPUNIT_TEST(determinantTest);
CPPUNIT_TEST(traceTest);
CPPUNIT_TEST(allTest);
CPPUNIT_TEST_SUITE_END();

private:
	std::mt19937 m_rng;

public:
FivePointTest()
{
}

void setUp()
{
	m_rng = std::mt19937(0);
}

void tearDown()
{
}

void polynomial1Test()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng), y = ufd(m_rng), z = ufd(m_rng);
	FloatType x2 = x * x, xy = x * y, xz = x * z, y2 = y * y, yz = y * z, z2 = z * z;

	FloatType c1[4], c2[4];
	for (int i = 0; i < 4; i++)
	{
		c1[i] = ufd(m_rng);
		c2[i] = ufd(m_rng);
	}
	FloatType a1 = c1[0] * x + c1[1] * y + c1[2] * z + c1[3];
	FloatType a2 = c2[0] * x + c2[1] * y + c2[2] * z + c2[3];

	FloatType c12[10];
	FivePoint::FivePoint<FloatType>::polynomial_multiplication1(c1, c2, c12);
	FloatType a12 = c12[0] * x2 + c12[1] * xy + c12[2] * xz +
		c12[3] * y2 + c12[4] * yz +
		c12[5] * z2 +
		c12[6] * x + c12[7] * y + c12[8] * z + c12[9];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void polynomial2Test()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng), y = ufd(m_rng), z = ufd(m_rng);
	FloatType x2 = x * x, xy = x * y, xz = x * z, y2 = y * y, yz = y * z, z2 = z * z;
	FloatType x3 = x * x2, y3 = y * y2, z3 = z * z2;
	FloatType x2y = x2 * y, xy2 = x * y2, x2z = x2 * z, y2z = y2 * z, xz2 = x * z2, yz2 = y * z2, xyz = x * y * z;

	FloatType c1[4];
	for (int i = 0; i < 4; i++)
	{
		c1[i] = ufd(m_rng);
	}
	FloatType c2[10];
	for (int i = 0; i < 10; i++)
	{
		c2[i] = ufd(m_rng);
	}

	FloatType a1 = c1[0] * x + c1[1] * y + c1[2] * z + c1[3];
	FloatType a2 = c2[0] * x2 + c2[1] * xy + c2[2] * xz +
		c2[3] * y2 + c2[4] * yz +
		c2[5] * z2 +
		c2[6] * x + c2[7] * y + c2[8] * z + c2[9];

	FloatType c12[20];
	FivePoint::FivePoint<FloatType>::polynomial_multiplication2(c1, c2, c12);
	FloatType a12 =
		c12[0] * x3 + c12[1] * x2y + c12[2] * xy2 +
		c12[3] * y3 + c12[4] * x2z + c12[5] * xyz +
		c12[6] * y2z + c12[7] * xz2 + c12[8] * yz2 + c12[9] * z3 +
		c12[10] * x2 + c12[11] * xy + c12[12] * y2 +
		c12[13] * xz + c12[14] * yz +
		c12[15] * z2 +
		c12[16] * x + c12[17] * y + c12[18] * z + c12[19];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void determinantTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng), y = ufd(m_rng), z = ufd(m_rng);
	FloatType x2 = x * x, xy = x * y, xz = x * z, y2 = y * y, yz = y * z, z2 = z * z;
	FloatType x3 = x * x2, y3 = y * y2, z3 = z * z2;
	FloatType x2y = x2 * y, xy2 = x * y2, x2z = x2 * z, y2z = y2 * z, xz2 = x * z2, yz2 = y * z2, xyz = x * y * z;

	FloatType E1[9], E2[9], E3[9], E4[9];
	for (int i = 0; i < 9; i++)
	{
		E1[i] = ufd(m_rng);
		E2[i] = ufd(m_rng);
		E3[i] = ufd(m_rng);
		E4[i] = ufd(m_rng);
	}

	FivePoint::FivePoint<FloatType>::Matrix_3x3 E;
	E(0, 0) = x * E1[0] + y * E2[0] + z * E3[0] + E4[0];
	E(0, 1) = x * E1[1] + y * E2[1] + z * E3[1] + E4[1];
	E(0, 2) = x * E1[2] + y * E2[2] + z * E3[2] + E4[2];
	E(1, 0) = x * E1[3] + y * E2[3] + z * E3[3] + E4[3];
	E(1, 1) = x * E1[4] + y * E2[4] + z * E3[4] + E4[4];
	E(1, 2) = x * E1[5] + y * E2[5] + z * E3[5] + E4[5];
	E(2, 0) = x * E1[6] + y * E2[6] + z * E3[6] + E4[6];
	E(2, 1) = x * E1[7] + y * E2[7] + z * E3[7] + E4[7];
	E(2, 2) = x * E1[8] + y * E2[8] + z * E3[8] + E4[8];

	FloatType c[20];
	FivePoint::FivePoint<FloatType>::getDeterminantConstraints(E1, E2, E3, E4, c);
	FloatType det =
		c[0] * x3 + c[1] * x2y + c[2] * xy2 +
		c[3] * y3 + c[4] * x2z + c[5] * xyz +
		c[6] * y2z + c[7] * xz2 + c[8] * yz2 + c[9] * z3 +
		c[10] * x2 + c[11] * xy + c[12] * y2 +
		c[13] * xz + c[14] * yz +
		c[15] * z2 +
		c[16] * x + c[17] * y + c[18] * z + c[19];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(E.determinant(), det, ut_delta);
}

void traceTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng), y = ufd(m_rng), z = ufd(m_rng);
	FloatType x2 = x * x, xy = x * y, xz = x * z, y2 = y * y, yz = y * z, z2 = z * z;
	FloatType x3 = x * x2, y3 = y * y2, z3 = z * z2;
	FloatType x2y = x2 * y, xy2 = x * y2, x2z = x2 * z, y2z = y2 * z, xz2 = x * z2, yz2 = y * z2, xyz = x * y * z;

	FloatType E1[9], E2[9], E3[9], E4[9];
	for (int i = 0; i < 9; i++)
	{
		E1[i] = ufd(m_rng);
		E2[i] = ufd(m_rng);
		E3[i] = ufd(m_rng);
		E4[i] = ufd(m_rng);
	}

	FivePoint::FivePoint<FloatType>::Matrix_3x3 E;
	E(0, 0) = x * E1[0] + y * E2[0] + z * E3[0] + E4[0];
	E(0, 1) = x * E1[1] + y * E2[1] + z * E3[1] + E4[1];
	E(0, 2) = x * E1[2] + y * E2[2] + z * E3[2] + E4[2];
	E(1, 0) = x * E1[3] + y * E2[3] + z * E3[3] + E4[3];
	E(1, 1) = x * E1[4] + y * E2[4] + z * E3[4] + E4[4];
	E(1, 2) = x * E1[5] + y * E2[5] + z * E3[5] + E4[5];
	E(2, 0) = x * E1[6] + y * E2[6] + z * E3[6] + E4[6];
	E(2, 1) = x * E1[7] + y * E2[7] + z * E3[7] + E4[7];
	E(2, 2) = x * E1[8] + y * E2[8] + z * E3[8] + E4[8];
	FivePoint::FivePoint<FloatType>::Matrix_3x3 EEt = E * E.transpose();
	FivePoint::FivePoint<FloatType>::Matrix_3x3 T = EEt * E - (EEt.trace()) * E / 2;

	FloatType cT[180];
	FivePoint::FivePoint<FloatType>::getSingularValueConstraints(E1, E2, E3, E4, cT);
	FivePoint::FivePoint<FloatType>::Matrix_3x3 T2;
	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++)
		{
			const FloatType *c = cT + (3 * j + i) * 20;
			T2(i, j) =
				c[0] * x3 + c[1] * x2y + c[2] * xy2 +
				c[3] * y3 + c[4] * x2z + c[5] * xyz +
				c[6] * y2z + c[7] * xz2 + c[8] * yz2 + c[9] * z3 +
				c[10] * x2 + c[11] * xy + c[12] * y2 +
				c[13] * xz + c[14] * yz +
				c[15] * z2 +
				c[16] * x + c[17] * y + c[18] * z + c[19];
		}
	}

	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++)
		{
			CPPUNIT_ASSERT_DOUBLES_EQUAL(T(i, j), T2(i, j), ut_delta);
		}
	}
}

void allTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	FloatType pts1[5 * 2], pts2[5 * 2];

	for (int i = 0; i < 10; i++)
	{
		// this point correspondences has no physical meanings!
		pts1[i] = ufd(m_rng);
		pts2[i] = ufd(m_rng);
	}

	int nSolutions = 0;
	std::vector<FloatType> Es;
	FivePoint::FivePoint<FloatType> solver;
	bool ret = solver.getMatrix(pts1, pts2, nSolutions, Es);
	CPPUNIT_ASSERT(ret);
	for (int k = 0; k < nSolutions; k++)
	{
		FloatType *pE = &(Es[9 * k]);
		Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mE(pE);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mE.determinant(), ut_delta); // check determinant constraints

		FivePoint::FivePoint<FloatType>::Matrix_3x3 EEt = mE * mE.transpose();
		FivePoint::FivePoint<FloatType>::Matrix_3x3 T = EEt * mE - (EEt.trace()) * mE / 2;
		for (int j = 0; j < 3; j++)
		{
			for (int i = 0; i < 3; i++)
			{
				 // check singular value constraints
				CPPUNIT_ASSERT_DOUBLES_EQUAL(0, T(j, i), ut_delta);
			}
		}

		for (int i = 0; i < 5; i++)
		{
			Eigen::Matrix<FloatType, 3, 1> p1;
			Eigen::Matrix<FloatType, 3, 1> p2;
			p1[0] = pts1[2*i];
			p1[1] = pts1[2*i + 1];
			p1[2] = 1;
			p2[0] = pts2[2*i];
			p2[1] = pts2[2*i + 1];
			p2[2] = 1;
			FloatType p2Ep1 = p2.transpose() * mE * p1;
			CPPUNIT_ASSERT_DOUBLES_EQUAL(0, p2Ep1, ut_delta); // check epipolar constraints
		}
	}
}

};

CPPUNIT_TEST_SUITE_REGISTRATION(FivePointTest);
