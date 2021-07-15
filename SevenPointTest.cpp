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

#include "SevenPoint.hpp"

#include <random>

namespace
{
typedef double FloatType;
const double ut_delta = 1E-7;
}

class SevenPointTest: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(SevenPointTest);
CPPUNIT_TEST(polynomial1Test);
CPPUNIT_TEST(polynomial2Test);
CPPUNIT_TEST(determinantTest);
CPPUNIT_TEST(allTest);
CPPUNIT_TEST_SUITE_END();

private:
	std::mt19937 m_rng;

public:
SevenPointTest()
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

	FloatType x = ufd(m_rng);
	FloatType x2 = x * x;

	FloatType c1[2], c2[2];
	for (int i = 0; i < 2; i++)
	{
		c1[i] = ufd(m_rng);
		c2[i] = ufd(m_rng);
	}
	FloatType a1 = c1[0] * x + c1[1];
	FloatType a2 = c2[0] * x + c2[1];

	FloatType c12[3];
	SevenPoint::SevenPoint<FloatType>::polynomial_multiplication1(c1, c2, c12);
	FloatType a12 = c12[0] * x2 + c12[1] * x + c12[2];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void polynomial2Test()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng);
	FloatType x2 = x * x, x3 = x * x2;

	FloatType c1[2], c2[3];
	for (int i = 0; i < 2; i++)
	{
		c1[i] = ufd(m_rng);
	}
	for (int i = 0; i < 3; i++)
	{
		c2[i] = ufd(m_rng);
	}

	FloatType a1 = c1[0] * x + c1[1];
	FloatType a2 = c2[0] * x2 + c2[1] * x + c2[2];

	FloatType c12[4];
	SevenPoint::SevenPoint<FloatType>::polynomial_multiplication2(c1, c2, c12);
	FloatType a12 = c12[0] * x3 + c12[1] * x2 + c12[2] * x + c12[3];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void determinantTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng);
	FloatType x2 = x * x, x3 = x * x2;

	FloatType F1[9], F2[9];
	for (int i = 0; i < 9; i++)
	{
		F1[i] = ufd(m_rng);
		F2[i] = ufd(m_rng);
	}

	SevenPoint::SevenPoint<FloatType>::Matrix_3x3 F;
	F(0, 0) = x * F1[0] + F2[0];
	F(0, 1) = x * F1[1] + F2[1];
	F(0, 2) = x * F1[2] + F2[2];
	F(1, 0) = x * F1[3] + F2[3];
	F(1, 1) = x * F1[4] + F2[4];
	F(1, 2) = x * F1[5] + F2[5];
	F(2, 0) = x * F1[6] + F2[6];
	F(2, 1) = x * F1[7] + F2[7];
	F(2, 2) = x * F1[8] + F2[8];

	FloatType c[4];
	SevenPoint::SevenPoint<FloatType>::getDeterminantConstraints(F1, F2, c);
	FloatType det =	c[0] * x3 + c[1] * x2 + c[2] * x + c[3];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(F.determinant(), det, ut_delta);
}

void allTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	FloatType pts1[7 * 2], pts2[7 * 2];

	for (int i = 0; i < 14; i++)
	{
		// this point correspondences has no physical meanings!
		pts1[i] = ufd(m_rng);
		pts2[i] = ufd(m_rng);
	}

	int nSolutions = 0;
	std::vector<FloatType> Fs;
	SevenPoint::SevenPoint<FloatType> solver;
	bool ret = solver.getMatrix(pts1, pts2, nSolutions, Fs);
	CPPUNIT_ASSERT(ret);

	for (int k = 0; k < nSolutions; k++)
	{
		FloatType *pF = &(Fs[9 * k]);
		Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mF(pF);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mF.determinant(), ut_delta); // check determinant constraints

		for (int i = 0; i < 7; i++)
		{
			Eigen::Matrix<FloatType, 3, 1> p1;
			Eigen::Matrix<FloatType, 3, 1> p2;
			p1[0] = pts1[2*i];
			p1[1] = pts1[2*i + 1];
			p1[2] = 1;
			p2[0] = pts2[2*i];
			p2[1] = pts2[2*i + 1];
			p2[2] = 1;
			FloatType p2Fp1 = p2.transpose() * mF * p1;
			CPPUNIT_ASSERT_DOUBLES_EQUAL(0, p2Fp1, ut_delta); // check epipolar constraints
		}
	}
}
};

CPPUNIT_TEST_SUITE_REGISTRATION(SevenPointTest);
