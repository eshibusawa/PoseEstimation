// This file is part of PoseEstimation.
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

#include <cppunit/TestCase.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>

#include "P3P.hpp"

#include <ctime>
#include <random>

namespace
{
typedef double FloatType;
const double ut_delta = 1E-7;
}

class P3PTest: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(P3PTest);
CPPUNIT_TEST(polynomial1Test);
CPPUNIT_TEST(polynomial2Test);
CPPUNIT_TEST(convertPoseTest);
CPPUNIT_TEST(degenerateCaseTest);
CPPUNIT_TEST(allTest);
CPPUNIT_TEST_SUITE_END();

private:
	std::mt19937 m_rng;
	FloatType m_R[9];
	FloatType m_t[3];
	FloatType m_X[9];
	FloatType m_x[9];

private:
	static inline void projection(const FloatType *X, const FloatType *R, const FloatType *t, FloatType *x)
	{
		// x = R'*(X - t)
		// x is unitary feature vector
		Eigen::Map<const Eigen::Matrix<FloatType, 3, 3> > mX(X);
		Eigen::Map<const Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
		Eigen::Map<const Eigen::Matrix<FloatType, 3, 1> > mt(t);
		Eigen::Map<Eigen::Matrix<FloatType, 3, 3> > mx(x);
		mx = mR.transpose() * (mX.colwise() - mt);
		mx.colwise().normalize();
	}

public:
P3PTest() : m_rng(std::time(0))
{
}

void setUp()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	// compute random rotation
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(m_R);
	for (int k = 0; k < 3; k++)
	{
		for (int l = 0; l < 3; l++)
		{
			mR(k, l) = ufd(m_rng);
		}
	}
	Eigen::JacobiSVD<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > svd(mR, Eigen::ComputeFullU | Eigen::ComputeFullV);
	mR = svd.matrixU() * svd.matrixV().transpose();
	if (mR.determinant() < 0)
	{
		mR.col(0) = -mR.col(0);
	}
	// compute random translation
	Eigen::Map<Eigen::Matrix<FloatType, 3, 1> > mt(m_t);
	mt[0] = 0.5 * ufd(m_rng);
	mt[1] = 0.5 * ufd(m_rng);
	mt[2] = 6 + 0.5 * ufd(m_rng);
	mt = -mR * mt;

	// compute random point
	for (int k = 0; k < 9; k++)
	{
		m_X[k] = 2 * ufd(m_rng);
	}

	projection(m_X, m_R, m_t, m_x);
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
	P3P::P3P<FloatType>::polynomialMultiplication1(c1, c2, c12);
	FloatType a12 = c12[0] * x2 + c12[1] * x + c12[2];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void polynomial2Test()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);

	FloatType x = ufd(m_rng);
	FloatType x2 = x * x, x3 = x * x2, x4 = x * x3;

	FloatType c1[3], c2[3];
	for (int i = 0; i < 3; i++)
	{
		c1[i] = ufd(m_rng);
		c2[i] = ufd(m_rng);
	}

	FloatType a1 = c1[0] * x2 + c1[1] * x + c1[2];
	FloatType a2 = c2[0] * x2 + c2[1] * x + c2[2];

	FloatType c12[5];
	P3P::P3P<FloatType>::polynomialMultiplication2(c1, c2, c12);
	FloatType a12 = c12[0] * x4 + c12[1] * x3 + c12[2] * x2 + c12[3] * x + c12[4];

	CPPUNIT_ASSERT_DOUBLES_EQUAL(a1 * a2, a12, ut_delta);
}

void convertPoseTest()
{
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3> > mX(m_X);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(m_R);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 1> > mt(m_t);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3> > mx(m_x);

	P3P::P3P<FloatType>::convertPose(m_R, m_t);
	Eigen::Matrix<FloatType, 3, 3> x2 = (mR * mX).colwise() + mt;
	x2.colwise().normalize();
	for (int j = 0; j < 3; j++)
	{
		for (int i = 0; i < 3; i++)
		{
			CPPUNIT_ASSERT_DOUBLES_EQUAL(mx(j, i), x2(j, i), ut_delta);
		}
	}
}

void degenerateCaseTest()
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	FloatType lambda = ufd(m_rng);
	FloatType XNG[9], xNG[9];
	for (int k = 0; k < 9; k++)
	{
		XNG[k] = m_X[k];
	}
	// compute colinear points
	XNG[6] = m_X[0] + (m_X[3] - m_X[0]) / lambda;
	XNG[7] = m_X[1] + (m_X[4] - m_X[1]) / lambda;
	XNG[8] = m_X[2] + (m_X[5] - m_X[2]) / lambda;
	projection(XNG, m_R, m_t, xNG);

	int nSols = 0;
	std::vector<FloatType> Ps;
	P3P::P3P<FloatType> p3p;
	bool ret = p3p.getMatrix(xNG, XNG, nSols, Ps);
	CPPUNIT_ASSERT(!ret);
}

void allTest()
{
	int nSols = 0;
	std::vector<FloatType> Ps;
	P3P::P3P<FloatType> p3p;
	bool ret = p3p.getMatrix(m_x, m_X, nSols, Ps);
	CPPUNIT_ASSERT(ret);

	for (int k = 0; k < nSols; k++)
	{
		FloatType *pC = &(Ps[(3 + 9) * k]);
		FloatType *pR = pC + 3;
		FloatType x[9];

		Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(pR);
		Eigen::Matrix<FloatType, 3, 3> mRR = mR.transpose() * mR;
		// check orthogonality
		CPPUNIT_ASSERT_DOUBLES_EQUAL(1, mRR(0, 0), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(1, mRR(1, 1), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(1, mRR(2, 2), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(0, 1), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(0, 2), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(1, 0), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(1, 2), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(2, 0), ut_delta);
		CPPUNIT_ASSERT_DOUBLES_EQUAL(0, mRR(2, 1), ut_delta);

		projection(m_X, pR, pC, x);
		for (int l = 0; l < 3; l++)
		{
			// check reprojection
			CPPUNIT_ASSERT_DOUBLES_EQUAL(m_x[3 * l] / m_x[3 * l + 2], x[3 * l] / x[3 * l + 2], ut_delta);
			CPPUNIT_ASSERT_DOUBLES_EQUAL(m_x[3 * l + 1] / m_x[3 * l + 2], x[3 * l + 1] / x[3 * l + 2], ut_delta);
			// This test fails in the following pose / points, however, p3p.m also fails.
			// R = [0.0605363 -0.629406 -0.774715; -0.806916 -0.487721  0.333189; -0.587556   0.60496 -0.537402];
			// t = [4.56167; -2.37697; 3.33159];
			// X = [1.88968 0.377921 1.22166; 0.243473 -0.220319 -0.197514; 1.58067 -1.37491 0.0545471];
			// x = [-0.301917 0.115956 -0.00685386; -0.15865 -0.190168 -0.18273; 0.940041 0.97488 0.983139];
		}
	}
}
};

CPPUNIT_TEST_SUITE_REGISTRATION(P3PTest);
