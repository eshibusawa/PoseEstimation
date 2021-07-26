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

#include "ThreePoint.hpp"
#include "ThreePointUtil.hpp"

#include <random>

namespace
{
typedef double FloatType;
const double ut_delta = 1E-7;
}

class ThreePointTest: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(ThreePointTest);
CPPUNIT_TEST(rotationTest);
CPPUNIT_TEST(allTest);
CPPUNIT_TEST_SUITE_END();

private:
	std::mt19937 m_rng;

public:
ThreePointTest()
{
}

void setUp()
{
	m_rng = std::mt19937(0);
}

void tearDown()
{
}

void rotationTest()
{
	// construct random rotation
	FloatType q[4] = {};
	FloatType R[9] = {};
	ThreePoint::getRandomRotation(m_rng, R, q);
	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR(R);

	// q -> R test
	ThreePoint::ThreePoint<FloatType>::getRotation(q, R);
	for (int i = 0, k = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			CPPUNIT_ASSERT_DOUBLES_EQUAL(mR(i, j), R[k], ut_delta);
			k++;
		}
	}
}

void allTest()
{
	// construct random rotation
	FloatType R[9] = {};
	ThreePoint::getRandomRotation(m_rng, R);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);

	// construct random translation
	FloatType t[3] = {};
	ThreePoint::getRandomTranslation(m_rng, t);
	Eigen::Matrix<FloatType, 3, 1> mt(t);

	// random scale
	FloatType scale = 1;
	ThreePoint::getRandomScale(m_rng, scale);

	// construct random point on unit shpere
	FloatType X1[9], X2[9];
	ThreePoint::getRandomPoint(3, m_rng, X1);

	// transform
	ThreePoint::ThreePoint<FloatType>::transformPoints(3, scale, R, t, X1, X2);

	FloatType estR[9], estt[3];
	FloatType estScale;
	ThreePoint::ThreePoint<FloatType>::getRT(3, X1, X2, estR, estt, estScale);

	CPPUNIT_ASSERT_DOUBLES_EQUAL(scale, estScale, ut_delta);
	for (int k = 0; k < 9; k++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(R[k], estR[k], ut_delta);
	}
	for (int k = 0; k < 3; k++)
	{
		CPPUNIT_ASSERT_DOUBLES_EQUAL(t[k], estt[k], ut_delta);
	}
}
};

CPPUNIT_TEST_SUITE_REGISTRATION(ThreePointTest);
