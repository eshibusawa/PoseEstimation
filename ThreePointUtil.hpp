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

#ifndef THREE_POINT_UTIL_HPP_
#define THREE_POINT_UTIL_HPP_

#include "ThreePoint.hpp"

#include <Eigen/Geometry>

#include <random>

namespace ThreePoint
{
template <typename FloatType, typename RNG>
void getRandomRotation(RNG &rng, FloatType *R, FloatType *q = NULL)
{
	// construct random rotation
	std::uniform_real_distribution<FloatType> urd(0, static_cast<FloatType>(2 * M_PI));
	FloatType rx = urd(rng);
	FloatType ry = urd(rng);
	FloatType rz = urd(rng);
	Eigen::Quaternion<FloatType> mq;
	mq = Eigen::AngleAxis<FloatType>(rz, Eigen::Matrix<FloatType, 3, 1>::UnitZ()) *
		Eigen::AngleAxis<FloatType>(ry, Eigen::Matrix<FloatType, 3, 1>::UnitY()) *
		Eigen::AngleAxis<FloatType>(rx, Eigen::Matrix<FloatType, 3, 1>::UnitX());

	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
	mR = mq.toRotationMatrix();

	if (q != NULL)
	{
		q[0] = mq.x();
		q[1] = mq.y();
		q[2] = mq.z();
		q[3] = mq.w();
	}
}

template <typename FloatType, typename RNG>
void getRandomTranslation(RNG &rng, FloatType *t)
{
	// cosntruct random translation
	std::uniform_real_distribution<FloatType> urd(static_cast<FloatType>(-0.5), static_cast<FloatType>(0.5));
	t[0] = 5 + urd(rng);
	t[1] = 5 + urd(rng);
	t[2] = 5 + urd(rng);
}

template <typename FloatType, typename RNG>
inline void getRandomScale(RNG &rng, FloatType &s)
{
	std::uniform_real_distribution<FloatType> urd(static_cast<FloatType>(-0.05), static_cast<FloatType>(0.05));
	s = 1 + urd(rng);
}

template <typename FloatType, typename RNG>
void getRandomPoint(int n, RNG &rng, FloatType *X)
{
	// construct random point on unit shpere
	std::uniform_real_distribution<FloatType> urd(static_cast<FloatType>(-0.5), static_cast<FloatType>(0.5));
	for (int i = 0; i < 3 * n; i++)
	{
		X[i] = urd(rng);
	}
	Eigen::Map<Eigen::Matrix<FloatType, 3, Eigen::Dynamic> > mX(X, 3, n);
	mX.colwise().normalize();
}

template <typename FloatType>
void getRandomCorrespondences(int n, FloatType *R, FloatType *t, FloatType &s, std::vector<FloatType> &pts1, std::vector<FloatType> &pts2)
{
	std::random_device rd;
	std::mt19937 rng(rd());

	// construct random rotation
	getRandomRotation(rng, R);

	// construct random translation
	getRandomTranslation(rng, t);

	// random scale
	getRandomScale(rng, s);

	// construct random point on unit shpere
	pts1.resize(3 * n);
	getRandomPoint(n, rng, &(pts1[0]));

	// transform
	pts2.resize(3 * n);
	ThreePoint<FloatType>::transformPoints(n, s, R, t, &(pts1[0]), &(pts2[0]));
}

}

#endif // THREE_POINT_UTIL_HPP_