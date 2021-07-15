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

#ifndef FIVE_POINT_UTIL_HPP_
#define FIVE_POINT_UTIL_HPP_

#include <Eigen/Dense>

#include <random>
#include <vector>

namespace FivePoint
{

template <typename FloatType, typename RNG>
void getRandomPose(RNG &rng, FloatType *R, FloatType *t)
{
    using std::cos;
    using std::sin;
	std::uniform_real_distribution<FloatType> urd(0, static_cast<FloatType>(M_PI));
	FloatType phi   = 2 * urd(rng);
	FloatType theta = urd(rng);
	FloatType psi   = 2 * urd(rng);

	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> >mR(R);

	mR(0, 0) = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
	mR(0, 1) = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
	mR(0, 2) = sin(psi) * sin(theta);
	mR(1, 0) = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
	mR(1, 1) = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
	mR(1, 2) = cos(psi) * sin(theta);
	mR(2, 0) = sin(theta) * sin(phi);
	mR(2, 1) = -sin(theta) * cos(phi);
	mR(2, 2) = cos(theta);
	t[0] = 0.0;
	t[1] = 0.0;
	t[2] = 6.0;
}

template <typename FloatType, typename RNG>
void getRandomPoints(RNG &rng, int n, std::vector<FloatType> &p)
{
    using std::cos;
    using std::sin;
	const FloatType pi = static_cast<FloatType>(M_PI);
	std::uniform_real_distribution<FloatType> urd(0, 1);

	p.resize(0);
	p.reserve(3 * n);
	for (int i = 0; i < n; i++)
	{
		FloatType theta = pi * urd(rng), phi = 2 * pi * urd(rng), R = 2 * urd(rng);
		FloatType X =  sin(theta) * sin(phi) * R;
		FloatType Y = -sin(theta) * cos(phi) * R;
		FloatType Z =  cos(theta) * R;

		p.push_back(X);
		p.push_back(Y);
		p.push_back(Z);
	}
}

template <typename FloatType>
void getRandomCorrespondences(int n, FloatType *R, FloatType *t, std::vector<FloatType> &pts1, std::vector<FloatType> &pts2)
{
	std::random_device rd;
	std::mt19937 rng(rd());
	FloatType R1[9], R2[9], t1[3], t2[3];
	getRandomPose(rng, R1, t1);
	getRandomPose(rng, R2, t2);

	std::vector<FloatType> pts;
	getRandomPoints(rng, n, pts);

	Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> > mP(&(pts[0]), 3, n);
	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR1(R1);
	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR2(R2);
	Eigen::Matrix<FloatType, 3, 1> mt1(t1);
	Eigen::Matrix<FloatType, 3, 1> mt2(t2);

	Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> mP1(3, n);
	mP1 = (mR1 * mP).colwise() + mt1;
	pts1.resize(2*n);
	Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> > mp1(&(pts1[0]), 2, n);
	mp1.row(0) = mP1.row(0).array() / mP1.row(2).array();
	mp1.row(1) = mP1.row(1).array() / mP1.row(2).array();

	Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> mP2(3, n);
	mP2 = (mR2 * mP).colwise() + mt2;
	pts2.resize(2*n);
	Eigen::Map<Eigen::Matrix<FloatType, Eigen::Dynamic, Eigen::Dynamic> > mp2(&(pts2[0]), 2, n);
	mp2.row(0) = mP2.row(0).array() / mP2.row(2).array();
	mp2.row(1) = mP2.row(1).array() / mP2.row(2).array();

	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 1> > mt(t);
	mR = mR2*mR1.transpose();
	mt = mt2 - mR*mt1;
}

template <typename FloatType>
inline void getSkewSymmetricMatrix(const FloatType *t, FloatType *T)
{
	// skew symmetric matrix
	T[0] = T[4] = T[8] = 0;
	T[3] = t[2];
	T[1] = -T[3];
	T[2] = t[1];
	T[6] = -T[2];
	T[7] = t[0];
	T[5] = -T[7];
}

template <typename FloatType>
inline FloatType getSampsonsError(const FloatType *E, const FloatType *p1, const FloatType *p2)
{
	// Sampson's error
	FloatType Ep10 = (E[0] * p1[0] + E[1] * p1[1] + E[2]);
	FloatType Ep11 = (E[3] * p1[0] + E[4] * p1[1] + E[5]);
	FloatType p2E0 = (p2[0] * E[0] + p2[1] * E[3] + E[6]);
	FloatType p2E1 = (p2[0] * E[1] + p2[1] * E[4] + E[7]);
	// p2 * E * p1
	FloatType p2Ep1 = (p2[0]) * Ep10 + (p2[1]) * Ep11 +
		(E[6] * p1[0] + E[7] * p1[1] + E[8]);
	return p2Ep1 / std::sqrt(p2E0 * p2E0 + p2E1 * p2E1 + Ep10 * Ep10 + Ep11 * Ep11);
}

}

#endif // FIVE_POINT_UTIL_HPP_