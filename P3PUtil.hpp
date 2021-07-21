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

#ifndef P3P_UTIL_HPP_
#define P3P_UTIL_HPP_

#include <Eigen/Dense>

#include <random>
#include <vector>

namespace P3P
{
template <typename FloatType, typename RNG>
void getRandomPose(RNG &rng, FloatType *R, FloatType *t)
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	// compute random rotation
	Eigen::Matrix<FloatType, 3, 3> mR1;
	for (int k = 0; k < 3; k++)
	{
		for (int l = 0; l < 3; l++)
		{
			mR1(k, l) = ufd(rng);
		}
	}
	Eigen::JacobiSVD<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > svd(mR1, Eigen::ComputeFullU | Eigen::ComputeFullV);
	mR1 = svd.matrixU() * svd.matrixV().transpose();
	if (mR1.determinant() < 0)
	{
		mR1.col(0) = -mR1.col(0);
	}
	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
	mR = mR1;

	// compute random translation
	Eigen::Map<Eigen::Matrix<FloatType, 3, 1> > mt(t);
	mt[0] = 0.5 * ufd(rng);
	mt[1] = 0.5 * ufd(rng);
	mt[2] = 6 + 0.5 * ufd(rng);
}

template <typename FloatType, typename RNG>
void getRandomPoints(RNG &rng, int n, std::vector<FloatType> &p)
{
	std::uniform_real_distribution<FloatType> ufd(-1, 1);
	// compute random point
	p.reserve(3 * n);
	p.resize(0);
	for (int k = 0; k < n; k++)
	{
		p.push_back(2 * ufd(rng));
		p.push_back(2 * ufd(rng));
		p.push_back(2 * ufd(rng));
	}
}

template <typename FloatType>
void getRandomProjections(int n, FloatType *R, FloatType *t, std::vector<FloatType> &p2d, std::vector<FloatType> &p3d)
{
	// x = R*X + t
	std::random_device rd;
	std::mt19937 rng(rd());
	getRandomPose(rng, R, t);
	getRandomPoints(rng, n, p3d);

	Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
	Eigen::Map<Eigen::Matrix<FloatType, 3, 1> > mt(t);
	Eigen::Map<Eigen::Matrix<FloatType, 3, Eigen::Dynamic> > mX(&(p3d[0]), 3, n);
	// project 3D point
	Eigen::Matrix<FloatType, 3, Eigen::Dynamic> mTmp(3, n);
	mTmp = (mR*mX).colwise() + mt;
	// get 2D Point
	p2d.resize(2 * n);
	Eigen::Map<Eigen::Matrix<FloatType, 2, Eigen::Dynamic> > mx(&(p2d[0]), 2, n);
	// perspective projection
	mx.row(0) = mTmp.row(0).array() / mTmp.row(2).array();
	mx.row(1) = mTmp.row(1).array() / mTmp.row(2).array();
}
}

#endif // P3P_UTIL_HPP_