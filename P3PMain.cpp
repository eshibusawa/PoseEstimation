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

#include "P3PUtil.hpp"
#include "P3PRansacModel.hpp"
#include "Ransac.hpp"

#include <iostream>
#include <iterator>

namespace
{
typedef double FloatType;
}

bool ransac_test();

int main(int argc, char** argv)
{
	using std::cout;
	using std::endl;
	bool ret = false;

	// (1) P3P ransac
	ret = ransac_test();
	if (!ret)
	{
		return EXIT_FAILURE;
	}
	cout << endl;

	return EXIT_SUCCESS;
}

bool ransac_test()
{
	using std::cout;
	using std::endl;
	cout << ">>>> RANSAC test:" << endl;

	PoseEstimation::RANSAC<P3P::RansacModel<FloatType>, P3P::RT<FloatType> > p3p;
	const float epsilon = 0.4f;
	p3p.setEpsilon(epsilon);

	const int nAll = 30;
	const int nOutlier = static_cast<int>(nAll * epsilon);
	const int nInlier = nAll - nOutlier;

	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR;
	Eigen::Matrix<FloatType, 3, 1> mt;
	std::vector<FloatType> p2d, p3d;
	P3P::getRandomProjections(nInlier, mR.data(), mt.data(), p2d, p3d);
	std::vector<int> trueIndices(nInlier, 1);
	{
		FloatType Ro[9], to[3];
		std::vector<FloatType> p2do, p3do;
		P3P::getRandomProjections(nOutlier, Ro, to, p2do, p3do);
		std::vector<int> outlierIndices(nOutlier, 0);

		std::copy(p2do.begin(), p2do.end(), std::back_inserter(p2d));
		std::copy(p3do.begin(), p3do.end(), std::back_inserter(p3d));
		std::copy(outlierIndices.begin(), outlierIndices.end(), std::back_inserter(trueIndices));
	}

	p3p.setThreshold(1E-9);
	p3p.setCorrespondences(p2d, p3d);

	P3P::RT<FloatType> estP;
	std::vector<int> estIndices;
	int estNInlier = 0;
	bool ret = p3p.compute(estP, estNInlier, estIndices);
	if (!ret)
	{
		return false;
	}

	cout << "True Rotation:" << endl;
	cout << mR << endl;
	cout << "Found Rotation:" << endl;
	cout << Eigen::Map<Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> >(estP.R) << endl;
	cout << endl;
	cout << "True Translation:" << endl;
	cout << mt << endl;
	cout << "Found Translation:" << endl;
	cout << Eigen::Map<Eigen::Matrix<FloatType, 3, 1> >(estP.t) << endl;
	cout << endl;

	cout << "True indices:" << endl;
	cout << "(" << nInlier << ") ";
	std::copy(trueIndices.begin(), trueIndices.end(), std::ostream_iterator<int>(cout, " "));
	cout << endl;
	cout << "Found indices:" << endl;
	cout << "(" << estNInlier << ") ";
	std::copy(estIndices.begin(), estIndices.end(), std::ostream_iterator<int>(cout, " "));
	cout << endl;

	return true;
}
