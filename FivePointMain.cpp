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

#include "FivePoint.hpp"
#include "SevenPoint.hpp"
#include "FivePointRansacModel.hpp"
#include "FivePointUtil.hpp"
#include "Ransac.hpp"

#include <iostream>
#include <iterator>

namespace
{
typedef double FloatType;
}

template <typename Model>
bool ransac_test()
{
	using std::cout;
	using std::endl;
	cout << ">>>> RANSAC test:" << endl;

	Model RP;
	const float epsilon = 0.4f;
	RP.setEpsilon(epsilon);

	const int nAll = 30;
	const int nOutlier = static_cast<int>(nAll * epsilon);
	const int nInlier = nAll - nOutlier;

	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR;
	Eigen::Matrix<FloatType, 3, 1> mt;
	std::vector<FloatType> pts1, pts2;
	FivePoint::getRandomCorrespondences(nInlier, mR.data(), mt.data(), pts1, pts2);
	std::vector<int> trueIndices(nInlier, 1);
	{
		FloatType Ro[9], to[3];
		std::vector<FloatType> pts1o, pts2o;
		FivePoint::getRandomCorrespondences(nOutlier, Ro, to, pts1o, pts2o);
		std::vector<int> outlierIndices(nOutlier, 0);

		std::copy(pts1o.begin(), pts1o.end(), std::back_inserter(pts1));
		std::copy(pts2o.begin(), pts2o.end(), std::back_inserter(pts2));
		std::copy(outlierIndices.begin(), outlierIndices.end(), std::back_inserter(trueIndices));
	}

	RP.setThreshold(1E-9);
	RP.setCorrespondences(pts1, pts2);

	FivePoint::E<FloatType> estM;
	std::vector<int> estIndices;
	int estNInlier = 0;
	bool ret = RP.compute(estM, estNInlier, estIndices);
	if (!ret)
	{
		return false;
	}
	Eigen::Map< Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mEstM(estM.E);

	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mT;
	FivePoint::getSkewSymmetricMatrix(mt.data(), mT.data());
	Eigen::Matrix<FloatType, 3, 3> mTrueM = (mT*mR).normalized();
	if (std::signbit(mTrueM(0, 0)) != std::signbit(mEstM(0, 0)))
	{
		mEstM = -mEstM;
	}
	cout << "True Matrix:" << endl;
	cout << mTrueM << endl;
	cout << "Found Matrix:" << endl;
	cout << mEstM.normalized() << endl;
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

int main(int argc, char** argv)
{
	typedef PoseEstimation::RANSAC<FivePoint::RansacModel<FloatType, FivePoint::FivePoint<FloatType> >, FivePoint::E<FloatType> > FPR;
	typedef PoseEstimation::RANSAC<FivePoint::RansacModel<FloatType, SevenPoint::SevenPoint<FloatType> >, FivePoint::E<FloatType> > SPR;

	using std::cout;
	using std::endl;
	bool ret = false;

	// (1) five point ransac
	cout << "Five Point (Essential Matrix)" << endl;
	ret = ransac_test<FPR>();
	if (!ret)
	{
		return EXIT_FAILURE;
	}
	cout << endl;

	// (2) seven point ransac
	cout << "Seven Point (Fundamental Matrix)" << endl;
	ret = ransac_test<SPR>();
	if (!ret)
	{
		return EXIT_FAILURE;
	}
	cout << endl;

	return EXIT_SUCCESS;
}
