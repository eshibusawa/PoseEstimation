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

#include "ThreePointRansacModel.hpp"
#include "ThreePointUtil.hpp"
#include "Ransac.hpp"

#include <iostream>
#include <iterator>

bool ransac_test();

int main(int argc, char** argv)
{
	using std::cout;
	using std::endl;
	bool ret = false;

	// (1) three point ransac
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
	typedef double FloatType;
	using std::cout;
	using std::endl;
	cout << ">>>> RANSAC test:" << endl;

	PoseEstimation::RANSAC<ThreePoint::RansacModel<FloatType>, ThreePoint::Sim3<FloatType> > Sim3E;
	const float epsilon = 0.4f;
	Sim3E.setEpsilon(epsilon);

	const int nAll = 30;
	const int nOutlier = static_cast<int>(nAll * epsilon);
	const int nInlier = nAll - nOutlier;

	ThreePoint::Sim3<FloatType> trueRT;
	std::vector<FloatType> pts1, pts2;
	ThreePoint::getRandomCorrespondences(nInlier, trueRT.R, trueRT.t, trueRT.s, pts1, pts2);
	std::vector<int> trueIndices(nInlier, 1);
	{
		FloatType so;
		FloatType Ro[9], to[3];
		std::vector<FloatType> pts1o, pts2o;
		ThreePoint::getRandomCorrespondences(nOutlier, Ro, to, so, pts1o, pts2o);
		std::vector<int> outlierIndices(nOutlier, 0);

		std::copy(pts1o.begin(), pts1o.end(), std::back_inserter(pts1));
		std::copy(pts2o.begin(), pts2o.end(), std::back_inserter(pts2));
		std::copy(outlierIndices.begin(), outlierIndices.end(), std::back_inserter(trueIndices));
	}

	Sim3E.setThreshold(1E-9);
	Sim3E.setCorrespondences(pts1, pts2);

	ThreePoint::Sim3<FloatType> estRT;
	std::vector<int> estIndices;
	int estNInlier = 0;
	bool ret = Sim3E.compute(estRT, estNInlier, estIndices);
	if (!ret)
	{
		return false;
	}

	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mTrueR(trueRT.R);
	Eigen::Matrix<FloatType, 3, 1> mTruet(trueRT.t);
	Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mEstR(estRT.R);
	Eigen::Matrix<FloatType, 3, 1> mEstt(estRT.t);

	cout << "True Rotation:" << endl;
	cout << mTrueR << endl;
	cout << "Found Rotation:" << endl;
	cout << mEstR << endl;
	cout << endl;

	cout << "True Translation:" << endl;
	cout << mTruet << endl;
	cout << "Found Translation:" << endl;
	cout << mEstt << endl;
	cout << endl;

	cout << "True Scale:" << endl;
	cout << trueRT.s << endl;
	cout << "Found Scale:" << endl;
	cout << estRT.s << endl;
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
