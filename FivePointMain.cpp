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

// The views and conclusions contained in the software and documentation are those
// of the authors and should not be interpreted as representing official policies,
//   either expressed or implied, of the FreeBSD Project.

#include "FivePointRansacModel.hpp"
#include "FivePointUtil.hpp"
#include "Ransac.hpp"

#include <iostream>
#include <iterator>

bool ransac_test();

int main(int argc, char** argv)
{
	bool ret = false;

	// (1) ransac
	ret = ransac_test();
	if (!ret)
	{
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

bool ransac_test()
{
	typedef double FloatType;
	using std::cout;
	using std::endl;
	cout << ">>>> RANSAC test:" << endl;

	PoseEstimation::RANSAC<FivePoint::RansacModel<FloatType>, FivePoint::E<FloatType> > FPRP;
	const float epsilon = 0.4f;
	FPRP.setEpsilon(epsilon);

	const int nAll = 30;
	const int nOutlier = static_cast<int>(nAll * epsilon);
	const int nInlier = nAll - nOutlier;

	FloatType R[9], t[3];
	std::vector<FloatType> pts1, pts2;
	FivePoint::getRandomCorrespondenses(nInlier, R, t, pts1, pts2);
	std::vector<int> trueIndices(nInlier, 1);
	{
		FloatType Ro[9], to[3];
		std::vector<FloatType> pts1o, pts2o;
		FivePoint::getRandomCorrespondenses(nOutlier, Ro, to, pts1o, pts2o);
		std::vector<int> outlierIndices(nOutlier, 0);

		std::copy(pts1o.begin(), pts1o.end(), std::back_inserter(pts1));
		std::copy(pts2o.begin(), pts2o.end(), std::back_inserter(pts2));
		std::copy(outlierIndices.begin(), outlierIndices.end(), std::back_inserter(trueIndices));
	}

	FPRP.setThreshold(1E-9);
	FPRP.setCorrespondenses(pts1, pts2);

	FivePoint::E<FloatType> estE;
	std::vector<int> estIndices;
	int estNInlier = 0;
	bool ret = FPRP.compute(estE, estNInlier, estIndices);
	if (!ret)
	{
		return false;
	}

	Eigen::Map< Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mEstE(estE.E);
	Eigen::Map< Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> > mR(R);
	Eigen::Matrix<FloatType, 3, 3> mT;
	// skew symmetric matrix
	mT(0, 0) = mT(1, 1) = mT(2, 2) = 0;
	mT(1, 0) = t[2];
	mT(0, 1) = -mT(1, 0);
	mT(0, 2) = t[1];
	mT(2, 0) = -mT(0, 2);
	mT(2, 1) = t[0];
	mT(1, 2) = -mT(2, 1);
	Eigen::Matrix<FloatType, 3, 3> mTrueE = (mT*mR).normalized();
	if (std::signbit(mTrueE(0, 0)) != std::signbit(mEstE(0, 0)))
	{
		mEstE = -mEstE;
	}
	cout << "True E:" << endl;
	cout << mTrueE << endl;
	cout << "Found E:" << endl;
	cout << mEstE.normalized() << endl;
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
