// This file is part of PoseEstimation.
// This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see 2-Clause BSD license below.
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

// Copyright (c) 2009, V. Lepetit, EPFL
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

#include "EPnP.hpp"
#include "EPnPRansacModel.hpp"
#include "Ransac.hpp"
#include "PoseUtil.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <iterator>

namespace
{
const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;

const int n = 10;
const double noise = 10;

const int ntest = 5000;
}

bool pose_estimation_test();
bool precition_time_test();
bool ransac_test();

int main(int argc, char** argv)
{
	std::srand(std::time(0));
	bool ret = false;

	// (1) pose estimation
	ret = pose_estimation_test();
	if (!ret)
	{
		return EXIT_FAILURE;
	}

	// (2) precision / time
	ret = precition_time_test();
	if (!ret)
	{
		return EXIT_FAILURE;
	}

	// (3) ransac
	ret = ransac_test();
	if (!ret)
	{
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

bool pose_estimation_test()
{
	using std::cout;
	using std::endl;
	cout << ">>>> Pose estimation test:" << endl;

	EPnP::epnp<double> PnP;

	PnP.set_internal_parameters(uc, vc, fu, fv);
	PnP.set_maximum_number_of_correspondences(n);

	double R_true[3][3], t_true[3];
	EPnP::random_pose(R_true, t_true);

	PnP.reset_correspondences();
	for(int i = 0; i < n; i++) {
		double Xw, Yw, Zw, u, v;

		EPnP::random_point(Xw, Yw, Zw);

		EPnP::project_with_noise(uc, vc, fu, fv, R_true, t_true, Xw, Yw, Zw, noise, u, v);
		PnP.add_correspondence(Xw, Yw, Zw, u, v);
	}

	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est);
	double rot_err, transl_err;

	PnP.relative_error(rot_err, transl_err, R_true, t_true, R_est, t_est);
	cout << ">>> Reprojection error: " << err2 << endl;
	cout << ">>> rot_err: " << rot_err << ", transl_err: " << transl_err << endl;
	cout << endl;
	cout << "'True reprojection error':"
		<< PnP.reprojection_error(R_true, t_true) << endl;
	cout << endl;
	cout << "True pose:" << endl;
	PnP.print_pose(R_true, t_true);
	cout << endl;
	cout << "Found pose:" << endl;
	PnP.print_pose(R_est, t_est);
	cout << endl;

	return true;
}

bool precition_time_test()
{
	using std::cout;
	using std::endl;
	cout << ">>>> Precision time test:" << endl;

	EPnP::epnp<double> PnP;
	EPnP::epnp<float> PnPF;

	PnP.set_internal_parameters(uc, vc, fu, fv);
	PnP.set_maximum_number_of_correspondences(n);
	PnPF.set_internal_parameters(uc, vc, fu, fv);
	PnPF.set_maximum_number_of_correspondences(n);

	double R_true[3][3], t_true[3];
	EPnP::random_pose(R_true, t_true);

	PnP.reset_correspondences();
	for(int i = 0; i < n; i++) {
		double Xw, Yw, Zw, u, v;

		EPnP::random_point(Xw, Yw, Zw);

		EPnP::project_with_noise(uc, vc, fu, fv, R_true, t_true, Xw, Yw, Zw, noise, u, v);
		PnP.add_correspondence(Xw, Yw, Zw, u, v);
		PnPF.add_correspondence(Xw, Yw, Zw, u, v);
	}

	double R_est[3][3], t_est[3];
	double err2 = 0;
	// (1) double precision
    auto begin = std::chrono::system_clock::now();
	for (int i = 0; i < ntest; i++)
	{
		err2 = PnP.compute_pose(R_est, t_est);
	}
    auto end = std::chrono::system_clock::now();
	double time_dp = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	time_dp /= ntest;

	// (2) single precision
	float R_estF[3][3], t_estF[3];
	float err2F = 0;
    begin = std::chrono::system_clock::now();
	for (int i = 0; i < ntest; i++)
	{
		err2F = PnPF.compute_pose(R_estF, t_estF);
	}
    end = std::chrono::system_clock::now();
	double time_sp = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	time_sp /= ntest;

	double rot_err, transl_err;
	PnP.relative_error(rot_err, transl_err, R_true, t_true, R_est, t_est);
	cout << ">>> Reprojection error (double): " << err2 << endl;
	cout << ">>> rot_err (double): " << rot_err << ", transl_err (double): " << transl_err << endl;
	cout << endl;

	double R_estFD[3][3], t_estFD[3];
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			R_estFD[i][j] = static_cast<double>(R_estF[i][j]);
		}
		t_estFD[i] = static_cast<double>(t_estF[i]);
	}
	PnP.relative_error(rot_err, transl_err, R_true, t_true, R_estFD, t_estFD);
	cout << ">>> Reprojection error (float): " << err2F << endl;
	cout << ">>> rot_err (float): " << rot_err << ", transl_err (float): " << transl_err << endl;
	cout << endl;

	cout << "True pose:" << endl;
	PnP.print_pose(R_true, t_true);
	cout << endl;
	cout << "Found pose: (double)" << endl;
	PnP.print_pose(R_est, t_est);
	cout << endl;
	cout << "Found pose: (float)" << endl;
	PnP.print_pose(R_estFD, t_estFD);

	cout << endl;
	cout << "Time: (double) " << time_dp << " [us]" << endl;
	cout << "Time: (float) " << time_sp << " [us]" << endl;
	cout << endl;

	return true;
}

bool ransac_test()
{
	typedef double FloatType;
	using std::cout;
	using std::endl;
	cout << ">>>> RANSAC test:" << endl;

	PoseEstimation::RANSAC<EPnP::RansacModel<FloatType>, EPnP::RT<FloatType> > PnP;
	const float epsilon = 0.4f;
	PnP.setEpsilon(epsilon);
	PnP.setRANSACParameter(300, 0.99f, true);

	const double outlier_noise = 100;
	const int nAll = 3 * n;
	const int nOutlier = static_cast<int>(nAll * epsilon);
	const int nInlier = nAll - nOutlier;

	FloatType R_true[3][3], t_true[3];
	EPnP::random_pose(R_true, t_true);
	PnP.setInternalParameters(uc, vc, fu, fv);
	std::vector<int> trueIndices;
	for(int i = 0; i < nAll; i++)
	{
		double Xw, Yw, Zw, u, v;
		EPnP::random_point(Xw, Yw, Zw);
		if (i < nInlier)
		{
			EPnP::project_with_noise(uc, vc, fu, fv, R_true, t_true, Xw, Yw, Zw, noise, u, v);
			trueIndices.push_back(1);
		}
		else
		{
			EPnP::project_with_noise(uc, vc, fu, fv, R_true, t_true, Xw, Yw, Zw, outlier_noise * noise, u, v);
			trueIndices.push_back(0);
		}

		PnP.addCorrespondence(Xw, Yw, Zw, u, v);
	}

	EPnP::RT<FloatType> estParam;
	std::vector<int> estIndices;
	int estNInlier = 0;
	PnP.setThreshold(std::sqrt(3 * noise * noise));
	bool ret = PnP.compute(estParam, estNInlier, estIndices);
	if (!ret)
	{
		return false;
	}

	cout << "True pose:" << endl;
	EPnP::epnp<FloatType>::print_pose(R_true, t_true);
	cout << endl;
	cout << "Found pose:" << endl;
	EPnP::epnp<FloatType>::print_pose(estParam.R, estParam.t);
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
