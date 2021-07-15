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

#ifndef EPNP_HPP_
#define EPNP_HPP_

#include <Eigen/Dense>

#include <iostream>
#include <vector>

namespace EPnP
{
template <typename FloatType>
class epnp {
private:
	typedef Eigen::Matrix<FloatType, 1, 10> Matrix_1x10;
	typedef Eigen::Matrix<FloatType, 2, 12> Matrix_2x12;
	typedef Eigen::Matrix<FloatType, 3, 3> Matrix_3x3;
	typedef Eigen::Matrix<FloatType, 3, 1> Matrix_3x1;
	typedef Eigen::Matrix<FloatType, 4, 1> Matrix_4x1;
	typedef Eigen::Matrix<FloatType, 5, 1> Matrix_5x1;
	typedef Eigen::Matrix<FloatType, 6, 1> Matrix_6x1;
	typedef Eigen::Matrix<FloatType, 6, 3> Matrix_6x3;
	typedef Eigen::Matrix<FloatType, 6, 4> Matrix_6x4;
	typedef Eigen::Matrix<FloatType, 6, 5> Matrix_6x5;
	typedef Eigen::Matrix<FloatType, 6, 10> Matrix_6x10;
	typedef Eigen::Matrix<FloatType, 12, 12> Matrix_12x12;
	typedef Eigen::Matrix<FloatType, 12, 1> Matrix_12x1;

public:
	epnp() : number_of_correspondences(0)
		, uc(0)
		, vc(0)
		, fu(0)
		, fv(0)
		, cws_determinant(0)
	{
	}
	virtual ~epnp()
	{
	}

	void set_internal_parameters(FloatType uc, FloatType vc, FloatType fu, FloatType fv)
	{
		this->uc = uc;
		this->vc = vc;
		this->fu = fu;
		this->fv = fv;
	}

	void set_maximum_number_of_correspondences(int n)
	{
		pws.resize(0);
		us.resize(0);
		alphas.resize(0);
		pcs.resize(0);
		pws.reserve(3*n);
		us.reserve(2*n);
		alphas.reserve(4*n);
		pcs.reserve(3*n);
	}

	void reset_correspondences(void)
	{
		pws.resize(0);
		us.resize(0);
		alphas.resize(0);
		pcs.resize(0);
		number_of_correspondences = 0;
	}

	void add_correspondence(FloatType X, FloatType Y, FloatType Z, FloatType u, FloatType v)
	{
		pws.push_back(X);
		pws.push_back(Y);
		pws.push_back(Z);

		us.push_back(u);
		us.push_back(v);
		number_of_correspondences++;
	}

	FloatType compute_pose(FloatType R[3][3], FloatType t[3])
	{
		choose_control_points();
		compute_barycentric_coordinates();

		Matrix_2x12 mM;
		Matrix_12x12 mMtM;
		mMtM.setZero();
		for(int i = 0; i < number_of_correspondences; i++) {
			const FloatType *as = &(alphas[4*i]);
			const FloatType u = us[2 * i], v = us[2 * i + 1];
			for(int j = 0; j < 4; j++) {
				mM(0, 3 * j) = as[j] * fu;
				mM(0, 3 * j + 1) = 0;
				mM(0, 3 * j + 2) = as[j] * (uc - u);

				mM(1, 3 * j    ) = 0;
				mM(1, 3 * j + 1) = as[j] * fv;
				mM(1, 3 * j + 2) = as[j] * (vc - v);
			}
			mMtM += (mM.transpose() * mM);
		}

		Eigen::JacobiSVD<Matrix_12x12> svd(mMtM, Eigen::ComputeFullV);
		Matrix_12x12 mV = svd.matrixV();
		Matrix_6x10 mL_6x10;
		Matrix_6x1 mRho;
		compute_L_6x10(mV, mL_6x10);
		compute_rho(mRho);

		FloatType Betas[4][4], rep_errors[4];
		FloatType Rs[4][3][3], ts[4][3];

		find_betas_approx_1(mL_6x10, mRho, Betas[1]);
		gauss_newton(mL_6x10, mRho, Betas[1]);
		rep_errors[1] = compute_R_and_t(mV, Betas[1], Rs[1], ts[1]);

		find_betas_approx_2(mL_6x10, mRho, Betas[2]);
		gauss_newton(mL_6x10, mRho, Betas[2]);
		rep_errors[2] = compute_R_and_t(mV, Betas[2], Rs[2], ts[2]);

		find_betas_approx_3(mL_6x10, mRho, Betas[3]);
		gauss_newton(mL_6x10, mRho, Betas[3]);
		rep_errors[3] = compute_R_and_t(mV, Betas[3], Rs[3], ts[3]);

		int N = 1;
		if (rep_errors[2] < rep_errors[1]) N = 2;
		if (rep_errors[3] < rep_errors[N]) N = 3;

		copy_R_and_t(Rs[N], ts[N], R, t);

		return rep_errors[N];
	}

	FloatType reprojection_error(const FloatType R[3][3], const FloatType t[3]) const
	{
		FloatType sum2 = 0;

		for(int i = 0; i < number_of_correspondences; i++) {
			const FloatType *pw = &(pws[3 * i]);
			FloatType Xc = dot(R[0], pw) + t[0];
			FloatType Yc = dot(R[1], pw) + t[1];
			FloatType inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
			FloatType ue = uc + fu * Xc * inv_Zc;
			FloatType ve = vc + fv * Yc * inv_Zc;
			FloatType u = us[2 * i], v = us[2 * i + 1];

			sum2 += std::sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
		}

		return sum2 / number_of_correspondences;
	}

public:
	static void copy_R_and_t(const FloatType R_src[3][3], const FloatType t_src[3], FloatType R_dst[3][3], FloatType t_dst[3])
	{
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++)
				R_dst[i][j] = R_src[i][j];
			t_dst[i] = t_src[i];
		}
	}

	static void print_pose(const FloatType R[3][3], const FloatType t[3])
	{
		std::cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << std::endl;
		std::cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << std::endl;
		std::cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << std::endl;
	}

	static void relative_error(FloatType & rot_err, FloatType & transl_err,
			const FloatType Rtrue[3][3], const FloatType ttrue[3],
			const FloatType Rest[3][3],  const FloatType test[3])
	{
		FloatType qtrue[4], qest[4];

		mat_to_quat(Rtrue, qtrue);
		mat_to_quat(Rest, qest);

		FloatType rot_err1 = std::sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
					(qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
					(qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
					(qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
			std::sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

		FloatType rot_err2 = std::sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
					(qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
					(qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
					(qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
			std::sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

		rot_err = std::min(rot_err1, rot_err2);

		transl_err =
			std::sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
			(ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
			(ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
			std::sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
	}

private:
	void choose_control_points()
	{
		// Take C0 as the reference points centroid:
		cws[0][0] = cws[0][1] = cws[0][2] = 0;
		for(int i = 0; i < number_of_correspondences; i++)
			for(int j = 0; j < 3; j++)
				cws[0][j] += pws[3 * i + j];

		for(int j = 0; j < 3; j++)
			cws[0][j] /= number_of_correspondences;

		const FloatType *ppws = &(pws[0]);
		FloatType q[3];
		Matrix_3x3 mPW0tPW0;
		mPW0tPW0.setZero();
		for(int i = 0; i < number_of_correspondences; i++) {
			q[0] = ppws[0] - cws[0][0];
			q[1] = ppws[1] - cws[0][1];
			q[2] = ppws[2] - cws[0][2];
			mPW0tPW0(0, 0) += (q[0] * q[0]);
			mPW0tPW0(0, 1) += (q[0] * q[1]);
			mPW0tPW0(0, 2) += (q[0] * q[2]);
			mPW0tPW0(1, 1) += (q[1] * q[1]);
			mPW0tPW0(1, 2) += (q[1] * q[2]);
			mPW0tPW0(2, 2) += (q[2] * q[2]);
			ppws += 3;
		}
		mPW0tPW0(1, 0) = mPW0tPW0(0, 1);
		mPW0tPW0(2, 0) = mPW0tPW0(0, 2);
		mPW0tPW0(2, 1) = mPW0tPW0(1, 2);

		Eigen::JacobiSVD<Matrix_3x3> svd(mPW0tPW0, Eigen::ComputeFullV);
		Matrix_3x1 DC = svd.singularValues();
		Matrix_3x3 VC = svd.matrixV();

		for(int i = 1; i < 4; i++) {
			FloatType k = std::sqrt(DC[i - 1] / number_of_correspondences);
			for(int j = 0; j < 3; j++)
				cws[i][j] = cws[0][j] + k * VC(j, i - 1);
		}
	}

	void compute_barycentric_coordinates()
	{
		Matrix_3x3 mCC;
		for(int i = 0; i < 3; i++)
			for(int j = 1; j < 4; j++)
				mCC(i, j - 1) = cws[j][i] - cws[0][i];

		Eigen::JacobiSVD<Matrix_3x3> svd(mCC, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Matrix_3x1 ma;
		Matrix_3x1 mb;

		for(int i = 0; i < number_of_correspondences; i++) {
			FloatType *pi = &(pws[3 * i]);
			FloatType *a = &(alphas[4 * i]);
			mb[0] = (pi[0] - cws[0][0]);
			mb[1] = (pi[1] - cws[0][1]);
			mb[2] = (pi[2] - cws[0][2]);
			ma = svd.solve(mb);
			a[1] = ma[0];
			a[2] = ma[1];
			a[3] = ma[2];
			a[0] = 1 - a[1] - a[2] - a[3];
		}
	}

	void compute_ccs(const FloatType * betas, const Matrix_12x12 &mV)
	{
		for(int i = 0; i < 4; i++)
			ccs[i][0] = ccs[i][1] = ccs[i][2] = 0;

		for(int i = 0; i < 4; i++) {
			for(int j = 0; j < 4; j++)
				for(int k = 0; k < 3; k++)
					ccs[j][k] += betas[i] * mV(3 * j + k, 11 - i);
		}
	}

	void compute_pcs()
	{
		for(int i = 0; i < number_of_correspondences; i++) {
			FloatType *a = &(alphas[4 * i]);
			FloatType *pc = &(pcs[3 * i]);

			for(int j = 0; j < 3; j++)
				pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
		}
	}

	void solve_for_sign()
	{
		if (pcs[2] < 0) {
			for(int i = 0; i < 4; i++)
				for(int j = 0; j < 3; j++)
					ccs[i][j] = -ccs[i][j];

			for(int i = 0; i < number_of_correspondences; i++) {
				pcs[3 * i    ] = -pcs[3 * i];
				pcs[3 * i + 1] = -pcs[3 * i + 1];
				pcs[3 * i + 2] = -pcs[3 * i + 2];
			}
		}
	}

	void compute_rho(Matrix_6x1 &mRho) const
	{
		mRho[0] = dist2(cws[0], cws[1]);
		mRho[1] = dist2(cws[0], cws[2]);
		mRho[2] = dist2(cws[0], cws[3]);
		mRho[3] = dist2(cws[1], cws[2]);
		mRho[4] = dist2(cws[1], cws[3]);
		mRho[5] = dist2(cws[2], cws[3]);
	}

	FloatType compute_R_and_t(const Matrix_12x12 &mV, const FloatType * betas, FloatType R[3][3], FloatType t[3])
	{
		compute_ccs(betas, mV);
		compute_pcs();

		solve_for_sign();

		estimate_R_and_t(R, t);

		return reprojection_error(R, t);
	}

	void estimate_R_and_t(FloatType R[3][3], FloatType t[3]) const
	{
		FloatType pc0[3] = {}, pw0[3] = {};

		for(int i = 0; i < number_of_correspondences; i++) {
			const FloatType *pc = &(pcs[3 * i]);
			const FloatType *pw = &(pws[3 * i]);

			for(int j = 0; j < 3; j++) {
				pc0[j] += pc[j];
				pw0[j] += pw[j];
			}
		}
		for(int j = 0; j < 3; j++) {
			pc0[j] /= number_of_correspondences;
			pw0[j] /= number_of_correspondences;
		}

		Matrix_3x3 mABt;
		mABt.setZero();
		for(int i = 0; i < number_of_correspondences; i++) {
			const FloatType *pc = &(pcs[3 * i]);
			const FloatType *pw = &(pws[3 * i]);

			for(int j = 0; j < 3; j++) {
				mABt(j, 0) += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
				mABt(j, 1) += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
				mABt(j, 2) += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
			}
		}

		Eigen::JacobiSVD<Matrix_3x3> svd(mABt, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Matrix_3x3 mABt_U = svd.matrixU();
		Matrix_3x3 mABt_V = svd.matrixV();
		Matrix_3x3 mR = mABt_U * (mABt_V).transpose();
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				R[i][j] = mR(i, j);

		const FloatType det =
			R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
			R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

		if (det < 0) {
			R[2][0] = -R[2][0];
			R[2][1] = -R[2][1];
			R[2][2] = -R[2][2];
		}

		t[0] = pc0[0] - dot(R[0], pw0);
		t[1] = pc0[1] - dot(R[1], pw0);
		t[2] = pc0[2] - dot(R[2], pw0);
	}

private:
	static inline FloatType dist2(const FloatType *p1, const FloatType *p2)
	{
		return
			(p1[0] - p2[0]) * (p1[0] - p2[0]) +
			(p1[1] - p2[1]) * (p1[1] - p2[1]) +
			(p1[2] - p2[2]) * (p1[2] - p2[2]);
	}

	static inline  FloatType dot(const FloatType *v1, const FloatType *v2)
	{
		return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
	}

	static void mat_to_quat(const FloatType R[3][3], FloatType q[4])
	{
		FloatType tr = R[0][0] + R[1][1] + R[2][2];
		FloatType n4;

		if (tr > 0) {
			q[0] = R[1][2] - R[2][1];
			q[1] = R[2][0] - R[0][2];
			q[2] = R[0][1] - R[1][0];
			q[3] = tr + 1;
			n4 = q[3];
		} else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
			q[0] = 1 + R[0][0] - R[1][1] - R[2][2];
			q[1] = R[1][0] + R[0][1];
			q[2] = R[2][0] + R[0][2];
			q[3] = R[1][2] - R[2][1];
			n4 = q[0];
		} else if (R[1][1] > R[2][2]) {
			q[0] = R[1][0] + R[0][1];
			q[1] = 1 + R[1][1] - R[0][0] - R[2][2];
			q[2] = R[2][1] + R[1][2];
			q[3] = R[2][0] - R[0][2];
			n4 = q[1];
		} else {
			q[0] = R[2][0] + R[0][2];
			q[1] = R[2][1] + R[1][2];
			q[2] = 1 + R[2][2] - R[0][0] - R[1][1];
			q[3] = R[0][1] - R[1][0];
			n4 = q[2];
		}
		FloatType scale = 1 / (2 * std::sqrt(n4));

		q[0] *= scale;
		q[1] *= scale;
		q[2] *= scale;
		q[3] *= scale;
	}

	static void compute_L_6x10(const Matrix_12x12 &mV, Matrix_6x10 &mL_6x10)
	{
		const int r[] = {11, 10, 9, 8};
		FloatType dv[4][6][3];
		for(int i = 0; i < 4; i++) {
			int a = 0, b = 1;
			for(int j = 0; j < 6; j++) {
				dv[i][j][0] = mV(3 * a, r[i]) - mV(3 * b, r[i]);
				dv[i][j][1] = mV(3 * a + 1, r[i]) - mV(3 * b + 1, r[i]);
				dv[i][j][2] = mV(3 * a + 2, r[i]) - mV(3 * b + 2, r[i]);

				b++;
				if (b > 3) {
					a++;
					b = a + 1;
				}
			}
		}

		for(int i = 0; i < 6; i++) {
			mL_6x10(i, 0) =     dot(dv[0][i], dv[0][i]);
			mL_6x10(i, 1) = 2 * dot(dv[0][i], dv[1][i]);
			mL_6x10(i, 2) =     dot(dv[1][i], dv[1][i]);
			mL_6x10(i, 3) = 2 * dot(dv[0][i], dv[2][i]);
			mL_6x10(i, 4) = 2 * dot(dv[1][i], dv[2][i]);
			mL_6x10(i, 5) =     dot(dv[2][i], dv[2][i]);
			mL_6x10(i, 6) = 2 * dot(dv[0][i], dv[3][i]);
			mL_6x10(i, 7) = 2 * dot(dv[1][i], dv[3][i]);
			mL_6x10(i, 8) = 2 * dot(dv[2][i], dv[3][i]);
			mL_6x10(i, 9) =     dot(dv[3][i], dv[3][i]);
		}
	}

	static void find_betas_approx_1(const Matrix_6x10 &mL_6x10, const Matrix_6x1 &mRho, FloatType *betas)
	{
		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_1 = [B11 B12     B13         B14]
		Matrix_6x4 mL_6x4;
		Matrix_4x1 mB4;

		mL_6x4.col(0) =  mL_6x10.col(0);
		mL_6x4.col(1) =  mL_6x10.col(1);
		mL_6x4.col(2) =  mL_6x10.col(3);
		mL_6x4.col(3) =  mL_6x10.col(6);

		mB4 = mL_6x4.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(mRho);

		if (mB4[0] < 0) {
			betas[0] = std::sqrt(-mB4[0]);
			betas[1] = -mB4[1] / betas[0];
			betas[2] = -mB4[2] / betas[0];
			betas[3] = -mB4[3] / betas[0];
		} else {
			betas[0] = std::sqrt(mB4[0]);
			betas[1] = mB4[1] / betas[0];
			betas[2] = mB4[2] / betas[0];
			betas[3] = mB4[3] / betas[0];
		}
	}

	static void find_betas_approx_2(const Matrix_6x10 &mL_6x10, const Matrix_6x1 &mRho, FloatType *betas)
	{
		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_2 = [B11 B12 B22                            ]
		Matrix_6x3 mL_6x3;
		Matrix_3x1 mB3;

		mL_6x3 = mL_6x10.leftCols(3);

		mB3 = mL_6x3.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(mRho);

		if (mB3[0] < 0) {
			betas[0] = std::sqrt(-mB3[0]);
			betas[1] = (mB3[2] < 0) ? std::sqrt(-mB3[2]) : 0;
		} else {
			betas[0] = std::sqrt(mB3[0]);
			betas[1] = (mB3[2] > 0) ? std::sqrt(mB3[2]) : 0;
		}

		if (mB3[1] < 0)
			betas[0] = -betas[0];
		betas[2] = 0;
		betas[3] = 0;
	}

	static void find_betas_approx_3(const Matrix_6x10 &mL_6x10, const Matrix_6x1 &mRho, FloatType *betas)
	{
		// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
		// betas_approx_3 = [B11 B12 B22 B13 B23                    ]
		Matrix_6x5 mL_6x5;
		Matrix_5x1 mB5;

		mL_6x5 =  mL_6x10.leftCols(5);
		mB5 = mL_6x5.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(mRho);

		if (mB5[0] < 0) {
			betas[0] = std::sqrt(-mB5[0]);
			betas[1] = (mB5[2] < 0) ? std::sqrt(-mB5[2]) : 0;
		} else {
			betas[0] = std::sqrt(mB5[0]);
			betas[1] = (mB5[2] > 0) ? std::sqrt(mB5[2]) : 0;
		}
		if (mB5[1] < 0)
			betas[0] = -betas[0];
		betas[2] = mB5[3] / betas[0];
		betas[3] = 0;
	}

	static void compute_A_and_b_gauss_newton(const Matrix_6x10 &mL_6x10, const Matrix_6x1 &mRho, const FloatType *betas, Matrix_6x4 &mA, Matrix_6x1 &mB)
	{
		for(int i = 0; i < 6; i++) {
			mA(i, 0) = 2 * mL_6x10(i, 0) * betas[0] +     mL_6x10(i, 1) * betas[1] +     mL_6x10(i, 3) * betas[2] +     mL_6x10(i, 6) * betas[3];
			mA(i, 1) =     mL_6x10(i, 1) * betas[0] + 2 * mL_6x10(i, 2) * betas[1] +     mL_6x10(i, 4) * betas[2] +     mL_6x10(i, 7) * betas[3];
			mA(i, 2) =     mL_6x10(i, 3) * betas[0] +     mL_6x10(i, 4) * betas[1] + 2 * mL_6x10(i, 5) * betas[2] +     mL_6x10(i, 8) * betas[3];
			mA(i, 3) =     mL_6x10(i, 6) * betas[0] +     mL_6x10(i, 7) * betas[1] +     mL_6x10(i, 8) * betas[2] + 2 * mL_6x10(i, 9) * betas[3];

			mB[i] = mRho[i] -
			(
			mL_6x10(i, 0) * betas[0] * betas[0] +
			mL_6x10(i, 1) * betas[0] * betas[1] +
			mL_6x10(i, 2) * betas[1] * betas[1] +
			mL_6x10(i, 3) * betas[0] * betas[2] +
			mL_6x10(i, 4) * betas[1] * betas[2] +
			mL_6x10(i, 5) * betas[2] * betas[2] +
			mL_6x10(i, 6) * betas[0] * betas[3] +
			mL_6x10(i, 7) * betas[1] * betas[3] +
			mL_6x10(i, 8) * betas[2] * betas[3] +
			mL_6x10(i, 9) * betas[3] * betas[3]
			);
		}
	}

	static void gauss_newton(const Matrix_6x10 &mL_6x10, const Matrix_6x1 &mRho, FloatType *betas)
	{
		const int iterations_number = 5;
		Matrix_6x4 mA;
		Matrix_6x1 mB;
		Matrix_4x1 mX;

		for(int k = 0; k < iterations_number; k++) {
			compute_A_and_b_gauss_newton(mL_6x10, mRho, betas, mA, mB);
			mX = mA.colPivHouseholderQr().solve(mB);

			for(int i = 0; i < 4; i++)
				betas[i] += mX[i];
		}
	}

private:
	int number_of_correspondences;
	FloatType uc, vc, fu, fv;
	FloatType cws_determinant;
	FloatType cws[4][3], ccs[4][3];
	std::vector<FloatType> pws, us, alphas, pcs;
};

}
#endif // EPNP_HPP_
