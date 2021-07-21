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

#ifndef P3P_RANSAC_MODEL_HPP_
#define P3P_RANSAC_MODEL_HPP_

#include "P3P.hpp"

#include <limits>
#include <vector>

namespace P3P
{
template <typename FloatType>
struct RT
{
	// x = R*X + t
	FloatType R[9];
	FloatType t[3];
};

template <typename FloatType>
class RansacModel
{
public:
	RansacModel() :
		m_nSet(0)
		, m_epsilon(0.4f)
		, m_solver({})
		, m_threshold(0)
		, m_nSolutions(0)
	{
	}

	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}

	// p2d = [x1 y1 x2 y2 ... xn yn]
	// p3d = [X1 Y1 Z1 X2 Y2 Z2 ... Xn Yn Zn]
	bool setCorrespondences(const std::vector<FloatType> &p2d, const std::vector<FloatType> &p3d)
	{
		if ((p2d.empty()) || (p2d.size() % 2 != 0) ||
			(p3d.size() % 3 != 0) || ((p2d.size() % 2) != (p3d.size() % 3)))
		{
			return false;
		}
		m_p2d = p2d;
		m_p3d = p3d;
		m_nSet = p2d.size() / 2;

		return true;
	}

	void setThreshold(FloatType threshold)
	{
		m_threshold = threshold;
	}

protected:
	bool iteration(const std::vector<int> &selected, RT<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		if ((m_nSet < m_minSet) || (selected.size() < m_minSet))
		{
			return false;
		}

		bool ret = solveSelected(selected);
		if ((!ret) || (m_nSolutions == 0))
		{
			nInlier = 0;
			std::vector<int> z(m_nSet, 0);
			indices.swap(z);
			return true;
		}

		checkInlier(p, nInlier, indices);
		P3P<FloatType>::convertPose(p.R, p.t); // R'*(X - t) -> R*X + t

		return true;
	}

private:
	bool solveSelected(const std::vector<int> &selected)
	{
		FloatType p2dn[m_minSet * 3], p3d[m_minSet * 3];
		for (size_t k = 0; k < m_minSet; k++)
		{
			const size_t idx = 2 * selected[k];
			const size_t idx3 = 3 * selected[k];
			const size_t idx32 = 3 * k;
			// 2D point -> unitary feature vector
			FloatType s = 1 / std::sqrt((m_p2d[idx] * m_p2d[idx]) + (m_p2d[idx + 1] * m_p2d[idx + 1]) + 1);
			p2dn[idx32] = m_p2d[idx] * s;
			p2dn[idx32 + 1] = m_p2d[idx + 1] * s;
			p2dn[idx32 + 2] = s;
			// 3D point
			p3d[idx32] = m_p3d[idx3];
			p3d[idx32 + 1] = m_p3d[idx3 + 1];
			p3d[idx32 + 2] = m_p3d[idx3 + 2];
		}
		return m_solver.getMatrix(p2dn, p3d, m_nSolutions, m_Ps);
	}

	void checkInlier(RT<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		nInlier = 0;
		indices.resize(m_nSet);
		std::fill(indices.begin(), indices.end(), 0);

		FloatType averageError = std::numeric_limits<FloatType>::max();
		std::vector<int> candidateIndices(m_nSet);
		int candidateNInlier = 0;
		FloatType candidateAverageError = 0;
		Eigen::Map<const Eigen::Matrix<FloatType, 3, Eigen::Dynamic> > mX(&(m_p3d[0]), 3, m_nSet);
		Eigen::Map<const Eigen::Matrix<FloatType, 2, Eigen::Dynamic> > mx(&(m_p2d[0]), 2, m_nSet);
		for (int l = 0; l < m_nSolutions; l++)
		{
			FloatType *pt = &(m_Ps[(3 + 9) * l]);
			Eigen::Matrix<FloatType, 3, 1> mt(pt);
			Eigen::Matrix<FloatType, 3, 3, Eigen::RowMajor> mR(pt + 3);
			Eigen::Matrix<FloatType, 3, Eigen::Dynamic> mx2 = mR.transpose()*(mX.colwise() - mt);
			Eigen::Matrix<FloatType, 2, Eigen::Dynamic> mErr(2, m_nSet);
			mErr.row(0) = mx.row(0).array() - (mx2.row(0).array() / mx2.row(2).array());
			mErr.row(1) = mx.row(1).array() - (mx2.row(1).array() / mx2.row(2).array());
			Eigen::Matrix<FloatType, Eigen::Dynamic, 1> vErr = mErr.colwise().squaredNorm();

			candidateAverageError = 0;
			candidateNInlier = 0;
			std::fill(candidateIndices.begin(), candidateIndices.end(), 0);
			for (size_t k = 0; k < m_nSet; k++)
			{
				if (vErr[k] < m_threshold)
				{
					candidateNInlier++;
					candidateIndices[k] = 1;
					candidateAverageError += vErr[k];
				}
			}
			candidateAverageError /= candidateNInlier;

			if ((candidateNInlier > nInlier) ||
				((candidateNInlier == nInlier) && (candidateAverageError < averageError)))
			{
				averageError = candidateAverageError;
				nInlier = candidateNInlier;
				indices.swap(candidateIndices);
				p.t[0] = pt[0];
				p.t[1] = pt[1];
				p.t[2] = pt[2];
				for (int k = 0; k < 9; k++)
				{
					p.R[k] = pt[k + 3];
				}
			}
		}
	}

protected:
	static const size_t m_minSet; // minimum data number for parameter esimation
	static const bool m_acceptArbitraryNSet; // if true, this model accepts arbitrary data number for parameter esimation
	size_t m_nSet; // whole data number
	float m_epsilon; // propotion of outliers

private:
	P3P<FloatType> m_solver;
	FloatType m_threshold;
	std::vector<FloatType> m_p2d, m_p3d; // 3D-2D point correspondences
	int m_nSolutions; // number of Pose matrices
	std::vector<FloatType> m_Ps; // m_p3d matrices (row major)
};
template <typename FloatType>
const size_t RansacModel<FloatType>::m_minSet = 3;
template <typename FloatType>
const bool RansacModel<FloatType>::m_acceptArbitraryNSet = false;

}

#endif // P3P_RANSAC_MODEL_HPP_
