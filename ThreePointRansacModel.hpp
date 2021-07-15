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

#ifndef THREE_POINT_RANSAC_MODEL_HPP_
#define THREE_POINT_RANSAC_MODEL_HPP_

#include "ThreePoint.hpp"

#include <vector>

namespace ThreePoint
{
template <typename FloatType>
struct Sim3
{
	FloatType R[9];
	FloatType t[3];
	FloatType s;
};

template <typename FloatType>
class RansacModel
{
public:
	RansacModel() :
		m_nSet(0)
		, m_epsilon(0.4f)
		, m_threshold(0)
	{
	}

	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}

	bool setCorrespondences(const std::vector<FloatType> &pts1, const std::vector<FloatType> &pts2)
	{
		if ((pts1.empty()) || (pts1.size() % 3 != 0) || (pts1.size() != pts2.size()))
		{
			return false;
		}
		m_pts1 = pts1;
		m_pts2 = pts2;
		m_nSet = pts1.size() / 3;

		return true;
	}

	void setThreshold(FloatType threshold)
	{
		m_threshold = threshold;
	}

protected:
	bool iteration(const std::vector<int> &selected, Sim3<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		if ((m_nSet < m_minSet) || (selected.size() < m_minSet))
		{
			return false;
		}
		solveSelected(selected, p);

		checkInlier(p, nInlier, indices);
		return true;
	}

private:
	void solveSelected(const std::vector<int> &selected, Sim3<FloatType> &p)
	{
		std::vector<FloatType> pts1(selected.size() * 3), pts2(selected.size() * 3);
		for (size_t k = 0; k < selected.size(); k++)
		{
			const size_t idx = 3 * selected[k];
			const size_t idx2 = 3 * k;
			pts1[idx2] = m_pts1[idx];
			pts1[idx2 + 1] = m_pts1[idx + 1];
			pts1[idx2 + 2] = m_pts1[idx + 2];
			pts2[idx2] = m_pts2[idx];
			pts2[idx2 + 1] = m_pts2[idx + 1];
			pts2[idx2 + 2] = m_pts2[idx + 2];
		}
		ThreePoint<FloatType>::getRT(selected.size(), &(pts1[0]), &(pts2[0]), p.R, p.t, p.s);
	}

	void checkInlier(Sim3<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		nInlier = 0;
		indices.resize(m_nSet);
		std::fill(indices.begin(), indices.end(), 0);

		Eigen::Matrix<FloatType, 3, Eigen::Dynamic> mX2h(3, m_nSet);
		ThreePoint<FloatType>::transformPoints(m_nSet, p.s, p.R, p.t, &(m_pts1[0]), mX2h.data());
		Eigen::Map<const Eigen::Matrix<FloatType, 3, Eigen::Dynamic> > mX2(&(m_pts2[0]), 3, m_nSet);
		Eigen::Matrix<FloatType, 3, Eigen::Dynamic> mErr = mX2h - mX2;
		Eigen::Matrix<FloatType, Eigen::Dynamic, 1> vErr = mErr.colwise().squaredNorm();
		for (size_t k = 0; k < m_nSet; k++)
		{
			if (vErr[k] < m_threshold)
			{
				nInlier++;
				indices[k] = 1;
			}
		}
	}

protected:
	static const size_t m_minSet; // minimum data number for parameter esimation
	static const bool m_acceptArbitraryNSet; // if true, this model accepts arbitrary data number for parameter esimation
	size_t m_nSet; // whole data number
	float m_epsilon; // propotion of outliers

private:
	FloatType m_threshold;
	std::vector<FloatType> m_pts1, m_pts2; // 3D point correspondences
};
template <typename FloatType>
const size_t RansacModel<FloatType>::m_minSet = 3;
template <typename FloatType>
const bool RansacModel<FloatType>::m_acceptArbitraryNSet = true;

}

#endif // THREE_POINT_RANSAC_MODEL_HPP_
