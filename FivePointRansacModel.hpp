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

#ifndef FIVE_POINT_RANSAC_MODEL_HPP_
#define FIVE_POINT_RANSAC_MODEL_HPP_

#include "FivePoint.hpp"

#include <iostream>
#include <limits>
#include <vector>

namespace FivePoint
{
template <typename FloatType>
struct E
{
	FloatType E[9];
};

template <typename FloatType>
class RansacModel
{
public:
	RansacModel() :
		m_nSet(0)
		, m_epsilon(0.4f)
		, m_solver()
		, m_threshold(0)
		, m_nSolutions(0)
	{
	}

	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}

	bool setCorrespondenses(const std::vector<FloatType> &pts1, const std::vector<FloatType> &pts2)
	{
		if ((pts1.empty()) || (pts1.size() % 2 != 0) || (pts1.size() != pts2.size()))
		{
			return false;
		}
		m_pts1 = pts1;
		m_pts2 = pts2;
		m_nSet = pts1.size() / 2;

		return true;
	}

	void setThreshold(FloatType threshold)
	{
		m_threshold = threshold;
	}

protected:
	bool iteration(const std::vector<int> &selected, E<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		if ((m_nSet < m_minSet) || (selected.size() < m_minSet))
		{
			return false;
		}

		bool ret = solveSelected(selected);
		if (!ret)
		{
			nInlier = 0;
			std::vector<int> z(m_nSet, 0);
			indices.swap(z);
			return true;
		}

		checkInlier(p, nInlier, indices);
		return true;
	}

private:
	bool solveSelected(const std::vector<int> &selected)
	{
		FloatType pts1[5*2], pts2[5*2];
		for (size_t k = 0; k < selected.size(); k++)
		{
			const size_t idx = 2 * selected[k];
			const size_t idx2 = 2 * k;
			pts1[idx2] = m_pts1[idx];
			pts1[idx2 + 1] = m_pts1[idx + 1];
			pts2[idx2] = m_pts2[idx];
			pts2[idx2 + 1] = m_pts2[idx + 1];
		}
		return m_solver.getEMatrix(pts1, pts2, m_nSolutions, m_Es);
	}

	void checkInlier(E<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		nInlier = 0;
		indices.resize(m_nSet);
		FloatType averageError = std::numeric_limits<FloatType>::max();
		std::vector<int> candidateIndices(m_nSet);
		int candidateNInlier = 0;
		FloatType candidateAverageError = 0;
		E<FloatType> candidateE;
		for (int l = 0; l < m_nSolutions; l++)
		{
			for (int i = 0; i < 9; i++)
			{
				candidateE.E[i] = m_Es[9*l + i];
			}
			const FloatType *E = candidateE.E;
			candidateAverageError = 0;
			candidateNInlier = 0;
			std::fill(candidateIndices.begin(), candidateIndices.end(), 0);
			for (size_t k = 0; k < m_nSet; k++)
			{
				FloatType *p1 = &(m_pts1[2 * k]);
				FloatType *p2 = &(m_pts2[2 * k]);
				// Sampson's error
				FloatType Ep10 = (E[0] * p1[0] + E[1] * p1[1] + E[2]);
				FloatType Ep11 = (E[3] * p1[0] + E[4] * p1[1] + E[5]);
				FloatType p2E0 = (p2[0] * E[0] + p2[1] * E[3] + E[6]);
				FloatType p2E1 = (p2[0] * E[1] + p2[1] * E[4] + E[7]);
				// p2 * E * p1
				FloatType p2Ep1 = (p2[0]) * Ep10 + (p2[1]) * Ep11 +
					(E[6] * p1[0] + E[7] * p1[1] + E[8]);

				FloatType err = p2Ep1 / std::sqrt(p2E0 * p2E0 + p2E1 * p2E1 + Ep10 * Ep10 + Ep11 * Ep11);
				err *= err;

				if (err < m_threshold)
				{
					candidateNInlier++;
					candidateIndices[k] = 1;
					candidateAverageError += err;
				}
			}
			candidateAverageError /= candidateNInlier;

			if ((candidateNInlier > nInlier) ||
				((candidateNInlier == nInlier) && (candidateAverageError < averageError)))
			{
				averageError = candidateAverageError;
				nInlier = candidateNInlier;
				indices.swap(candidateIndices);
				p = candidateE;
			}
		}
	}

protected:
	static const size_t m_minSet; // minimum data number for parameter esimation
	static const bool m_acceptArbitraryNSet; // if true, this model accepts arbitrary data number for parameter esimation
	size_t m_nSet; // whole data number
	float m_epsilon; // propotion of outliers

private:
	FivePoint<FloatType> m_solver;
	FloatType m_threshold;
	std::vector<FloatType> m_pts1, m_pts2; // 2D point correspondences
	int m_nSolutions; // number of Essential matrices
	std::vector<FloatType> m_Es; // Essential matrices (row major)
};

template <typename FloatType>
const size_t RansacModel<FloatType>::m_minSet = 5;
template <typename FloatType>
const bool RansacModel<FloatType>::m_acceptArbitraryNSet = false;
}

#endif // FIVE_POINT_RANSAC_MODEL_HPP_
