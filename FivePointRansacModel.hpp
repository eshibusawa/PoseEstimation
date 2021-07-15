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

#include "FivePointUtil.hpp"

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

template <typename FloatType, typename Model>
class RansacModel : public Model
{
public:
	RansacModel() :
		Model()
		, m_nSet(0)
		, m_epsilon(0.4f)
		, m_threshold(0)
		, m_nSolutions(0)
	{
	}

	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}

	bool setCorrespondences(const std::vector<FloatType> &pts1, const std::vector<FloatType> &pts2)
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
		if ((m_nSet < Model::m_minSet) || (selected.size() < Model::m_minSet))
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
		FloatType pts1[Model::m_minSet * 2], pts2[Model::m_minSet * 2];
		for (size_t k = 0; k < selected.size(); k++)
		{
			const size_t idx = 2 * selected[k];
			const size_t idx2 = 2 * k;
			pts1[idx2] = m_pts1[idx];
			pts1[idx2 + 1] = m_pts1[idx + 1];
			pts2[idx2] = m_pts2[idx];
			pts2[idx2 + 1] = m_pts2[idx + 1];
		}
		return Model::getMatrix(pts1, pts2, m_nSolutions, m_Es);
	}

	void checkInlier(E<FloatType> &p, int &nInlier, std::vector<int> &indices)
	{
		nInlier = 0;
		indices.resize(m_nSet);
		std::fill(indices.begin(), indices.end(), 0);

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
				FloatType err = getSampsonsError(E, &(m_pts1[2 * k]), &(m_pts2[2 * k]));
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
	size_t m_nSet; // whole data number
	float m_epsilon; // propotion of outliers

private:
	FloatType m_threshold;
	std::vector<FloatType> m_pts1, m_pts2; // 2D point correspondences
	int m_nSolutions; // number of Essential matrices
	std::vector<FloatType> m_Es; // Essential matrices (row major)
};

}

#endif // FIVE_POINT_RANSAC_MODEL_HPP_
