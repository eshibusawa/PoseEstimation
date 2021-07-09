// This file is part of PoseEstimation.
// Copyright (c) 2021, Eijiro Shibusawa
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

#ifndef RANSAC_HPP_
#define RANSAC_HPP_

#include <random>
#include <vector>

namespace PoseEstimation
{

template <typename Model, typename Parameters>
class RANSAC : public Model
{
public:
	RANSAC() :
		Model()
		, m_rng()
		, m_maxIterations(300)
		, m_probability(0.99f)
		, m_isRefineNeeded(false)
	{
	}

	virtual ~RANSAC()
	{
	}

	bool compute(Parameters &p, int &nInlier, std::vector<int> &indices)
	{
		// setup (0)
		bool ret = false;
		float epsilon = 0; // propotion of outliers
		int minSet = 0; // minimum data number for parameter esimation
		int nSet = 0; // whole data number
		// (0.1)
		Model::getModelParameters(minSet, nSet, epsilon);
		if ((minSet <= 0) || (nSet <=0) || (nSet <= minSet))
		{
			return false;
		}
		if ((epsilon >= 1) || (epsilon <= 0))
		{
			return false;
		}

		// (0.2) determine maximum iteration count
		int iterations = std::ceil(std::log(1 - m_probability) / std::log(1 - std::pow(epsilon, minSet)));
		iterations = std::max(1, std::min(iterations, m_maxIterations));

		// (0.3)
		int bestNInliers = 0, candidateNInlier = 0;
		Parameters bestParameters, candidateParameters;
		std::vector<int> bestIndices, candidateIndices, selected;

		// (1) estimation
		selected.reserve(static_cast<int>(nSet * (1.0f - epsilon)));
		selected.resize(minSet);
		std::uniform_int_distribution<int> uid(0, nSet - 1);
		for (int k = 0; k < iterations; k++)
		{
			for (int l = 0; l < minSet; l++)
			{
				selected[l] = uid(m_rng);
			}
			ret = Model::iteration(selected, candidateParameters, candidateNInlier, candidateIndices);
			if (!ret)
			{
				return false;
			}
			if (bestNInliers < candidateNInlier)
			{
				bestNInliers = candidateNInlier;
				bestParameters = candidateParameters;
				bestIndices.swap(candidateIndices);
			}
		}

		if (!m_isRefineNeeded)
		{
			return true;
		}

		// (2) refinement
		selected.resize(0);
		for (size_t l = 0; l < bestIndices.size(); l++)
		{
			if (bestIndices[l])
			{
				selected.push_back(l);
			}
		}
		ret = Model::iteration(selected, bestParameters, candidateNInlier, candidateIndices);
		if (!ret)
		{
			return false;
		}
		p = bestParameters;
		nInlier = candidateNInlier;
		indices.swap(candidateIndices);

		return true;
	}

	void setRANSACParameter(int maxIterations, float probability, bool isRefineNeeded)
	{
		m_maxIterations = maxIterations;
		m_probability = probability;
		m_isRefineNeeded = isRefineNeeded;
	}

	void setRNG(uint32_t seed)
	{
		m_rng = std::mt19937(seed);
	}

private:
	std::mt19937 m_rng;
	int m_maxIterations;
	float m_probability;
	bool m_isRefineNeeded;
};

}

#endif // RANSAC_HPP_