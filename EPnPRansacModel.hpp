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

#ifndef EPNP_RANSAC_MODELHPP_
#define EPNP_RANSAC_MODELHPP_

#include <EPnP.hpp>

namespace EPnP
{
template <typename FloatType>
struct RT
{
	FloatType R[3][3];
	FloatType t[3];
};

template <typename FloatType>
class RansacModel
{
public:
	RansacModel() :
		m_solver()
		, m_nCorrespondences(0)
		, m_epsilon(0.4f)
		, m_threshold(0)
		, m_uc(0)
		, m_vc(0)
		, m_fu(0)
		, m_fv(0)
	{
	}

	void addCorrespondence(FloatType X, FloatType Y, FloatType Z, FloatType u, FloatType v)
	{
		m_pws.push_back(X);
		m_pws.push_back(Y);
		m_pws.push_back(Z);

		m_us.push_back(u);
		m_us.push_back(v);
		m_nCorrespondences++;
	}

	void resetCorrespondence()
	{
		m_nCorrespondences = 0;
		m_us.resize(0);
		m_pws.resize(0);
	}

	void setInternalParameters(FloatType uc, FloatType vc, FloatType fu, FloatType fv)
	{
		m_uc = uc;
		m_vc = vc;
		m_fu = fu;
		m_fv = fv;
		m_solver.set_internal_parameters(uc, vc, fu, fv);
	}

	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}

	void setThreshold(FloatType threshold)
	{
		m_threshold = threshold;
	}

protected:
	void getModelParameters(int &minSet, int &nSet, float &epsilon)
	{
		minSet = 4;
		nSet = m_nCorrespondences;
		epsilon = m_epsilon;
	}

	bool iteration(const std::vector<int> &selected, RT<FloatType> &p, int &nInliar, std::vector<int> &indices)
	{
		const size_t minSet = 4;
		if ((m_nCorrespondences < minSet) || (selected.size() < minSet))
		{
			return false;
		}

		m_solver.set_maximum_number_of_correspondences(m_nCorrespondences);
		double err2 = solveSelected(selected, p);
		checkInliar(p, nInliar, indices);
		return true;
	}

private:
	double solveSelected(const std::vector<int> &selected, RT<FloatType> &p)
	{
		m_solver.reset_correspondences();
		for (size_t k = 0; k < selected.size(); k++)
		{
			const size_t ius = 2*k;
			const size_t ipws = 3*k;
			m_solver.add_correspondence(m_pws[ipws], m_pws[ipws + 1], m_pws[ipws + 2], m_us[ius], m_us[ius + 1]);
		}
		return m_solver.compute_pose(p.R, p.t);
	}

	void checkInliar(const RT<FloatType> &p, int &nInliar, std::vector<int> &indices)
	{
		nInliar = 0;
		indices.resize(m_nCorrespondences);
		for (size_t k = 0; k < m_nCorrespondences; k++)
		{
			const size_t ius = 2*k;
			const size_t ipws = 3*k;
			const FloatType u = m_us[ius], v = m_us[ius + 1];
			FloatType Xc = (p.R[0][0] * m_pws[ipws]) + (p.R[0][1] * m_pws[ipws + 1]) + (p.R[0][2] * m_pws[ipws + 2]) + p.t[0];
			FloatType Yc = (p.R[1][0] * m_pws[ipws]) + (p.R[1][1] * m_pws[ipws + 1]) + (p.R[1][2] * m_pws[ipws + 2]) + p.t[1];
			FloatType invZc = 1/((p.R[2][0] * m_pws[ipws]) + (p.R[2][1] * m_pws[ipws + 1]) + (p.R[2][2] * m_pws[ipws + 2]) + p.t[2]);
			FloatType ue = m_fu * Xc * invZc + m_uc;
			FloatType ve = m_fv * Yc * invZc + m_vc;

			FloatType err2 = std::sqrt( (ue - u)*(ue - u) + (ve - v)*(ve - v));
			if (err2 < m_threshold)
			{
				indices[k] = 1;
				nInliar++;
			}
			else
			{
				indices[k] = 0;
			}
		}
	}

	epnp<FloatType> m_solver;
	size_t m_nCorrespondences;
	float m_epsilon; // propotion of outliers
	FloatType m_threshold; // threshold of reprojection error
	FloatType m_uc; // principal point uc [px]
	FloatType m_vc; // principal point vc [px]
	FloatType m_fu; // focal length fu [px]
	FloatType m_fv; // focal length fv [px]
	std::vector<FloatType> m_us; // 2D points
	std::vector<FloatType> m_pws; // 3D points
};
}

#endif // EPNP_RANSAC_MODELHPP_
