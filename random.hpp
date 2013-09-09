/**
 *	@file		random.hpp
 *	@brief		An implemenation of randomize functions
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2007-2013, Seonho Oh
 *		All rights reserved. 
 * 
 *		Redistribution and use in source and binary forms, with or without  
 *		modification, are permitted provided that the following conditions are  
 *		met: 
 * 
 *		    * Redistributions of source code must retain the above copyright  
 *		    notice, this list of conditions and the following disclaimer. 
 *		    * Redistributions in binary form must reproduce the above copyright  
 *		    notice, this list of conditions and the following disclaimer in the  
 *		    documentation and/or other materials provided with the distribution. 
 *		    * Neither the name of the <ORGANIZATION> nor the names of its  
 *		    contributors may be used to endorse or promote products derived from  
 *		    this software without specific prior written permission. 
 * 
 *		THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS  
 *		IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  
 *		TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A  
 *		PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER  
 *		OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,  
 *		EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,  
 *		PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR  
 *		PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF  
 *		LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING  
 *		NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS  
 *		SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 */

#pragma once

#include <armadillo>
#include <random>

namespace arma_ext
{
	using namespace arma;

	std::mt19937 eng;

	/**
	 *	@brief	
	 *	@return	
	 */
	inline double rand()
	{
		static std::uniform_real_distribution<double> ur;
		double value = ur(eng);
		ur(eng);
		return value;
	}

	/**
	 *	@brief	Uniformly distributed pseudorandom numbers.
	 *	@param rows number of rows.
	 *	@param cols number of columns.
	 *	@return	A rows-by-cols matrix containing pseudorandom values drawn from the standard uniform distribution on the open interval (0, 1).
	 */
	template <typename mat_type>
	inline mat_type rand(const size_type rows, const size_type cols)
	{
		typedef mat_type::elem_type eT;

		mat_type out(rows, cols);
		//std::uniform_real_distribution<eT> ur;
		//out.imbue([&]()->eT {
			//eT value = ur(eng);
			//ur(eng);
			//return value;
			//return rand();
		//});
		out.imbue(&rand);
		
		return out;
	}

	/**
	 *	@brief	Overloaded function for rand.
	 *	@return A n-by-n matrix containing pseudorandom values drawn from the standard uniform distribution on the open interval (0, 1).
	 */
	template <typename mat_type>
	inline mat_type rand(const size_type n)
	{
		if (mat_type::is_col || mat_type::is_row) {
			mat_type out(n);
			out.imbue(&rand);
			return out;
		} else
			return arma_ext::rand<mat_type>(n, n);
	}

	template <typename mat_type>
	inline mat_type randn(const size_type rows, const size_type cols)
	{
		typedef mat_type::elem_type eT;
		static std::normal_distribution<eT> nr;

		mat_type out(rows, cols);

		out.imbue([&]()->eT {
			double value = nr(eng);
			nr(eng);
			return value;
		});

		return out;
	}

	/**
	 *	@brief	Random permutation.
	 *	@param n number of values.
	 *	@return	A row vector containing a random permutation of the integer from 0 to (n - 1) inclusive.
	 */
	uvec randperm(size_type n)
	{
		arma::vec values = arma_ext::rand<arma::vec>(n);
		std::vector<std::pair<size_type, double> > pairs(n);
		concurrency::parallel_for(size_type(0), n, [&](size_type i) {
			pairs[i] = std::make_pair(i, values(i));
		});

		std::stable_sort(pairs.begin(), pairs.end(), [&](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b)->bool {
			return b.second > a.second;
		});

		arma::uvec out(n);
		auto itr = pairs.begin();
		out.imbue([&]() { return (itr++)->first; });
		return out;
	}

	uvec randvalues(const uvec& in, uword k)
	{
		uvec out;

		uword N = in.n_elem;

		if (k > N) k = N;

		if ((double)k / N < 0.0001) {
			uvec i1 = conv_to<uvec>::from(arma::unique(arma::ceil(N * rand<vec>(k))));
			out = in.elem(i1);
		} else {
			uvec i2 = randperm(N);
			out = in.elem(sort(i2.subvec(0, k - 1)));
		}

		return out;
	}
}