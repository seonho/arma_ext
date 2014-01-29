/**
 *	@file		random.hpp
 *	@brief		Random number generation functions
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


#if defined(USE_CXX11) || defined(USE_BOOST)
#include "rand_impl.hpp"
#else
#include "rand_impl_arma.hpp"
#endif

#include "mpl.hpp"

#ifdef USE_PPL
#include <ppl.h>
#if	(_MSC_VER <= 1600)
/// From Microsoft Visual Studio 2012, Concurrency namespace has been changed to concurrency.
/// For compatibility, namespace alias is used
namespace concurrency = Concurrency;
#endif

#endif

namespace arma_ext
{
	using namespace arma;

#ifndef USE_CXX11
	namespace internal {
		template <typename T1, typename T2>
		struct sort_pair_by_second_descend {
			bool operator() (const std::pair<T1, T2>& a, const std::pair<T1, T2>& b) {
				return b.second > a.second;
			}
		};
	}
#endif

	//!	@addtogroup	rand
	//!	@{

	/**
	 *	@brief	Random permutation.
	 *	@param n number of values.
	 *	@return	A row vector containing a random permutation of the integer from \f$0\f$ to \f$(n - 1)\f$ inclusive.
	 */
	uvec randperm(size_type n)
	{
		arma::vec values = arma_ext::rand<arma::vec>(n);
		std::vector<std::pair<size_type, double> > pairs(n);
        
#if defined(USE_PPL)
		concurrency::parallel_for(size_type(0), n, [&](size_type i) {
#elif defined(USE_OPENMP)
	#pragma omp parallel for
		for (int si = 0 ; si < (int)n ; si++) {
			size_type i = (size_type)si;
#else
        for (size_type i = 0 ; i < n ; i++) {
#endif
			pairs[i] = std::make_pair(i, values(i));
#ifdef USE_PPL
		});
#else
        }
                                  
#endif

#ifdef USE_CXX11
		std::stable_sort(pairs.begin(), pairs.end(), [&](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b)->bool {
			return b.second > a.second;
		});
#else
		std::stable_sort(pairs.begin(), pairs.end(), internal::sort_pair_by_second_descend<size_t, double>());
#endif

		arma::uvec out(n);
//#ifdef USE_CXX11
//		auto itr = pairs.begin();
//        out.imbue([&]() { return (itr++)->first; });
//#else
        std::vector<std::pair<arma_ext::size_type, double> >::iterator itr = pairs.begin();
        for (uword i = 0 ; i < out.n_elem ; i++)
            out[i] = (itr++)->first;
//#endif
		
		return out;
	}

	/**
	 *	@brief	Random permutation with a given vector
	 *	@param in	The input vector.
	 *	@param k	The maximum number of elements of the vector to be returned.
	 *	@param junk	Reserved.
	 *	@return Permuted vector
	 */
	template <typename T>
	typename std::enable_if<std::or_<T::is_col, T::is_row>::value, T>::type randvalues(const T& in, uword k, const typename arma::enable_if< arma::is_arma_type<T>::value>::result* junk = 0)
	{
		T out;
		uword N = in.n_elem;

		if (k > N) k = N;

		if ((double)k / N < 0.0001) {
			uvec i1 = arma::unique( arma::conv_to<uvec>::from(arma::ceil(rand<vec>(k) * N)) );
			out = in.elem(i1);
		} else {
			uvec i2 = randperm(N);
			out = in.elem(sort(i2.subvec(0, k - 1)));
		}

		return out;
	}

	//!	@}
}
