/**
 *	@file		random.hpp
 *	@brief		Random number generation functions
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2013-2015, Seonho Oh
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

#ifdef ARMA_EXT_USE_CPP11
#include <random>
#else
#pragma message "Compiling " __FILE__ " with armadillo randomizer functions."
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

#ifndef ARMA_EXT_USE_CPP11
	namespace internal
    {
		template <typename T1, typename T2>
		struct sort_pair_by_second_descend
        {
			inline bool operator() (const std::pair<T1, T2>& a, const std::pair<T1, T2>& b)
            {
				return b.second > a.second;
			}
		};
	}
#endif

	//!	@addtogroup	rand
	//!	@{

#ifdef ARMA_EXT_USE_CPP11
	std::mt19937 eng; ///< Mersenne twister engine.
#endif

	/**
	 *	@brief	Uniformly distributed pseudorandom number.
	 *	@return	A pseudorandom value drawn from the standard uniform distribution on the open interval \f$(0, 1)\f$.
	 */
	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type rand()
	{
#ifdef ARMA_EXT_USE_CPP11
		static std::uniform_real_distribution<T> ur;
		T value = ur(eng);
		ur(eng); // skip one time
		return value;
#else
		return arma::randu<T>();
#endif
	}

	/**
	 *	@brief	Uniformly distributed pseudorandom numbers.
	 *	@param rows the number of rows.
	 *	@param cols the number of columns.
	 *	@return	A rows-by-cols matrix containing pseudorandom values drawn from the standard uniform distribution on the open interval \f$(0, 1)\f$.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type rand(const size_type rows, const size_type cols)
	{
#ifdef ARMA_EXT_USE_CPP11
		T out(rows, cols);
        out.imbue(&rand<typename T::elem_type>);
		return out;
#else
		return randu<T>(rows, cols);
#endif
	}
	
	/**
	 *	@brief	Overloaded function for rand.
	 *	@return A \f$n\f$-by-\f$n\f$ matrix containing pseudorandom values drawn from the standard uniform distribution on the open interval \f$(0, 1)\f$.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type rand(const size_type n)
	{
		if (T::is_col)
			return arma_ext::rand<T>(n, 1);
		else if (T::is_row)
			return arma_ext::rand<T>(1, n);
		else
			return arma_ext::rand<T>(n, n);
	}
	
	/**
	 *	@brief	Normally distributed pseudorandom numbers.
	 *	@return	A pseudorandom value drawn from the standard normal distribution.
	 */
	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type randn()
	{
#ifdef ARMA_EXT_USE_CPP11
		static std::normal_distribution<T> nr;
		T value = nr(eng);
		nr(eng);
		return value;
#else
		return arma::randn<T>();
#endif
	}

	/**
	 *	@brief	Normally distributed pseudorandom numbers.
	 *	@param rows the number of rows.
	 *	@param cols the number of columns.
	 *	@return	A rows-by-cols matrix containing pseudorandom values drawn from the standard normal distribution.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type randn(const size_type rows, const size_type cols)
	{
#ifdef ARMA_EXT_USE_CPP11
		T out(rows, cols);
        out.imbue(&randn<typename T::elem_type>);
		return out;
#else
		return arma::randn<T>(rows, cols);
#endif
	}
	
	/**
	 *	@brief	Overloaded function for randn.
	 *	@return A \f$n\f$-by-\f$n\f$ matrix containing pseudorandom values drawn from the standard normal distribution.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type randn(const size_type n)
	{
		if (T::is_col)
			return arma_ext::randn<T>(n, 1);
		else if (T::is_row)
			return arma_ext::randn<T>(1, n);
		else
			return arma_ext::randn<T>(n, n);
	}

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

#ifdef ARMA_EXT_USE_CPP11
		std::stable_sort(pairs.begin(), pairs.end(),
                         [&](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b)->bool {
			return b.second > a.second;
		});
#else
		std::stable_sort(pairs.begin(), pairs.end(), internal::sort_pair_by_second_descend<size_t, double>());
#endif

		arma::uvec out(n);
#ifdef ARMA_EXT_USE_CPP11
		auto itr = pairs.begin();
        out.imbue([&]() { return (itr++)->first; });
#else
        std::vector<std::pair<arma_ext::size_type, double> >::iterator itr = pairs.begin();
        for (uword i = 0 ; i < out.n_elem ; i++)
            out[i] = (itr++)->first;
#endif
		
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
	typename std::enable_if<std::or_<T::is_col, T::is_row>::value, T>::type randvalues(const T& in,
                                                                                       uword k,
                                                                                       const typename arma::enable_if< arma::is_arma_type<T>::value>::result* junk = 0)
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
