/**
 *	@file		random_impl_arma.hpp
 *	@brief		Random number generation functions implementation
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2007-2014, Seonho Oh
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

namespace arma_ext
{
	using namespace arma;

	//!	@addtogroup	rand
	//!	@{

	/**
	 *	@brief	Uniformly distributed pseudorandom number.
	 *	@return	A pseudorandom value drawn from the standard uniform distribution on the open interval \f$(0, 1)\f$.
	 */
	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type rand()
	{
		return arma::randu<T>();
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
		return arma::randu<T>(rows, cols);
	}
	
	/**
	 *	@brief	Overloaded function for rand.
	 *	@return A \f$n\f$-by-\f$n\f$ matrix containing pseudorandom values drawn from the standard uniform distribution on the open interval \f$(0, 1)\f$.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type rand(const size_type n)
	{
		if (T::is_col)
			return arma::randu<T>(n, 1);
		else if (T::is_row)
			return arma::randu<T>(1, n);
		
		return arma_ext::rand<T>(n, n);
	}
	
	/**
	 *	@brief	Normally distributed pseudorandom numbers.
	 *	@return	A pseudorandom value drawn from the standard normal distribution.
	 */
	template <typename T>
	inline typename std::enable_if<std::is_floating_point<T>::value, T>::type randn()
	{
		return arma::randn<T>();
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
		return arma::randn<T>(rows, cols);
	}
	
	/**
	 *	@brief	Overloaded function for randn.
	 *	@return A \f$n\f$-by-\f$n\f$ matrix containing pseudorandom values drawn from the standard normal distribution.
	 */
	template <typename T>
	inline typename std::enable_if<arma::is_arma_type<T>::value, T>::type randn(const size_type n)
	{
		if (T::is_col)
			return arma::randn<T>(n, 1);
		else if (T::is_row)
			return arma::randn<T>(1, n);

		return arma_ext::randn<T>(n, n);
	}
}