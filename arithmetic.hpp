/**
 *	@file		arithmetic.hpp
 *	@brief		Arithmetic operations
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

#include <numeric>	// for std::adjacent_difference
#include <cassert>

namespace arma_ext
{
	//!	@addtogroup	arit
	//!	@{

	/**
	 *	@brief	Round operation for scalar value.
	 *	@param x the given scalar value
	 *	@return	 the rounded value
	 */
	template <typename T>
	inline typename std::enable_if<!arma::is_arma_type<T>::value, T>::type
		round(const T& x) { return (T)arma::eop_aux::round(x); }

	//! Modulus after division
	template <typename vec_type>
	inline vec_type mod(const vec_type& X, typename vec_type::elem_type Y)
	{
		typedef typename vec_type::elem_type elem_type;
		assert(Y != 0);
		
		vec_type M;
		switch (X.vec_state) {
		case 0: // matrix
			M = X - arma::conv_to<vec_type>::from(
				arma::floor(arma::conv_to<mat>::from(X) / (double)Y)) * Y;
			break;
		case 1:
		case 2:
			M = X - arma::conv_to<vec_type>::from(
				arma::floor(arma::conv_to<vec>::from(X) / (double)Y)) * Y;
			break;
		}
		
		return M;
	}

	//!	@}
}
