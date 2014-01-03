/**
 *	@file		logical.hpp
 *	@brief		Logical operations
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

namespace arma_ext
{
	using namespace arma;
    
#ifdef _MSC_VER
#define ISNAN _isnan
#else
#define ISNAN std::isnan
#endif

	//!	@defgroup	logicalop Logical Operations
	//!	@brief		True or false (Boolean) conditions.
	//!	@ingroup	ops
	//! @{

	/**
	 *	@brief	Determine whether any matrix elements are nonzero
	 *	@param	m the input matrix
	 *	@return	true / false
	 *	@see	http://www.mathworks.co.kr/kr/help/matlab/ref/any.html
	 */
	inline bool any(const umat& m)
	{
		return arma::any(arma::any(m));
	}

	/**
	 *	@brief	Array elements that are NaN
	 *	@param A An input vector
	 *	@returns an array the same sizes as A containing logical 1 (true) where the elements of A are NaNs and logical 0 (false) where they are not.
	 */
	uvec isnan(const vec& A)
	{
		uvec out(A.size());
		const double* iptr = A.memptr();
		uword* optr = out.memptr();
		for (uword i = 0 ; i < A.size() ; i++)
			optr[i] = ISNAN(iptr[i]) > 0 ? 1 : 0;
		
		return out;
	}

	/**
	 *	@brief	Check scalar value is NaN
	 *	@param value the scalar value
	 *	@return true/false
	 */
	inline bool isnan(double value)
	{
		return ISNAN(value) > 0 ? true : false;
	}
	
	/**
	 *	@brief	Find logical NOT of array or scalar input.
	 *	@param A An array or a scalar input.<br> input type should be uvec or umat or uword
	 *	@return A logical NOT of input array or scalar @c A.
	 */
	template <typename mat_type>
	inline mat_type logical_not(const mat_type& A)
	{
		return ones<mat_type>(A.n_elem) - A;
	}

	//! Template function specialization for logical_not.
	template <>
	inline uword logical_not(const uword& A)
	{
		return (1 - A);
	}

	//! @}
}