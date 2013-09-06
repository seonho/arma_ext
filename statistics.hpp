/**
 *	@file		statistics.hpp
 *	@brief		An implemenation of statistics operations
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

namespace arma_ext
{
	using namespace arma;

	/**
	 *	@brief	2-D correlation coefficient
	 *	@param A
	 *	@param B
	 *	@return	the correlation coefficient between @c A and @c B, where @c A and @c B are matrices or vectors of the same size.
	 *			@c corr2 computes the correlation coefficient using
	 *			\f[
	 *				r = \frac{ \sum_{m}\sum_{n}(A_{mn}-\bar{A}) (B_{mn}-\bar{B}) }{\sqrt{ \left( \sum_{m}\sum_{n}(A_{mn}-\bar{A})^2 \right) \left( \sum_{m}\sum_{n}(B_{mn}-\bar{B})^2 \right)}}
	 *			\f]
	 *			where
	 *			\f$ \bar{A}\f$=mean2(A), and \f$\bar{B}\f$=mean2(B).
	 *	@see	http://www.mathworks.co.kr/kr/help/images/ref/corr2.html
	 */
	template <typename mat_type, typename prec_type>
	prec_type corr2(const mat_type& A, const mat_type& B)
	{
		typedef Mat<prec_type> tmat;
		tmat A1, B1;
		A1 = conv_to<tmat>::from(A);
		B1 = conv_to<tmat>::from(B);
		A1 -= mean2(A1);
		B1 -= mean2(B1);
		return accu(A1 % B1) / sqrt(accu(square(A1)) * accu(square(B1)));
	}
}