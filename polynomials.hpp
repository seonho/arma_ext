/**
 *	@file		polynomials.hpp
 *	@brief		Polynomial functions.
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

namespace arma_ext
{
	using namespace arma;

	//!	@addtogroup	poly
	//!	@{
	
	/**
	 	@brief	Polynomial roots.<br>
				<br>
				The polynomial \f$s^3 - 6s^2 - 72s - 27\f$ is given then, the roots of this polynomial are returned in a column vector.
		
				@code
					vec p = "1 -6 -72 -27";
					cx_vec r = roots(p);
					real(r).print("r = "); // the polynomial p has real roots.
				@endcode
				
				The output is
				\f$ r = \begin{matrix} 12.1229 \\ -5.7345 \\ -0.3884 \end{matrix} \f$

	 	@param c The polynomial coefficient vector.
	 	@return	A column vector whose elements are the roots of the polynomial @c c.
		
	 */
	template <typename vec_type>
	cx_vec roots(const vec_type& c)
	{
		cx_vec r;
		cx_mat eigvec;
		roots(c, r, eigvec);
		return r;
	}
	
	/// Internal implementation of Polynomial roots function.
	template <typename vec_type>
	void roots(const vec_type& c, cx_vec& eigval, cx_mat& eigvec)
	{
		typedef typename vec_type::elem_type elem_type;
		typedef Mat<elem_type> mat_type;
		const uword n = c.n_elem - 1;
		mat_type A = arma_ext::diag(ones<Col<elem_type> >(n - 1, 1), -1);
		A.row(0) = -c(span(1, n)) / c[0];
		eig_gen(eigval, eigvec, A);
	}

	//!	@}
}