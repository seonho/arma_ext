/**
 *	@file		matrix_analysis.hpp
 *	@brief		An implemenation of array operations
 *	@author		seonho.oh@gmail.com
 *	@date		2013-10-07
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
	 *	@brief	Null space.<br>
				\f$ Z = null(A) \f$ is an orthonormal basis for the null space of A obtained from the singular value decomposition.<br>
				\f$ A * Z \f$ has negligible elements, @c Z.num_cols is the nullity of \f$A\f$, and \f$Z' * Z = I\f$.
	 *	@param A 
	 *	@return	An orthonormal basis for the null space of @c A
	 *	@note	This implementation highly dependent on the singular value decomposition.
	 */
	template <typename mat_type>
	mat_type null(const mat_type& A)
	{
		typedef typename mat_type::elem_type elem_type;

		const uword m = A.n_rows,
			  n = A.n_cols;
		
		mat_type U, V;
		Col<elem_type> S;
		
		// orthonormal basis
		arma::svd_econ(U, S, V, A);
		elem_type tol = std::max(m, n) * max(S) * std::numeric_limits<elem_type>::epsilon();
		uword r = sum(S > tol);
		return V.cols(r, n - 1);
	}
}