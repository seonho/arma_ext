/**
 *	@file		numdiff.hpp
 *	@brief		Numerical differentiation functions
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

#include <cassert>

namespace arma_ext
{
	using namespace arma;
	
	//!	@addtogroup	numintdiff
	//!	@{

	/**
	 *	@brief	Difference and approximate derivative.
	 *	@tparam elem_type the type of elements stored in the matrix.
	 *	@param X A vector or a matrix.
	 *	@param n The order of differentiation.
	 *	@param dim (only for matrix) The dimension to be differentiated along.
	 *	@return If @c X is a vector, then #diff(@c X) returns a vector, one element
	 *			shorter than @c X, of differences between adjacement elements:<br>
	 *			\f[ \begin{bmatrix} X(2) - X(1) & X(3) - X(2) & \cdots & X(n) - X(n - 1) \end{bmatrix} \f]
	 *			If @c X is a matrix, then #diff(@c X) returns a matrix of row differences:<br>
	 *			\f[ \begin{bmatrix} X(2:m, :) - X(1:m - 1, :) \end{bmatrix} \f]
	 *			In general, #diff(@c X) returns the difference calculated along the first non-singleton (#size(@c X, @c dim) > 1) dimension of @c X.<br>
	 *			<br>
	 *			#diff(@c X, @c n) applies #diff recursively @c n times, resulting in the nth difference. Thus #diff(@c X, 2) is the same as #diff(#diff(@c X)).<br>
	 *			<br>
	 *			#diff(@c X, @c n, @c dim) is the nth difference function calculated along the dimension specified by scalar @c dim.
	 *			If order @c n equals or exceeds the length of dimension @c dim, #diff returns an empty array.
	 *			
	 */
	template <typename elem_type>
	Mat<elem_type> diff(const Mat<elem_type>& X, size_type n = 1, size_type dim = 0)
	{
		assert(n > 0);
		Mat<elem_type> y;

		if (X.empty() || dim > 1 || X.n_elem == 1) return y;

		if (n > 1) return diff(diff(X, n - 1, dim));

		if (X.is_vec()) {
			y.resize(std::max(X.n_rows - 1, (size_type)1), std::max(X.n_cols - 1, (size_type)1));
			std::adjacent_difference(X.begin(), X.end(), 
				stdext::unchecked_array_iterator<elem_type *>(y.begin()));
			return y;
		}
		
		switch (dim) {
		case 0: // row difference
			y.resize(X.n_rows - 1, X.n_cols);
			for (size_type c = 0 ; c < X.n_cols ; c++)
				std::adjacent_difference(X.begin_col(c), X.end_col(c), 
				stdext::unchecked_array_iterator<elem_type *>(y.begin_col(c)));
			break;
		case 1: // column difference
			y.resize(X.n_rows, X.n_cols - 1);
			for (size_type c = 0 ; c < X.n_cols - 1; c++)
				y.col(c) = X.col(c + 1) - X.col(c);
			break;
		default:
			throw std::invalid_argument("dim should be 0 or 1"); //
		}

		return y;
	}

	//!	@}
}