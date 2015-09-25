/**
 *	@file		fundamentals.hpp
 *	@brief		Includes headers implement MATLAB language fundamentals.
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

//!	@defgroup	fund	Fundamentals
//! @brief		Syntax, operators, data types, array indexing and manipulation
//!	@{
//!		@defgroup	matar	Matrices and Arrays
//!		@brief		Array indexing, concatenation, sorting, and reshping; set and bit-wise operations
//!		@{
//!			@defgroup	arr		Array Creation and Concatenation
//!			@brief		Create or combine scalars, vectors, matrices, or arrays
//
//!			@defgroup	ind		Indexing
//!			@brief		Access array elements.
//
//!			@defgroup	arrdim	Array Dimensions
//!			@brief		Determine array size or shape
//
//!			@defgroup	sort	Sorting and Reshaping Arrays
//!			@brief		Sort, rotate, permute, reshape, or shift array contents
//!		@}
//
//!		@defgroup	ops		Operators and Elementary Operations
//!		@brief		Arithmetic, relational, logical, set, and bit-wise operations.
//!	@}

namespace arma_ext
{
	using namespace arma;

	//! @addtogroup	arr
	//!	@{

	/**
	 *	@brief	Diagonal matrices and diagonals of matrix.
	 *	@param v The input vector.
	 *	@param k The diagonal index.
	 *	@return	A square matrix of order \f$n+abs(k)\f$, with the elements of \f$v\f$ on the \f$k\f$th diagonal.<br>
	 *			\f$k=0\f$ represents the main diagonal, \f$k>0\f$ above the main diagonal, and \f$k<0\f$ below the main diagonal.
	 *	@note	This function is preliminary; it is not yet fully implemented.
	 */
	template <typename vec_type>
	Mat<typename vec_type::elem_type> diag(const vec_type& v , int k = 0)
	{
		typedef typename vec_type::elem_type elem_type;
		typedef Mat<elem_type> mat_type;

		const uword n = (v.is_col ? v.n_rows : v.n_cols) + (uword)abs(k);
		mat_type X = zeros<mat_type>(n, n);
		X.diag(k) = v;
		return X;
	}

	//!	@brief	An arma style intermediate interface class for replicating cells (matrix elements).
	//!	@see	repcel
	class op_repcel
	{
	public:
		//!	A rudimentary implementation of the replicating cells (matrix elements).
		template <typename T1>
		inline static void apply(Mat<typename T1::elem_type>& out, const Op<T1, op_repcel>& in)
		{
			arma_extra_debug_sigprint();

			typedef typename T1::elem_type eT;
  
			const unwrap_check<T1> tmp(in.m, out);
			const Mat<eT>& X     = tmp.M;

			const uword r = in.aux_uword_a;
			const uword c = in.aux_uword_b;

			const uword m = X.n_rows;
			const uword n = X.n_cols;

			out.set_size(m * r, n * c);

#if defined(USE_PPL)
			concurrency::parallel_for(uword(0), m, [&](uword i) {
#elif defined(USE_OPENMP)
	#pragma omp parallel for
			for (int si = 0 ; si < (int)m ; si++) {
				uword i = (uword)si;
#else
            for (uword i = 0 ; i < m ; i++) {
#endif
				for (uword j = 0 ; j < n ; j++) {
					out.submat(span(r * i, r * (i + 1) - 1), 
							   span(c * j, c * (j + 1) - 1)).fill(X.at(i, j));
				}
#ifdef USE_PPL
			});
#else
            }
#endif
		}
	};

	/**
	 *	@brief	Repeats cells (matrix elements) m x n times.
	 *	@param	A	The input matrix.
	 *	@param	r	The number of rows.
	 *	@param	c	The number of columns.
	 *	@return A replicated matrix.
	 */
	template <typename T1>
	inline const Op<T1, op_repcel> repcel(const Base<typename T1::elem_type, T1>& A, const size_type r, const size_type c)
	{
		arma_extra_debug_sigprint();

		return Op<T1, op_repcel>(A.get_ref(), r, c);
	}

	/**
	 *	@brief	Computes all possible ntuples
	 *	@param x	A vector of the tuple element 1
	 *	@param y	A vector of the tuple element 2
	 */
	template <typename vec_type>
	inline arma::Mat<typename vec_type::elem_type> ntuples(const vec_type& x, const vec_type& y)
	{
		arma::Mat<typename vec_type::elem_type> out(2, x.n_elem * y.n_elem);
		out.row(0) = repcel(x, 1, y.n_elem);
		out.row(1) = repmat(y, 1, x.n_elem);
		return out;
	}

	//!	@}

	//! @addtogroup	arrdim	
	//!	@{

	/// Array dimensions.
	template<typename T>
	inline arma::urowvec size(const arma::Mat<T>& x)
	{
		static_assert(ARMA_VERSION_MAJOR <= 5 && ARMA_VERSION_MINOR < 500, "This function is deprecated. Use arma::size instead.");

		arma::urowvec siz(2);
		siz[0] = x.n_rows; siz[1] = x.n_cols;
		return siz;
	}

	/// Array dimension for given dimension index.
	template <typename T>
	inline size_type size(const arma::Mat<T>& x, size_type dim)
	{
		static_assert(ARMA_VERSION_MAJOR <= 5 && ARMA_VERSION_MINOR < 500, "This function is deprecated. Use arma::size instead.");

		switch (dim) {
		case 0:
			return x.n_rows;
		case 1:
			return x.n_cols;
		default:
			throw std::invalid_argument("dim must be one of 0, 1.");
		}
	}

	/// Overloaded for Cube type.
	template <typename T>
	inline arma::urowvec size(const arma::Cube<T>& x)
	{
		static_assert(ARMA_VERSION_MAJOR <= 5 && ARMA_VERSION_MINOR < 500, "This function is deprecated. Use arma::size instead.");

		arma::urowvec siz(3);
		siz[0] = x.n_rows; siz[1] = x.n_cols; siz[2] = x.n_slices;
		return siz;
	}

	/// Cube type array dimension for given dimension index
	template <typename T>
	inline size_type size(const arma::Cube<T>& x, size_type dim)
	{
		switch (dim) {
		case 0:
			return x.n_rows;
		case 1:
			return x.n_cols;
		case 2:
			return x.n_slices;
		default:
			throw std::invalid_argument("dim must be one of 0, 1, 2.");
		}
	}

	//!	@}

	//!	@ingroup	sort
	//!	@{
	/**
	 *	@brief	Shifts the dimensions to the left and removes any leading singleton dimensions, or wraps the n leading dimensions to the end.
	 *	@param X A matrix.
	 *	@param n The count of leading dimensions to be shifted.
	 *	@note	arma::Mat only support 2D array.
	 *			If @c X is a row vector, then #shiftdim returns a column vector.<br>
	 *			If @c X is a matrix, then #shiftdim returns a matrix that leading non-singleton dimensions.
	 */
	template <typename T>
	Mat<T> shiftdim(const Mat<T>& X, arma::shword n = 0)
	{
		//if (n == 0) {
		//	// Find leading singleton dimensions
		//	urowvec siz = size(X);
		//	urowvec N = find(siz != 1, 1, "first");
		//	if (N[0] != 0) return shiftdim(X, N[0]);
		//	return X;
		//}
		if (X.is_rowvec())
			return X.t();
		return X;
	}

	//!	@}
}

#include "arithmetic.hpp"	// arithmetic operations
#include "indexing.hpp"		// indexing functions
#include "logical.hpp"		// logical operations