/**
 *	@file		arrayop.hpp
 *	@brief		An implemenation of array operations
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
#include "arma_ext/logicalop.hpp"
#include <cassert>

namespace arma_ext
{
	using namespace arma;

	/**
	 *	@brief	An implementation of MATLAB's colon operator start:invertal:end.
	 *			Create a vector contains [start, start + interval, ..., start + m * interval], where m = floor((end - start) / interval).
	 *	@param start	the first value
	 *	@param interval interval
	 *	@param end		the last value
	 *	@return	A vector that contains the sequence.
	 */
	template <typename vec_type>
	vec_type sequence(const typename vec_type::pod_type start, const typename vec_type::pod_type interval, const typename vec_type::pod_type end)
	{
		typedef typename vec_type::elem_type eT;
		typedef typename vec_type::pod_type   T;

		vec_type x;

		if (std::abs((double)end - start) / interval < 0) return x;

		x.set_size(uword(std::floor((end - start) / (double)interval)) + 1);
		eT* x_mem = x.memptr();
		for (uword i = 0 ; i < x.n_elem ; i++)
			x_mem[i] = eT(start + i * interval);

		return x;
	}

	/**
	 *	@brief	Overloaded function for sequence with interval as 1.
	 *			It is the same as start:end.
	 */
	template <typename vec_type>
	inline vec_type sequence(const typename vec_type::pod_type start, const typename vec_type::pod_type end)
	{
		return sequence<vec_type>(start, vec_type::pod_type(1), end);
	}

	/**
	 *	@brief	Repeats cells m x n times.
	 *	@param	in	The input matrix.
	 *	@param	r	The number of rows.
	 *	@param	c	The number of columns.
	 *	@return A	replicated matrix.
	 */
	template <typename T1>
	arma::Mat<T1> repcel(const arma::Mat<T1>& in, const size_type r, const size_type c)
	{
		arma::Mat<T1> out(r * in.n_rows, c * in.n_cols);
		//for (uword i = 0 ; i < in.n_rows ; i++) {
		concurrency::parallel_for(uword(0), in.n_rows, [&](uword i) {
			for (uword j = 0 ; j < in.n_cols ; j++) {
				out.submat(span(r * i, r * (i + 1) - 1), span(c * j, c * (j + 1) - 1)).fill(in.at(i, j));
			}
		});
		//}
		return out;
	}

	/**
	 *	@brief	Computes all possible ntuples
	 *	@param x	A vector of the tuple element 1
	 *	@param y	A vector of the tuple element 2
	 */
	template <typename vec_type>
	inline arma::Mat<typename vec_type::elem_type> ntuples(const vec_type& x, const vec_type& y)
	{
		arma::Mat<vec_type::elem_type> out(2, x.n_elem * y.n_elem);
		out.row(0) = repcel(x, 1, y.n_elem);
		out.row(1) = repmat(y, 1, x.n_elem);
		return out;
	}

	///
	// *	@brief	An implementation of matrix column selector from mask vector or index vector.
	// *	@param in an original matrix
	// *	@param c row vector contains logical mask or indices.
	// *	@param permute if true, c is indices, otherwise c is mask (default is false).
	// *	@return	return a new matrix contains columns that selectected from in matrix by c.
	// /
	//template <typename eT>
	//arma::Mat<eT> select_cols(arma::Mat<eT> in, const arma::urowvec& c, bool permute = false)
	//{
	//	arma::Mat<eT> out;
	//	uword n_cols;
	//	const uword* indexptr = nullptr;
	//	arma::urowvec index;
	//	if (permute) {
	//		n_cols = c.n_elem;
	//		indexptr = c.memptr();
	//	} else {
	//		n_cols = as_scalar(sum(c));
	//		index.set_size(n_cols);
	//		
	//		const uword* inmem = c.memptr();
	//		uword* ptr = index.memptr();
	//		for (uword i = 0, j = 0 ; i < c.n_elem ; i++) {
	//			if (inmem[i] > 0) ptr[j++] = i;
	//		}

	//		indexptr = index.memptr();
	//	}

	//	out.set_size(in.n_rows, n_cols);
	//	concurrency::parallel_for(uword(0), n_cols, [&](uword i) {
	//		out.col(i) = in.col(indexptr[i]);
	//	});

	//	return out;
	//}

	/**
	 *	@brief	Convert subscripts to linear indices.
	 *	@param rows	the row size of matrix
	 *	@param cols	the column size of matrix
	 *	@param r	the row subscript
	 *	@param c	the column subscript
	 *	@return	the linear index equivalent to the row and column subscript @c r and @c c for a matrix of size @c rows and @c cols.
	 */
	inline size_type sub2ind(size_type rows, size_type cols, size_type r, size_type c)
	{
		assert(r < rows && c < cols);
		return r + rows * c;
	}

	/// template function specialization for double type
	inline size_type sub2ind(double rows, double cols, double r, double c)
	{
		return sub2ind(round<size_type>(rows), round<size_type>(cols), round<size_type>(r), round<size_type>(c));
	}

	/**
	 *	@brief	Median without NaN
	 *	@param x a vector or a matrix
	 */
	template <typename vec_type>
	inline double median_(const vec_type& x)
	{
		return median(x.elem(find(arma_ext::isnan(x) == 0)));
	}

	/**
	 *	@brief	Average or mean of matrix elements
	 *	@param A An input matrix
	 */
	template <typename mat_type>
	inline typename mat_type::elem_type mean2(const mat_type& A)
	{
		return mean(vectorise(A));
	}

	/// Padding type
	enum pad_type : uword  {
		constant,		///< Pad array with constant value.
		circular,		///< Pad with circular repetition of elements within the dimension.
		replicate,		///< Pad by repeating border elements of array.
		symmetric,		///< Pad array with mirror reflections of itself.
		symmetric101	///< edca | abcdefgh | gfedc
	};

	/// Padding direction
	enum pad_direction : uword {
		both,		///< Pads before the first element and after the last array element along each dimension.
		pre,		///< Pad after the last array element along each dimension.
		post		///< Pad before the first element along each dimension.
	};
	
	/**
	 *	@brief	Pads array A with given conditions
	 *	@param A source array
	 *	@param rows	a row pad size
	 *	@param cols	a column pad size
	 *	@param pt	the pad type, see pad_type.
	 *	@param pd	the pad direction, see pad_direction.
	 *	@note	This function is preliminary; it is not yet fully optimized.
	 *			This version only support symmetric101_nobound and both direction
	 */
	template <typename T>
	T padarray(const T& A, uword rows, uword cols, pad_type pt = symmetric101, pad_direction pd = both)
	{
		typedef T::elem_type elem_type;
		const uword size = sizeof(elem_type);

		T out(A.n_rows + rows * 2, A.n_cols + cols * 2);
		
		// copy submatrix
		out(span(rows, A.n_rows + rows - 1), span(cols, A.n_cols + cols - 1)) = A;
		
		// pad array - horizontal
		concurrency::parallel_for(uword(0), cols, [&](uword i) {
			memcpy(out.colptr(i) + rows, A.colptr(cols - i), size * A.n_rows);
			memcpy(out.colptr(A.n_cols + i) + rows, A.colptr(cols - i), size * A.n_rows);
		});

		// pad - vertical
		const uword post0 = A.n_rows + rows;
		concurrency::parallel_for(uword(0), out.n_cols, [&](uword i) {
			elem_type* ptr = out.colptr(i);
			for (uword j = 0 ; j < rows ; j++)
				ptr[j] = ptr[post0 + j] = ptr[rows * 2 - i];
		});
		
		return out;
	}
}

#include "arrayop_ext.hpp"	// for handling conv_to::from