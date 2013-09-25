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

		if (((double)end - start) / interval < 0) return x;

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

	/// Padding method
	enum pad_method : uword  {
		constant,		///< Pad array with constant value.
		circular,		///< Pad with circular repetition of elements within the dimension.
		replicate,		///< Pad by repeating border elements of array.
		symmetric		///< Pad array with mirror reflections of itself.
	};

	/// Padding direction
	enum pad_direction : uword {
		both,		///< Pads before the first element and after the last array element along each dimension.
		pre,		///< Pad after the last array element along each dimension.
		post		///< Pad before the first element along each dimension.
	};
		
	/**
	 *	@brief	Pads array A with @c padsize (rows, cols) number of zeros along the k-th dimension of A. @c padsize should be a nonnegative integers.
	 *	@param A			The source array.
	 *	@param rows			A row pad size.
	 *	@param cols			A column pad size.
	 *	@return	Padded array.
	 */
	template <typename T>
	T padarray(const T& A, uword rows, uword cols)
	{
		typedef T::elem_type elem_type;
		return constantpad(A, rows, cols, elem_type(0), both);
	}

	/**
	 *	@brief	Pads array A with @c padval (a scalar) instead of with zeros in the direction specified by the @c direction.
	 *	@param A			The source array.
	 *	@param rows			A row pad size.
	 *	@param cols			A column pad size.
	 *	@param padval		A scalar value.
	 *	@param direction	The pad direction, see #pad_direction. By default, direction is 'both'.
	 *	@return	Padded array.
	 */
	template <typename T>
	T padarray(const T& A, uword rows, uword cols, typename T::elem_type padval, pad_direction direction = both)
	{
		return constantpad(A, rows, cols, padval, direction);
	}

	/**
	 *	@brief	Pads array A with using the specified @c method and @c direction.
	 *	@param A			The source array.
	 *	@param rows			A row pad size.
	 *	@param cols			A column pad size.
	 *	@param method		The pad method, see #pad_method.
	 *	@param direction	The pad direction, see #pad_direction.
	 *	@return	Padded array.
	 */
	template <typename T>
	T padarray(const T& A, uword rows, uword cols, pad_method method, pad_direction direction)
	{
		typedef T::elem_type elem_type;
		const uword size = sizeof(elem_type);

		if (method == constant)
			return constantpad(A, rows, cols, elem_type(0), direction);

		T out;
		arma::field<uvec> indices = getPaddingIndices(A, rows, cols, method, direction);

		uvec ri = indices(0), ci = indices(1);
		out.set_size(ri.n_elem, ci.n_elem);

		const uword colsize = size * A.n_rows;
		uword offset = (direction == post) ? 0 : rows;
		
		for (uword c = 0 ; c < ci.n_elem ; c++)
			memcpy(out.colptr(c) + offset, A.colptr(ci[c]), colsize);

		uvec copyflag;
		switch (direction) {
		case pre:
			copyflag = join_cols(ones<uvec>(rows, 1), zeros<uvec>(A.n_rows, 1));
			break;
		case post:
			copyflag = join_cols(zeros<uvec>(A.n_rows, 1), ones<uvec>(rows, 1));
			break;
		case both:
			copyflag = join_cols(join_cols(ones<uvec>(rows, 1), zeros<uvec>(A.n_rows, 1)), ones<uvec>(rows, 1));
			break;
		}

		for (uword r = 0 ; r < ri.n_elem ; r++) {
			if (copyflag[r])
				out.row(r) = out.row(ri[r] + offset);
		}

		return out;
	}

	/// internal function
	template <typename T>
	T constantpad(const T& A, uword rows, uword cols, typename T::elem_type padval, pad_direction direction)
	{
		typedef T::elem_type elem_type;
		const uword size = sizeof(elem_type);

		T out;

		switch (direction) {
		case both:
			out.set_size(A.n_rows + rows * 2, A.n_cols + cols * 2);
			out.fill(padval);
			out(span(rows, A.n_rows + rows - 1), span(cols, A.n_cols + cols - 1)) = A;
			break;
		case pre:
			out.set_size(A.n_rows + rows, A.n_cols + cols);
			out.fill(padval);
			out(span(rows, A.n_rows + rows - 1), span(cols, A.n_cols + cols - 1)) = A;
			break;
		case post:
			out.set_size(A.n_rows + rows, A.n_cols + cols);
			out.fill(padval);
			out(span(0, A.n_rows - 1), span(0, A.n_cols - 1)) = A;
			break;
		default:
			// throw exception
			throw std::invalid_argument("direction is invalid");
		}

		return out;
	}

	template <typename T>
	arma::field<arma::uvec> getPaddingIndices(const T& A, uword rows, uword cols, pad_method method, pad_direction direction)
	{
		switch (method) {
		case circular:
			return circularpad(A, rows, cols, direction);
		case symmetric:
			return symmetricpad(A, rows, cols, direction);
		case replicate:
			return replicatepad(A, rows, cols, direction);
		}

		return arma::field<arma::uvec>();
	}

	/// Modulus after division
	template <typename vec_type>
	inline vec_type mod(const vec_type& X, typename vec_type::elem_type Y)
	{
		typedef vec_type::elem_type elem_type;
		assert(Y != 0);
		
		vec_type M;
		switch (X.vec_state) {
		case 0: // matrix
			M = X - arma::conv_to<vec_type>::from(arma::floor(arma::conv_to<mat>::from(X) / (double)Y)) * Y;
			break;
		case 1:
		case 2:
			M = X - arma::conv_to<vec_type>::from(arma::floor(arma::conv_to<vec>::from(X) / (double)Y)) * Y;
			break;
		}
		
		return M;
	}

	template <typename T>
	arma::field<arma::uvec> circularpad(const T& A, uword rows, uword cols, pad_direction direction)
	{
		arma::field<arma::uvec> indices(2);
		int M, p;

		for (uword k = 0 ; k < 2 ; k++) {
			p = (k == 0) ? (int)rows : (int)cols;
			M = (k == 0) ? A.n_rows : A.n_cols;

			switch (direction) {
			case pre:
				indices(k) = arma::conv_to<uvec>::from(mod(sequence<ivec>(-p, M - 1), M));
				break;
			case post:
				indices(k) = arma::conv_to<uvec>::from(mod(sequence<ivec>(0, M + p - 1), M));
				break;
			case both:
				indices(k) = arma::conv_to<uvec>::from(mod(sequence<ivec>(-p, M + p - 1), M));
				break;
			}
		}

		return indices;
	}

	template <typename T>
	arma::field<arma::uvec> symmetricpad(const T& A, uword rows, uword cols, pad_direction direction)
	{
		arma::field<arma::uvec> indices(2);
		int M, p;

		for (uword k = 0 ; k < 2 ; k++) {
			p = (k == 0) ? (int)rows : (int)cols;
			M = (k == 0) ? A.n_rows : A.n_cols;

			arma::uvec dimNum = arma::conv_to<uvec>::from(arma::join_cols(sequence<ivec>(0, M - 1), sequence<ivec>(M - 1, -1, 0)));

			switch (direction) {
			case pre:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(sequence<ivec>(-p, M - 1), 2 * M)));
				break;
			case post:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(sequence<ivec>(0, M + p - 1), 2 * M)));
				break;
			case both:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(sequence<ivec>(-p, M + p - 1), 2 * M)));
				break;
			}
		}

		return indices;
	}

	template <typename T>
	arma::field<arma::uvec> replicatepad(const T& A, uword rows, uword cols, pad_direction direction)
	{
		arma::field<arma::uvec> indices(2);
		int M, p;

		for (uword k = 0 ; k < 2 ; k++) {
			p = (k == 0) ? (int)rows : (int)cols;
			M = (k == 0) ? A.n_rows : A.n_cols;

			switch (direction) {
			case pre:
				indices(k) = join_cols(zeros<uvec>(p, 1), sequence<uvec>(0, M - 1));
				break;
			case post:
				indices(k) = join_cols(sequence<uvec>(0, M - 1), ones<uvec>(rows, 1) * (M - 1));
				break;
			case both:
				indices(k) = join_cols(join_cols(zeros<uvec>(p, 1), sequence<uvec>(0, M - 1)), ones<uvec>(p, 1) * (M - 1));
				break;
			}
		}

		return indices;
	}

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
		typedef vec_type::elem_type elem_type;
		typedef Mat<typename elem_type> mat_type;

		const uword n = (v.is_col ? v.n_rows : v.n_cols) + (uword)abs(k);
		mat_type X = zeros<mat_type>(n, n);
		X.diag(k) = v;
		return X;
	}
}

#include "arrayop_ext.hpp"	// for handling conv_to::from