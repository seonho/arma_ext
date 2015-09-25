/**
 *	@file		indexing.hpp
 *	@brief		Indexing functions
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

#include <cassert>

namespace arma_ext
{
	using namespace arma;

	//!	@addtogroup	ind
	//!	@{

	/**
	 *	@brief	Convert subscripts to linear indices.
	 *	@param rows	the row size of matrix
	 *	@param cols	the column size of matrix
	 *	@param r	the row subscript
	 *	@param c	the column subscript
	 *	@return	the linear index equivalent to the row and column subscript @c r and @c c for a matrix of size @c rows and @c cols.
	 *	@note	Currently, only support 2D input.
	 */
	inline size_type sub2ind(size_type rows, size_type cols, size_type r, size_type c)
	{
		assert(r < rows && c < cols);
		return r + rows * c;
	}

	//!	Overloaded sub2ind for vectors
	inline ucolvec sub2ind(size_type rows, size_type cols, ucolvec r, ucolvec c)
	{
		//assert(r < rows && c < cols);
		return r + rows * c;
	}

	//! template function specialization for double type
	inline size_type sub2ind(double rows, double cols, double r, double c)
	{
		return sub2ind((size_type)arma_ext::round(rows), (size_type)arma_ext::round(cols), (size_type)arma_ext::round(r), (size_type)arma_ext::round(c));
	}

	/**
	 *	@brief	Subscripts from linear index
	 *	@param rows	the row size of matrix
	 *	@param cols	the column size of matrix
	 *	@param ind	A linear index.
	 *	@return	The subscripts correspond to the given linear index
	 */
	inline urowvec2 ind2sub(size_type rows, size_type cols, size_type ind)
	{
		urowvec2 out;
		out[0] = ind % rows; out[1] = ind / rows;
		return out;
	}

	/**
	 *	@brief	An implementation of MATLAB's colon operator start:interval:end.
	 *			Create a vector contains [start, start + interval, ..., start + m * interval], where m = floor((end - start) / interval).
	 *	@param start	the first value
	 *	@param interval interval
	 *	@param end		the last value
	 *	@param junk		reserved
	 *	@return	A vector that contains the sequence.
	 */
	template <typename vec_type>
	vec_type colon(const typename vec_type::pod_type start, 
				   const typename vec_type::pod_type interval, 
				   const typename vec_type::pod_type end, 
				   const typename arma_Mat_Col_Row_only<vec_type>::result* junk = 0)
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
	 *	@brief	Overloaded function for sequence with interval as 1.<br>
	 *			It is the same as start:end.
	 *	@param start	the first value
	 *	@param end		the last value
	 *	@param junk		reserved
	 */
	template <typename vec_type>
	inline vec_type colon(const typename vec_type::pod_type start, 
						  const typename vec_type::pod_type end, 
						  const typename arma_Mat_Col_Row_only<vec_type>::result* junk = 0)
	{
		return colon<vec_type>(start, typename vec_type::pod_type(1), end);
	}

	//!	@}
}