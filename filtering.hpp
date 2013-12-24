/**
 *	@file		filtering.hpp
 *	@brief		Convolution and digital filtering functions.
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

namespace arma_ext
{
	using namespace arma;

	//!	@addtogroup	foufilt
	//!	@{

	/**
	 *	Convolution types
	 */
	enum convolution_type : uword {
		full,	//!	The full two-dimensional convolution (default).
		same,	//!	The central part of the convolution of the same size as @c A.
		valid	//!	Only those parts of the convolution  that are computed without the zero-padded edges.<br>Using this option, #size (C) = max([ma - max(0, mb - 1), na - max(0, nb - 1)], 0).
	};

	/**
	 *	@brief	An arma style intermediate interface class implementation of 2D convolution operation
	 *	@see	conv2
	 */
	class glue_conv2
	{
	public:
		/**
		 *	@brief	A rudimentary implementation of the 2D convolution operation.
		 *	@param out 
		 *	@param X 
		 *	@return 
		 */
		template <typename T1, typename T2>
		inline static void apply(Mat<typename T1::elem_type>&out, const Glue<T1, T2, glue_conv2>& X)
		{
			arma_extra_debug_sigprint();

			typedef typename T1::elem_type elem_type;

			const Mat<elem_type>& a = X.A;
			const Mat<elem_type>& b = X.B;

			uword ma = a.n_rows, na = a.n_cols;
			uword mb = b.n_rows, nb = b.n_cols;
			uword mc = ma + mb - 1, nc = na + nb - 1;
			
			out.set_size(mc, nc);

			concurrency::parallel_for(uword(0), nc, [&](uword c) {
			//for (uword c = 0 ; c < nc ; c++) {
				for (uword r = 0 ; r < mc ; r++) {
					elem_type value = 0;

					const uword minu = (r + 1 > mb) ? r - mb + 1 : 0;
					const uword maxu = std::min(ma - 1, r);

					const uword minv = (c + 1 > nb) ? c - nb + 1 : 0;
					const uword maxv = std::min(na - 1, c);

					for (uword v = minv ; v <= maxv ; v++) {
						for (uword u = minu ; u <= maxu ; u++) {
							value += a(u, v) * b(r - u, c - v);
						}
					}

					out(r, c) = value;
				}
			//}
			});

			switch (X.aux_uword) {
			case full:
				// do nothing
				break;
			case valid:
				out = out(span(mb - 1,ma - 1), span(nb - 1,na - 1));
				break;
			case same:
				{
					uword r1 = (uword)std::floor(mb / 2.0);	// zero-begin index corrected
					uword r2 = r1 + ma - 1;
					uword c1 = (uword)std::floor(nb / 2.0);	// zero-begin index corrected
					uword c2 = c1 + na - 1;
					out = out(span(r1, r2), span(c1, c2));
				}
				break;
			default:
				break;
			}
		}
	};

	/**
	 *	@brief	2-D convolution of matrices A and B.<br>
	 *			
	 *	@param A			The input matrix.
	 *	@param B			The convolution kernel matrix.
	 *	@param conv_type	The convolution type
	 *	@return	convolution The result matrix.
	 *	@see	http://www.mathworks.co.kr/kr/help/matlab/ref/conv2.html
	 *	@see	convolution_type
	 *	@note	This function is preliminary; it is not yet fully optimized.
	 */
	template <typename T1, typename T2>
	inline const Glue<T1, T2, glue_conv2> conv2(const Base<typename T1::elem_type, T1>& A, const Base<typename T1::elem_type, T2>& B, const uword conv_type = full)
	{
		arma_extra_debug_sigprint();
  
		return Glue<T1, T2, glue_conv2>(A.get_ref(), B.get_ref(), conv_type);
	}

#ifndef DOXYGEN
	template <typename eT>
	__declspec(deprecated) arma::Mat<eT> conv2_(const arma::Mat<eT>& a, const arma::Mat<eT>& b, convolution_type conv_type = full)
	{
		// check arguments
		uword ma = a.n_rows, na = a.n_cols;
		uword mb = b.n_rows, nb = b.n_cols;

		if (conv_type == valid)
			assert(mb <= ma && nb <= na);

		arma::Mat<eT> c = zeros(ma + mb - 1, na + nb - 1);

		// do full convolution
		for (uword j = 0 ; j < nb ; j++) {
			uword c1 = j;
			uword c2 = c1 + na - 1;
			for (uword i = 0 ; i < mb ; i++) {
				uword r1 = i;
				uword r2 = r1 + ma - 1;
				c(span(r1, r2), span(c1, c2)) = c(span(r1, r2), span(c1, c2)) + b(i, j) * a;
				//concurrency::parallel_for (uword(0), na, [&](uword cl) {
				////for(uword cl = 0 ; cl < na ; cl++)/ {
				//	for (uword rk = 0 ; rk < ma ; rk++) {
				//		c(rk + r1, cl + c1) += b(i, j) * a(rk, cl);
				//	}
				////}
				//});
			}
		}

		switch (conv_type) {
		case same:
			{
				uword r1 = (uword)std::floor(mb / 2.0);	// zero-begin index corrected
				uword r2 = r1 + ma - 1;
				uword c1 = (uword)std::floor(nb / 2.0);	// zero-begin index corrected
				uword c2 = c1 + na - 1;
				c = c(span(r1, r2), span(c1, c2));
			}
			break;
		case valid:
			c = c(span(mb - 1,ma - 1), span(nb - 1,na - 1));
			break;
		case full:
		default:
			// nothing to do, full convolution done
			break;
		}

		return c;
	}
#endif
	//!	@}
}