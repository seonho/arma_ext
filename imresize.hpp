/**
 *	@file		imresize.hpp
 *	@brief		An implemenation of imresize function
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

#include <ppl.h>

#if	(_MSC_VER <= 1600)
/// From Microsoft Visual Studio 2012, Concurrency namespace has been changed to concurrency.
/// For compatibility, namespace alias is used
namespace concurrency = Concurrency;
#endif

namespace arma_ext
{
	using namespace arma;

	//template <typename eT>
	//arma::Mat<eT> separable_conv(const arma::Mat<eT>& a, const arma::Mat<eT>& h, const arma::Mat<eT>& v)
	//{
	//	arma::Mat<eT> b(a.n_rows, a.n_cols);
	//	arma::Mat<eT> c(a.n_rows, a.n_cols);
	//	
	//	uword offset = v.n_elem / 2;
	//	// first stage
	//	concurrency::parallel_for (uword(0), a.n_cols, [&](uword j) {
	//	//for (uword j = 0 ; j < a.n_cols ; j++) {
	//		for (uword i = 0 ; i < a.n_rows ; i++) {
	//			eT conv = 0;
	//			for (uword k = 0 ; k < v.n_elem ; k++) {
	//				// bounds check for row index
	//				if (i + k < offset || i + k >= a.n_rows)
	//					continue;
	//				conv += h(k) * a(i + k - offset, j);
	//			}
	//			b(i, j) = conv;
	//		}
	//	//}
	//	});

	//	b = b.t();

	//	offset = h.n_elem / 2;
	//	// second stage
	//	concurrency::parallel_for (uword(0), a.n_cols, [&](uword j) {
	//	//for (uword j = 0 ; j < a.n_cols ; j++) {
	//		for (uword i = 0 ; i < a.n_rows ; i++) {
	//			eT conv = 0;
	//			for (uword k = 0 ; k < h.n_elem ; k++) {
	//				// bound check for column index
	//				if (j + k < offset || j + k >= b.n_rows)
	//					continue;
	//				conv += v(k) * b(j + k, i);
	//			}
	//			c(i, j) = conv;
	//		}
	//	//}
	//	});

	//	b.clear();

	//	return c;
	//}

	/**
	 *	Convolution types
	 */
	enum convolution_type : uword {full, same, valid};

	/**
	 *	@brief	An arma style intermediate interface class implementation of 2D convolution operation
	 *	@see	conv2
	 */
	class glue_conv2
	{
	public:
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

					const uword minu = (r + 1 > mb) ? r - mb + 1 : 0/*std::max(r - mb + 1, uword(0))*/;
					const uword maxu = std::min(ma - 1, r);

					const uword minv = (c + 1 > nb) ? c - nb + 1 : 0/*std::max(c - nb + 1, uword(0))*/;
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

	/**
	 *	@name Internal functions and definitions for imresize
	 *	@{
	 */

	typedef double (*kernel_func)(double);
	typedef double (*kernel_func_modified)(kernel_func, double, double);

	/// cubic interpolation kernel
	inline double cubic(double x)
	{
		double absx = std::abs(x), 
			absx2 = absx * absx, 
			absx3 = absx2 * absx;

		double f = (1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) + 
			(-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * 
			((1 < absx) & (absx <= 2));

		return f;
	}

	inline double antialias(kernel_func kernel, double scale, double x)
	{
		return scale * kernel(scale * x);
	}

	inline double noantialias(kernel_func kernel, double /*scale*/, double x)
	{
		return kernel(x);
	}

	void contribution(uword in_length, uword out_length, 
		double scale, kernel_func kernel, 
		double kernel_width, bool antialiasing, 
		arma::mat& weights, arma::mat& indices)
	{
		kernel_func_modified h = &noantialias;	// No antialiasing; use unmodified kernel.

		if (scale < 1 && antialiasing) {
			// Use a modified kernel to simultaneously interpolate and
			// antialias.
			h = &antialias;
			kernel_width /= scale;
		}

		// Output-space coordinates.
		arma::colvec x(out_length);
		uword i = 1;
		x.imbue([&]() { return i++; });

		// Input-space coordinates. Calculate the inverse mapping such that 0.5
		// in output space maps to 0.5 in input space, and 0.5+scale in output
		// space maps to 1.5 in input space.
		//cv::Mat u = x / scale + 0.5 * (1 - 1 / scale); - precision 차이 발생
		double offset = 0.5 * (1 - 1 / scale);
		arma::colvec u = x / scale + offset;

		// What is the left-most pixel that can be involved in the computation?
		arma::colvec left = arma::floor(u - kernel_width / 2);

		// What is the maximum number of pixels that can be involved in the
		// computation?  Note: it's OK to use an extra pixel here; if the
		// corresponding weights are all zero, it will be eliminated at the end
		// of this function.
		uword P = (uword)std::ceil(kernel_width) + 2;

		// The indices of the input pixels involved in computing the k-th output
		// pixel are in row k of the indices matrix.
		indices = arma::mat(out_length, P);
		for (uword j = 0 ; j < P; j++)
			indices.col(j) = left + j;

		// The weights used to compute the k-th output pixel are in row k of the
		// weights matrix.
		weights = arma::mat(out_length, P);
		for (uword j = 0 ; j < P; j++)
		//concurrency::parallel_for(uword(0), P, [&](uword j) {
			weights.col(j) = u - indices.col(j);
		//});

		for (uword j = 0 ; j < P; j++) {
		//concurrency::parallel_for(uword(0), P, [&](uword j) {
			for (uword i = 0 ; i < u.n_elem ; i++) {
				weights(i, j) = h(kernel, scale, weights(i, j));
			}
		//});
		}

		// Normalize the weights matrix so that each row sums to 1.
		for (uword i = 0 ; i < weights.n_rows ; i++)
		//concurrency::parallel_for(uword(0), weights.n_rows, [&](uword i) {
			weights.row(i) /= sum(weights.row(i));
		//});

		// Clamp out-of-range indices; has the effect of replicating end-points.
		//indices = min(max(1, indices), in_length);
		//indices.transform([&](double val) { return std::min(std::max(1.0, val), (double)in_length); });
		std::for_each(indices.begin(), indices.end(), [&](double& val) {
		//concurrency::parallel_for_each(indices.begin(), indices.end(), [&](double& val) {
			val = std::min(std::max(1.0, val), (double)in_length);
		});

		// If a column in weights is all zero, get rid of it.
		arma::uvec alive = arma::ones<arma::uvec>(weights.n_cols);
		for (uword c = 0 ; c < weights.n_cols ; c++) {
		//concurrency::parallel_for (uword(0), weights.n_cols, [&](uword c) {
			if (!arma::any(weights.col(c))) alive(c) = 0;
		//});
		}

		if (arma::all(alive) == 0) {
			alive = find(alive);
			concurrency::parallel_invoke(	// parallel task launch
				[&] {
					weights = weights.cols(alive);
				},
				[&] {
					indices = indices.cols(alive);
				}
			);
		}
	}

	template <typename eT>
	inline arma::Mat<eT> resizeAlongDim(const arma::Mat<eT> in, size_t dim, const arma::mat& weights, const arma::mat& indices)
	{
		return imresizemex<uchar>(in, weights.t(), indices.t(), dim);
	}

	template <typename eT>
	arma::Mat<eT> imresizemex(arma::Mat<eT> in, 
		arma::mat weights, 
		arma::mat indices, 
		size_t dim)	// dir 0 is row first
	{
		//arma_ext::Size<uword> size(in.n_cols, in.n_rows);
		uvec2 size;
		size << in.n_rows << in.n_cols;

		if (dim == 1)	// column first
			std::swap(size(1), size(0));

		size(0) = weights.n_cols;

		Mat<eT> out(size(0), size(1));

		if (dim == 1)
			in = in.t();

		Concurrency::parallel_for(uword(0), out.n_cols, [&](uword c) {
		//for (uword c = 0 ; c < out.n_cols ; c++) {
			eT* optr = out.colptr(c);
			for (uword r = 0 ; r < out.n_rows ; r++) {
				double value = 0;
				double* wptr = weights.colptr(r);
				double* iptr = indices.colptr(r);

				for (uword p = 0 ; p < weights.n_rows ; p++)
					value += wptr[p] * in((uword)iptr[p] - 1, c);

				optr[r] = saturate_cast<eT>(value);
			}
		//}
		});

		if (dim == 1) out = out.t();

		return out;
	}

	/**@}*/

	/**
	 *	@brief	Resize image
	 *			This is an implementation of imresize function in MATLAB.
	 *			Only support 'bicubic' interpolation
	 *	@note	imresize function's 'bicubic' interpolation in MATLAB and resize function in OpenCV use different cubic interpolation coefficients.
	 *	@see	http://www.mathworks.co.kr/kr/help/images/ref/imresize.html
	 *			http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#resize
	 */
	template <typename eT>
	arma::Mat<eT> imresize(const arma::Mat<eT>& A, uword width, uword height)
	{
		typedef Mat<eT>::size_type size_type;

		kernel_func kernel = &cubic;
		double kernel_width = 4;
		bool antialiasing = true;
		// ignore colormap_method
		// ignore dither_option
		// ignore size_dim

		//arma_ext::Size<double> scale(width / (double)A.n_cols, height / (double)A.n_rows);
		arma::vec2 scale;
		scale << height / (double)A.n_rows << width / (double)A.n_cols;

		// determine which dimension to resize first
		size_type order[2] = {0, 1};

		if (scale(1) < scale(0)) std::swap(order[0], order[1]);

		// calculate interpolation weights and indices for each dimension
		std::vector<arma::mat> weights(2);
		std::vector<arma::mat> indices(2);
		
		concurrency::parallel_invoke(
			[&] {
				contribution(A.n_rows, height, scale(0), kernel, kernel_width, antialiasing, weights[0], indices[0]);
			},
			[&] {
				contribution(A.n_cols, width, scale(1), kernel, kernel_width, antialiasing, weights[1], indices[1]);
			}
		);

		arma::Mat<eT> B = A;

		for (size_t i = 0 ; i < 2 ; i++) {
			size_t dim = order[i];
			B = resizeAlongDim(B, dim, weights[dim], indices[dim]);
		}

		// post-processing - ignore

		return B;
	}
}