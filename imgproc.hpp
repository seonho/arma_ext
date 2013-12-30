/**
 *	@file		imgproc.hpp
 *	@brief		Image processing functions
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

#ifdef _MSC_VER

#include <ppl.h>

#if	(_MSC_VER <= 1600)
/// From Microsoft Visual Studio 2012, Concurrency namespace has been changed to concurrency.
/// For compatibility, namespace alias is used
namespace concurrency = Concurrency;
#endif
#endif

//#ifndef uchar
typedef unsigned char uchar;
//#endif

namespace arma_ext
{
	using namespace arma;

	//!	@defgroup	imgproc	Image Processing
	//!	@brief		Image processing functions.
	//!	@{

	/**
	 *	@brief	Template function for accurate conversion from one primitive to another
	 *			The functions saturate_cast resemble the standard C++ cast operations, such as static_cast<T>()
	 *			and others. They perform an efficient and accurate conversion from one primitive type to another.
	 *			saturate in the name means that when the input value v is out of the range of the target type,
	 *			the result is not formed just by taking low bits of the input, but instead the value is clipped.
	 *	@note	This partial implementation is taken from OpenCV. Only support double to unsigned char
	 */
	template <typename T1, typename T2>
	static inline T1 saturate_cast(const T2& v) { return T1(v); }

	template <typename T2>
	static inline unsigned char saturate_cast(const T2& v)
	{
		if (std::is_floating_point<T2>::value) {
			T2 rv = round(v);
			return (unsigned char)((unsigned)rv <= std::numeric_limits<unsigned char>::max() ? rv : rv > 0 ? std::numeric_limits<unsigned char>::max() : 0);
		}

		return (unsigned char)((unsigned)v <= std::numeric_limits<unsigned char>::max() ? v : v > 0 ? std::numeric_limits<unsigned char>::max() : 0);
	}

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

#ifndef DOXYGEN

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
		//cv::Mat u = x / scale + 0.5 * (1 - 1 / scale); - precision ���� �߻�
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
			//concurrency::parallel_invoke(	// parallel task launch
			//	[&] {
					weights = weights.cols(alive);
			//	},
			//	[&] {
					indices = indices.cols(alive);
			//	}
			//);
		}
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
#ifdef _MSC_VER
		Concurrency::parallel_for(uword(0), out.n_cols, [&](uword c) {
#else
		for (uword c = 0 ; c < out.n_cols ; c++) {
#endif
			eT* optr = out.colptr(c);
			for (uword r = 0 ; r < out.n_rows ; r++) {
				double value = 0;
				double* wptr = weights.colptr(r);
				double* iptr = indices.colptr(r);

				for (uword p = 0 ; p < weights.n_rows ; p++)
					value += wptr[p] * in((uword)iptr[p] - 1, c);

				optr[r] = saturate_cast<eT>(value);
			}
#ifdef _MSC_VER
		});
#else
        }
#endif

		if (dim == 1) out = out.t();

		return out;
	}

#endif
                                  
    template <typename eT>
    inline arma::Mat<eT> resizeAlongDim(const arma::Mat<eT> in, size_t dim, const arma::mat& weights, const arma::mat& indices)
    {
        return imresizemex<uchar>(in, weights.t(), indices.t(), dim);
    }

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
		typedef typename Mat<eT>::size_type size_type;

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
        
#ifdef _MSC_VER
		concurrency::parallel_invoke(
			[&] {
#endif
				contribution(A.n_rows, height, scale(0), kernel, kernel_width, antialiasing, weights[0], indices[0]);
#ifdef _MSC_VER
			},
			[&] {
#endif
				contribution(A.n_cols, width, scale(1), kernel, kernel_width, antialiasing, weights[1], indices[1]);
#ifdef _MSC_VER
			}
		);
#endif

		arma::Mat<eT> B = A;

		for (size_t i = 0 ; i < 2 ; i++) {
			size_t dim = order[i];
			B = resizeAlongDim(B, dim, weights[dim], indices[dim]);
		}

		// post-processing - ignore

		return B;
	}

	//! Padding method
	enum pad_method : uword  {
		constant,		//! Pad array with constant value.
		circular,		//! Pad with circular repetition of elements within the dimension.
		replicate,		//! Pad by repeating border elements of array.
		symmetric		//! Pad array with mirror reflections of itself.
	};

	//! Padding direction
	enum pad_direction : uword {
		both,		//! Pads before the first element and after the last array element along each dimension.
		pre,		//! Pad after the last array element along each dimension.
		post		//! Pad before the first element along each dimension.
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
		typedef typename T::elem_type elem_type;
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
		typedef typename T::elem_type elem_type;
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

#ifndef DOXYGEN

	/// internal function
	template <typename T>
	T constantpad(const T& A, uword rows, uword cols, typename T::elem_type padval, pad_direction direction)
	{
		typedef typename T::elem_type elem_type;

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
        default:
            throw std::invalid_argument("method should be one of pad_method!");
		}

		return arma::field<arma::uvec>();
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
				indices(k) = arma::conv_to<uvec>::from(mod(colon<ivec>(-p, M - 1), M));
				break;
			case post:
				indices(k) = arma::conv_to<uvec>::from(mod(colon<ivec>(0, M + p - 1), M));
				break;
			case both:
				indices(k) = arma::conv_to<uvec>::from(mod(colon<ivec>(-p, M + p - 1), M));
				break;
            default:
                throw std::invalid_argument("direction should be one of pad_direction");
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

			arma::uvec dimNum = arma::conv_to<uvec>::from(arma::join_cols(colon<ivec>(0, M - 1), colon<ivec>(M - 1, -1, 0)));

			switch (direction) {
			case pre:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(colon<ivec>(-p, M - 1), 2 * M)));
				break;
			case post:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(colon<ivec>(0, M + p - 1), 2 * M)));
				break;
			case both:
				indices(k) = dimNum.elem(arma::conv_to<uvec>::from(mod(colon<ivec>(-p, M + p - 1), 2 * M)));
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
				indices(k) = join_cols(zeros<uvec>(p, 1), colon<uvec>(0, M - 1));
				break;
			case post:
				indices(k) = join_cols(colon<uvec>(0, M - 1), ones<uvec>(rows, 1) * (M - 1));
				break;
			case both:
				indices(k) = join_cols(join_cols(zeros<uvec>(p, 1), colon<uvec>(0, M - 1)), ones<uvec>(p, 1) * (M - 1));
				break;
			}
		}

		return indices;
	}

#endif

	//!	@}
}