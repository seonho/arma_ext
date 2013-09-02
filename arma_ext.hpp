/**
 *	@file		arma_ext.hpp
 *	@brief		An implemenation of arma_ext namespace
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

#define ARMA_USE_LAPACK					// enable LAPACK
#define ARMA_USE_BLAS					// enable OpenBLAS
#define ARMA_NO_DEBUG

#include <armadillo>
#pragma comment(lib, "lapack_win32_MT")	// link LAPACK
#pragma comment(lib, "blas_win32_MT")
//#pragma comment(lib, "libopenblas")		// link OpenBLAS

/**
 *	@brief	An extension of armadillo library for MATLAB interface functions.
 *	@note	Partially implements MATLAB style function
 */
namespace arma_ext
{
	using namespace arma;
	typedef arma::uword size_type;
	
	/**
	 *	@brief	Round operation for scalar value.
	 *	@note	
	 *	@param v the given scalar value
	 *	@return	
	 */
	template <typename T>
	inline T round(const double& v)
	{
		return v < 0 ? static_cast<T>(ceil(v - 0.5)) : static_cast<T>(floor(v + 0.5));
	}

	/**
	 *	@brief	Template function for accurate conversion from one primitive to another
	 *			The functions saturate_cast resemble the standard C++ cast operations, such as static_cast<T>()
	 *			and others. They perform an efficient and accurate conversion from one primitive type to another.
	 *			saturate in the name means that when the input value v is out of the range of the target type,
	 *			the result is not formed just by taking low bits of the input, but instead the value is clipped.
	 *	@note	This partial implementation is taken from OpenCV. Only support double to unsigned char
	 */
	template <typename T1, typename T2>
	static inline T1 saturate_cast(T2 v) { return T1(v); }

	template <>
	static inline unsigned char saturate_cast(int v)
	{
		return (unsigned char)((unsigned)v <= std::numeric_limits<unsigned char>::max() ? v : v > 0 ? std::numeric_limits<unsigned char>::max() : 0);
	}

	template <>
	static inline unsigned char saturate_cast(double v)
	{
		int iv = round<int>(v);
		return (unsigned char)saturate_cast<unsigned char>(iv); 
	}
}

#include "logicalop.hpp"
#include "arrayop.hpp"
#include "imresize.hpp"
#include "random.hpp"
#include "statistics.hpp"