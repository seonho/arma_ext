/**
 *	@file		numint.hpp
 *	@brief		Numerical integration functions
 *	@author		seonho.oh@gmail.com
 *	@date		2013-07-01
 *	@version	1.0
 *
 *	@section	LICENSE
 *
 *		Copyright (c) 2007-2014, Seonho Oh
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

namespace arma_ext
{
	using namespace arma;

	//!	@addtogroup	numintdiff
	//!	@{

	//!	@brief	An arma style intermediate interface class for trapezoidal integration.
	//!	@see	trapz
	class glue_trapz
	{
	public:
		//!	A rudimentary implementation of the trapezoidal integration
		template <typename T1, typename T2>
		inline static void apply(Mat<typename T1::elem_type>&out, const Glue<T1, T2, glue_trapz>& X)
		{
			arma_extra_debug_sigprint();

			typedef typename T1::elem_type elem_type;

			const Col<elem_type>& x = X.A;
			const Mat<elem_type>& y = X.B;
			const size_type dim = X.aux_uword;

			size_type m = x.n_elem;
			arma_debug_check((m != size(y, dim)), "trapz(): given object size doesn't match");

			// Trapezoid sum computed with vector-matrix multiply.
			switch (dim) {
			case 0:
				out = diff(x).t() * ((y.rows(0, m - 2) + y.rows(1, m - 1)) / 2);
				break;
			case 1:
				out = diff(x).t() * ((y.cols(0, m - 2) + y.cols(1, m - 1)) / 2).t();
				break;
			default:
				throw std::invalid_argument("dim should be 0 or 1");
			}
		}
	};

	/**
	 *	@brief	Trapezoidal numerical integration
	 *	@param X The x-axis values
	 *	@param Y The y-axis values to be used for numerical integration.
	 *	@param dim	The dimension to be integrated across.
	 *	@return	Trapezoidal numerical integration of @c y along @c x.
	 *	@see	http://www.mathworks.co.kr/kr/help/matlab/ref/trapz.html
	 *	@note	This function is preliminary; it is not yet fully optimized.
	 */
	template <typename T1, typename T2>
	inline const Glue<T1, T2, glue_trapz> trapz(const Base<typename T1::elem_type, T1>& X, const Base<typename T1::elem_type, T2>& Y, const size_type dim = 0)
	{
		arma_extra_debug_sigprint();

		return Glue<T1, T2, glue_trapz>(X.get_ref(), Y.get_ref(), dim);
	}

	/**
	 *	@brief	Computes an approximation of the integral of @c Y via the
	 *			trapezoidal method (with unit spacing).<br> To compute the
	 *			integral for spacing different from one, multiply return value 
	 *			by the spacing increment.
	 *	@param Y The values to be used for numerical integration.
	 *	@return	Trapezoidal numerical integration of @c Y.
	 */
	template <typename T1>
	inline const Mat<typename T1::elem_type> trapz(const Base<typename T1::elem_type, T1>& Y)
	{
		arma_extra_debug_sigprint();

		return trapz(colon<Col<typename T1::elem_type> >(1, Y.get_ref().n_rows), Y);
	}

	//!	@}
}