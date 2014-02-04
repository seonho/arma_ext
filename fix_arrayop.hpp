/**
 *	@file		fix_arrayop.hpp
 *	@brief		Specializes template function arma::arrayop::convert
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

#pragma once

#include <armadillo>

#ifndef DOXYGEN

namespace arma {

//! Template function specialization of convert function used in conv_to class
template<>
arma_hot inline void arrayops::convert(unsigned char* dest, const double* src, const uword n_elem)
{
	uword i, j;
	for (i = 0, j = 1 ; j < n_elem ; i+=2, j+=2) {
		dest[i] = arma_ext::saturate_cast<unsigned char, double>(src[i]);
		dest[j] = arma_ext::saturate_cast<unsigned char, double>(src[j]);
	}

	if (i < n_elem)
		dest[i] = arma_ext::saturate_cast<unsigned char>(src[i]);
}

template<>
arma_hot inline void arrayops::convert(unsigned short* dest, const double* src, const uword n_elem)
{
	uword i, j;
	for (i = 0, j = 1 ; j < n_elem ; i+=2, j+=2) {
		dest[i] = arma_ext::saturate_cast<unsigned short>(src[i]);
		dest[j] = arma_ext::saturate_cast<unsigned short>(src[j]);
	}

	if (i < n_elem)
		dest[i] = arma_ext::saturate_cast<unsigned short>(src[i]);
}
    
}

#endif
