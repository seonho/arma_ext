/**
 *	@file		fix_fn_accu.hpp
 *	@brief		Specializes template function arma::accu_proxy_linear
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

/// Template function specialization for double accumulation
template <>
arma_hot
inline
double
accu_proxy_linear(const Proxy<subview_row<double> >& P)
{
	typedef subview_row<double>::elem_type      eT;
	typedef Proxy<subview_row<double> >::ea_type ea_type;

	ea_type A      = P.get_ea();
	const uword   n_elem = P.get_n_elem();

	eT val = eT(0);

	for(uword i = 0 ; i < n_elem ; i++)
		val += A[i];
		
	return val;
}

//template<>
//inline
//double
//arma::op_mean::direct_mean(const Mat<double>& X, const uword row)
//{
//	arma_extra_debug_sigprint();
//
//	typedef get_pod_type<double>::result T;
//
//	const uword X_n_cols = X.n_cols;
//
//	double val = double(0);
//
//	for(uword i = 0; i < X_n_cols; i++)
//		val += X.at(row,i);
//
//	const double result = val / T(X_n_cols);
//
//	return arma_isfinite(result) ? result : op_mean::direct_mean_robust(X, row);
//}
}

#endif