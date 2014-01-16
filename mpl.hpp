/**
 *	@file		mpl.hpp
 *	@brief		Metaprogramming language implementation.
 *	@author		seonho.oh@gmail.com
 *	@date		2013-11-01
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

#if __cplusplus >= 201103L || defined(_MSC_VER)
#include <type_traits>
namespace mpl = std;
#elif USE_BOOST
#include "type_traits_boost.hpp"
#else
#include "type_traits_tr1.hpp"
#endif

namespace std {

	////! Extend type_traits for template metaprogramming framework of compile-time algorithms, sequences and metafunctions.
	//namespace mpl
	//{
		//!	Defines struct for "logical or"
		template <bool _Test1, bool _Test2>
		struct or_
			: std::true_type
		{
		};

		//!	Specialization of "logical" or for false case.
		template<>
		struct or_<false, false>
			: std::false_type
		{
		};

		//!	Defines struct for "logical and"
		template <bool _Test1, bool _Test2>
		struct and_
			: std::false_type
		{
		};

		//! Specialization of "logical and" for true case.
		template <>
		struct and_<true, true>
			: std::true_type
		{
		};
	//}

}
