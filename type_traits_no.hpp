/**
 *	@file		type_traits_no.hpp
 *	@brief		An implementation of type traits functions
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

namespace std
{
	template <bool B, class T = void> struct enable_if { typedef T type; };
	template <class T> struct enable_if<false, T> {}; 
	
	template <class T> struct is_unsigned			{ static const bool value = false; };
	template <> struct is_unsigned<unsigned char>	{ static const bool value = true; };
	template <> struct is_unsigned<unsigned short>	{ static const bool value = true; };
	template <> struct is_unsigned<unsigned int>	{ static const bool value = true; };
	template <> struct is_unsigned<unsigned long>	{ static const bool value = true; };
	
	template <class T> struct is_floating_point		{ static const bool value = false; };
	template <> struct is_floating_point<float>		{ static const bool value = true; };
	template <> struct is_floating_point<double>	{ static const bool value = true; };
	
	template <typename T, typename U> struct is_same	{ static const bool value = false; };
	template <typename T> struct is_same<T, T>			{ static const bool value = true; };
	
	template <bool B, class T, class U> struct conditional	{ };
	template <class T, class U> struct conditional<true, T, U>	{ typedef T type; };
	template <class T, class U> struct conditional<false, T, U>	{ typedef U type; };
	
	struct true_type	{ static const bool value = true; };
	struct false_type	{ static const bool value = false; };
}
