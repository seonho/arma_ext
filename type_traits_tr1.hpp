#pragma once

#include <tr1/type_traits>

namespace std
{
	template <bool B, class T = void> struct enable_if {};
	template <class T> struct enable_if<true, T> { typedef T type; };

	template <class T> struct is_unsigned : std::tr1::is_unsigned<T> {};
	template <class T> struct is_floating_point : std::tr1::is_floating_point<T> {};
	template <typename T, typename U> struct is_same : std::tr1::is_same<T, U> {};

	template <bool B, class T, class U> struct conditional {};
	template <class T, class U> struct conditional <true, T, U> { typedef T type; };
	template <class T, class U> struct conditional <false, T, U> { typedef U type; };

	typedef std::tr1::true_type  true_type;
	typedef std::tr1::false_type false_type;
}
