#pragma once

#include <tr1/type_traits>

namespace std
{
	template <bool B, class T = void> struct enable_if : std::tr1::enable_if<B, T> {};
	template <class T> struct is_unsigned : std::tr1::is_unsigned<T> {};
	template <class T> struct is_floating_point : std::tr1::is_floating_point<T> {};
	template <typename T, typename U> struct is_same : std::tr1::is_same<T, U> {};
	template <bool B, class T, class U> struct conditional : std::tr1::conditional<B, T, U> {};
	typedef std::tr1::true_type  true_type;
	typedef std::tr1::false_type false_type;
}
