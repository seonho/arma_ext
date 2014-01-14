#pragma once

#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/integral_constant.hpp>

namespace std
{
	template <bool B, class T = void> struct enable_if : boost::enable_if_c<B, T> {};
	template <class T> struct is_unsigned : boost::is_unsigned<T> {};
	template <class T> struct is_floating_point : boost::is_floating_point<T> {};
	template <typename T, typename U> struct is_same : boost::is_same<T, U> {};
	template <bool B, class T, class U> struct conditional : boost::conditional<B, T, U> {};
	typedef boost::true_type  true_type;
	typedef boost::false_type false_type;
}
