#pragma once

namespace std
{
	//!	struct for logical or
	template <bool _Test1, bool _Test2>
	struct or_
		: true_type
	{
	};

	template<>
	struct or_<false, false>
		: false_type
	{
	};

	//!	struct for logical and
	template <bool _Test1, bool _Test2>
	struct and_
		: false_type
	{
	};

	template <>
	struct and_<true, true>
		: true_type
	{
	};
}