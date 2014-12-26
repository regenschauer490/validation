#pragma once

#include "setting.hpp"

namespace ac2014
{
template <class M, class F>
auto random_access_exp(M const& mat, uint size1, uint size2, F const& func, uint num_iteration)
{
	sig::SimpleRandom<uint> random1(0, size1 - 1, DEBUG_MODE);
	sig::SimpleRandom<uint> random2(0, size2 - 1, DEBUG_MODE);
	double sum = 0;

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	for (uint n = 0; n < num_iteration; ++n){
		sum += func(mat, random1(), random2());
	}
	tw.save();

	return tw.get_total_time<std::chrono::microseconds>();
}

template <class M, class F>
auto mprod_exp(M const& mat, F const& func)
{
	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	func(mat);
	tw.save();

	return tw.get_total_time<std::chrono::microseconds>();
}

template <class M, class F>
auto lu_exp(M const& mat, F const& func)
{
	M tmp = mat;

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	func(tmp);
	tw.save();

	return tw.get_total_time<std::chrono::microseconds>();
}

}