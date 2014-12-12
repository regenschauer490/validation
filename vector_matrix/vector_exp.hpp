#pragma once

#include "setting.hpp"

template <class V, class F>
auto random_access_exp(V const& vec, F const& func, uint num_iteration)
{
	const uint size = vec.size();
	sig::SimpleRandom<uint> random(0, size-1, DEBUG_MODE);
	double sum = 0;

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	for (uint n = 0; n < num_iteration; ++n){
		sum += func(vec, random());
	}
	tw.save();
	//std::cout << sum;

	return tw.get_total_time<std::chrono::microseconds>();
}

template <class V, class F>
auto iteration_exp(V const& vec, F const& func)
{
	const uint size = vec.size();
	double sum = 0;

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	sum += func(vec);
	tw.save();

	return tw.get_total_time<std::chrono::nanoseconds>();
}

template <class V, class F>
auto inner_prod_exp(V const& vec, F const& func, uint num_iteration)
{
	const uint size = vec.size();
	double sum = 0;

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	for (uint n = 0; n < num_iteration; ++n) {
		sum += func(vec);
	}
	tw.save();

	return tw.get_total_time<std::chrono::nanoseconds>();
}