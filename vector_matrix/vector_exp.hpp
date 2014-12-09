#pragma once

#include "setting.hpp"

template <class C, class F>
auto random_access_exp(C const& vec, F const& access_func, uint num_iteration)
{
	const uint size = vec.size();
	sig::SimpleRandom<uint> random(0, size, DEBUG_MODE);
	double sum = 0;
	sig::TimeWatch tw;

	for (uint n = 0; n < num_iteration; ++n){
		sum += access_func(vec, random());
	}
	tw.save();

	return tw.get_total_time<std::chrono::milliseconds>();
}


