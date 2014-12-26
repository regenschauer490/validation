#pragma once

#define _SCL_SECURE_NO_WARNINGS
#define NDEBUG
#define BOOST_UBLAS_NDEBUG
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "SigUtil/lib/tools/random.hpp"
#include "SigUtil/lib/tools/time_watch.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include <fstream>

namespace ac2014
{
#if SIG_MSVC_ENV
using sig::uint;
#endif

const bool DEBUG_MODE = 0;
}