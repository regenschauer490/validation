#pragma once

#include "SigUtil/lib/tools/random.hpp"
#include "SigUtil/lib/tools/time_watch.hpp"
#include "SigUtil/lib/calculation/basic_statistics.hpp"
#include <fstream>

#define NDEBUG
#define _DEBUG
#define _SCL_SECURE_NO_WARNINGS
#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

using uint = std::size_t;

const bool DEBUG_MODE = 0;
