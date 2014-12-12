#pragma once

#include "matrix_factorization.hpp"
#include <Eigen/Core>
#include <Eigen/SparseCore>

auto eigen_mf(std::vector<std::vector<uint>> const& ratings, uint num_factor, uint iteration)
{
	using Vector = Eigen::VectorXd;
	using Matrix = std::vector<Vector>;

	MatrixFactorization<Matrix> mf(ratings, [](uint i, uint j) { return Matrix(i, Vector(j)); }, num_factor);

	sig::TimeWatch<std::chrono::high_resolution_clock> tw;
	mf.train(iteration,
		[](int r, Vector const& lf1, Vector const& lf2) { return r - lf1.dot(lf2);  },
		[](Vector& lf1, Vector const& lf2, double e, double a, double l) { lf1 += a * (e * lf2 - l * lf1); }
	);
	tw.save();

	auto inner_prod = [](Vector const& lf1, Vector const& lf2) {
		return lf1.dot(lf2);
	};

	if (DEBUG_MODE) {
		std::cout << "\n eigen_mf estimate" << std::endl;
		for (uint u = 0; u < ratings.size(); ++u) {
			for (uint v = 0; v < ratings[0].size(); ++v) {
				std::cout << mf.estimate(u, v, inner_prod) << " ";
			}
			std::cout << std::endl;
		}
	}

	std::cout << "\n eigen_mf error: " << mf.error() << std::endl;

	return tw.get_total_time<std::chrono::milliseconds>();
}

void eigen_sparse_mf(std::vector<std::vector<int>> const& ratings, uint num_factor)
{
	using Vector = Eigen::SparseVector<double>;
	using Matrix = std::vector<Vector>;


}


void matrix_factorization_exp()
{
	const uint num_element = 100;
	const uint num_factor = std::sqrt(num_element);
	const double sparseness = 0.8;
	const uint iteration = 500;
	const uint average = 100;
	const std::string result_pass = "./matrix_factorization_exp.txt";

	auto sparse_ratings = make_matrix<uint>(num_element, num_element, 1, 5, sparseness);

	sig::array<std::vector<int64_t>, 2> time(2, std::vector<int64_t>(average));

	for (uint n = 0; n < average; ++n) {
		time[0][n] = eigen_mf(sparse_ratings, num_factor, iteration);
	}

	std::ofstream ofs(result_pass, std::ios::app);
	ofs << "\n matrix_factorization_exp time (ms)" << std::endl;
	ofs << "sparseness: " << sparseness << ", iteration: " << iteration << ", average: " << average << std::endl;
	ofs << "eigen matrix:\t" << sig::average(time[0]) << "(" << std::sqrt(sig::variance(time[0])) << ")" << std::endl;
	ofs << "eigen sparse_matrix:\t" << sig::average(time[1]) << "(" << std::sqrt(sig::variance(time[1])) << ")" << std::endl;
}

