#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include "matrix_factorization.hpp"
#include "vector_exp.hpp"
#include "Eigen/Core"
#include "Eigen/Sparse"


void eigen_mf(std::vector<std::vector<int>> const& ratings, uint num_factor)
{
	using Vector = Eigen::VectorXd;
	using Matrix = std::vector<Vector>;

	MatrixFactorization<Matrix> mf(ratings, [](uint i, uint j){ return Matrix(i, Vector(j)); }, num_factor);

	mf.train(1000,
		[](int r, Vector const& lf1, Vector const& lf2){ return r - lf1.dot(lf2);  },
		[](Vector& lf1, Vector const& lf2, double e, double a, double l){ lf1 += a * (e * lf2 - l * lf1); }
	);

	auto inner_prod = [](Vector const& lf1, Vector const& lf2){
		return lf1.dot(lf2);
	};

	if (DEBUG_MODE){
		std::cout << "\n estimate" << std::endl;
		for (uint u = 0; u < ratings.size(); ++u){
			for (uint v = 0; v < ratings[0].size(); ++v){
				std::cout << mf.estimate(u, v, inner_prod) << " ";
			}
			std::cout << std::endl;
		}
	}
}

void eigen_sparse_mf(std::vector<std::vector<int>> const& ratings, uint num_factor)
{
	using Vector = Eigen::SparseVector<double>;
	using Matrix = std::vector<Vector>;


}


void vector_access_exp(uint num_element, uint iteration)
{
	std::vector<double> stl(num_element);
	Eigen::VectorXd eigen(num_element);
	Eigen::SparseVector<double> eigen_sparse(num_element);

	auto time_stl = random_access_exp(stl, [](std::vector<double> const& v, uint i){ return v[i]; }, iteration);
	auto time_eigen = random_access_exp(eigen, [](Eigen::VectorXd const& v, uint i){ return v[i]; }, iteration);
	auto time_eigen_sparse = random_access_exp(eigen_sparse, [](Eigen::SparseVector<double> const& v, uint i){ return [i]; }, iteration);
}

int main()
{
	const uint N = 10000;
	const uint num_factor = std::sqrt(N);
	sig::SimpleRandom<int> random(1, 5, DEBUG_MODE);

	std::vector<std::vector<int>> test_ratings{
			{ 1, 0, 3, 2, 0 },
			{ 0, 4, 0, 3, 0 },
			{ 3, 2, 0, 0, 1 },
			{ 0, 0, 5, 0, 4 }
	};

	std::vector<std::vector<int>> sparse_ratings(N, std::vector<int>(N));

	uint ct = 0;
	for (auto& vec : sparse_ratings){
		for (auto& e : vec){
			if (random() < 5) continue;	// 80% 
			e = random();
			++ct;
		}
	}
	std::cout << ct / static_cast<double>(N*N) << std::endl;

	sig::TimeWatch tw1;
	eigen_mf(sparse_ratings, num_factor);
	tw1.save();
	std::cout << "\n total time : "<< tw1.get_total_time<std::chrono::milliseconds>() << " ms" << std::endl;

	return 0;
}