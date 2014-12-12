﻿#include "mf_exp.hpp"
#include "vector_exp.hpp"
#include "matrix_exp.hpp"
#include <valarray>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>
#include <Eigen/SparseLU>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/triangular.hpp>

template <class T>
auto make_vector(uint num_element, T value_min, T value_max, double sparseness)
{
	std::vector<T> vec(num_element);
	sig::SimpleRandom<T> v_rand(value_min, value_max, DEBUG_MODE);
	sig::SimpleRandom<double> sp_rand(0, 1, DEBUG_MODE);

	for (auto& e : vec) {
		if (sp_rand() < sparseness) continue;
		e = v_rand();
	}

	return vec;
}

template <class T>
auto make_matrix(uint num_row, uint num_col, T value_min, T value_max, double sparseness)
{
	std::vector<std::vector<T>> mat(num_row, std::vector<T>(num_col));
	sig::SimpleRandom<T> v_rand(value_min, value_max, DEBUG_MODE);
	sig::SimpleRandom<double> sp_rand(0, 1, DEBUG_MODE);

	for (auto& vec : mat) {
		for (auto& e : vec) {
			if (sp_rand() < sparseness) continue;
			e = v_rand();
		}
	}

	return mat;
}


using EigenVec = Eigen::VectorXd;
using EigenSparseVec = Eigen::SparseVector<double>;
using UblasVec = boost::numeric::ublas::vector<double>;
using UblasMapVec = boost::numeric::ublas::mapped_vector<double>;
using UblasCompVec = boost::numeric::ublas::compressed_vector<double>;
using UblasCoordVec = boost::numeric::ublas::coordinate_vector<double>;

template <class F1, class F2>
void vector_exp_impl(F1 const& func, F2 const& print_func, std::ofstream& ofs, uint average, uint num_element, double sparseness)
{
	std::vector<double> stl(num_element);
	std::valarray<double> stl_val(num_element);
	EigenVec eigen(num_element);
	EigenSparseVec eigen_sparse(num_element);
	UblasVec ublas(num_element);
	UblasMapVec ublas_map(num_element);
	UblasCompVec ublas_comp(num_element);
	UblasCoordVec ublas_coord(num_element);

	auto src = make_vector<double>(num_element, 0, 1, sparseness);

	for (uint i = 0; i < num_element; ++i) {
		stl[i] = src[i];
		stl_val[i] = src[i];
		eigen[i] = src[i];
		ublas[i] = src[i];
		if (src[i] != 0) {
			eigen_sparse.insertBack(i) = src[i];
			ublas_map(i) = src[i];
			ublas_comp(i) = src[i];
			ublas_coord(i) = src[i];
		}
	}

	sig::array<std::vector<int64_t>, 8> time(8, std::vector<int64_t>(average, 0));

	func(time, stl, stl_val, eigen, eigen_sparse, ublas, ublas_map, ublas_comp, ublas_coord);
	
	print_func(ofs, average, num_element, sparseness);
	ofs << "stl vector:\t" << sig::average(time[0]) << "(" << std::sqrt(sig::variance(time[0])) << ")" << std::endl;
	ofs << "stl valarray:\t" << sig::average(time[1]) << "(" << std::sqrt(sig::variance(time[1])) << ")" << std::endl;
	ofs << "eigen vector:\t" << sig::average(time[2]) << "(" << std::sqrt(sig::variance(time[2])) << ")" << std::endl;
	ofs << "eigen sparse_vector:\t" << sig::average(time[3]) << "(" << std::sqrt(sig::variance(time[3])) << ")" << std::endl;
	ofs << "ublas vector:\t" << sig::average(time[4]) << "(" << std::sqrt(sig::variance(time[4])) << ")" << std::endl;
	ofs << "ublas mapped_vector:\t" << sig::average(time[5]) << "(" << std::sqrt(sig::variance(time[5])) << ")" << std::endl;
	ofs << "ublas compressed_vector:\t" << sig::average(time[6]) << "(" << std::sqrt(sig::variance(time[6])) << ")" << std::endl;
	ofs << "ublas coordinate_vector:\t" << sig::average(time[7]) << "(" << std::sqrt(sig::variance(time[7])) << ")" << std::endl;
}

void vector_random_access_exp()
{
	const uint iteration = 10000;
	const uint average = 100;
	const std::string result_pass = "./vector_random_access_exp.txt";

	auto func = [&](
		sig::array<std::vector<int64_t>, 8>& time,
		std::vector<double>& stl,
		std::valarray<double>& stl_val,
		EigenVec& eigen,
		EigenSparseVec& eigen_sparse,
		UblasVec& ublas,
		UblasMapVec& ublas_map,
		UblasCompVec& ublas_comp,
		UblasCoordVec& ublas_coord
		) 
	{
		for (uint n = 0; n < average; ++n) {
			std::cout << n << std::endl;
			time[0][n] = random_access_exp(stl, [](std::vector<double> const& v, uint i) { return v[i]; }, iteration);
			time[1][n] = random_access_exp(stl_val, [](std::valarray<double> const& v, uint i) { return v[i]; }, iteration);
			time[2][n] = random_access_exp(eigen, [](EigenVec const& v, uint i) { return v.coeff(i); }, iteration);
			time[3][n] = random_access_exp(eigen_sparse, [](EigenSparseVec const& v, uint i) { return v.coeff(i); }, iteration);
			time[4][n] = random_access_exp(ublas, [](UblasVec const& v, uint i) { return v[i]; }, iteration);
			time[5][n] = random_access_exp(ublas_map, [](UblasMapVec const& v, uint i) { return v[i]; }, iteration);	// too slow
			time[6][n] = random_access_exp(ublas_comp, [](UblasCompVec const& v, uint i) { return v[i]; }, iteration);
			time[7][n] = random_access_exp(ublas_coord, [](UblasCoordVec const& v, uint i) { return v[i]; }, iteration);
		}
	};

	auto print_func = [iteration](std::ofstream& ofs, uint average, uint num_element, double sparseness) {
		ofs << "\n vector random access time (μs)" << std::endl;
		ofs << "sparseness: " << sparseness << ", iteration: " << iteration << ", average: " << average << std::endl;
	};

	std::cout << "vector_random_access_exp" << std::endl;

	std::ofstream ofs(result_pass, std::ios::app);
	vector_exp_impl(func, print_func, ofs, average, 10000, 0.1);
	vector_exp_impl(func, print_func, ofs, average, 10000, 0.9);
}

void vector_iteration_exp()
{
	const uint num_element = 100000;
	const uint average = 100;
	const std::string result_pass = "./vector_iteration_exp.txt";

	auto func = [&](
		sig::array<std::vector<int64_t>, 8>& time,
		std::vector<double>& stl,
		std::valarray<double>& stl_val,
		EigenVec& eigen,
		EigenSparseVec& eigen_sparse,
		UblasVec& ublas,
		UblasMapVec& ublas_map,
		UblasCompVec& ublas_comp,
		UblasCoordVec& ublas_coord
		)
	{
		for (uint n = 0; n < average; ++n) {
			std::cout << n << std::endl;
			time[0][n] = iteration_exp(stl, [](std::vector<double> const& v) {
				double d = 0;
				for(auto const& e : v) d += e; 
			});
			time[1][n] = iteration_exp(stl_val, [](std::valarray<double> const& v) {
				double d = 0;
				for(auto const& e : v) d += e; 
			});
			time[2][n] = iteration_exp(eigen, [](EigenVec const& v) {
				double d = 0;
				for (uint i = 0; i < v.size(); ++i) d += v[i];
			});
			time[3][n] = iteration_exp(eigen_sparse, [](EigenSparseVec const& v) {
				double d = 0;
				EigenSparseVec::InnerIterator it(v);
				for (; it; ++it) d += it.value();
			});
			time[4][n] = iteration_exp(ublas, [](UblasVec const& v) {
				double d = 0;
				for (auto const& e : v) d += e;
			});
			time[5][n] = iteration_exp(ublas_map, [](UblasVec const& v) {
				double d = 0;
				for (auto const& e : v) d += e;
			});
			time[6][n] = iteration_exp(ublas_comp, [](UblasCompVec const& v) {
				double d = 0;
				for (auto const& e : v) d += e;
			});
			time[7][n] = iteration_exp(ublas_coord, [](UblasCoordVec const& v) {
				double d = 0;
				for (auto const& e : v) d += e;
			});
		}
	};

	auto print_func = [](std::ofstream& ofs, uint average, uint num_element, double sparseness) {
		ofs << "\n vector iteration time (ns)" << std::endl;
		ofs << "sparseness: " << sparseness << ", number of element: " << num_element << ", average: " << average << std::endl;
	};
	
	std::cout << "vector_iteration_exp" << std::endl;

	std::ofstream ofs(result_pass, std::ios::app);
	vector_exp_impl(func, print_func, ofs, average, num_element, 0.1);
	vector_exp_impl(func, print_func, ofs, average, num_element, 0.9);
}

void vector_inner_prod_exp()
{
	const uint num_element = 100000;
	const uint average = 100;
	const std::string result_pass = "./vector_inner_prod_exp.txt";

	using boost::numeric::ublas::inner_prod;

	auto func = [&](
		sig::array<std::vector<int64_t>, 8>& time,
		std::vector<double>& stl,
		std::valarray<double>& stl_val,
		EigenVec& eigen,
		EigenSparseVec& eigen_sparse,
		UblasVec& ublas,
		UblasMapVec& ublas_map,
		UblasCompVec& ublas_comp,
		UblasCoordVec& ublas_coord
		)
	{
		for (uint n = 0; n < average; ++n) {
			std::cout << n << std::endl;
			time[0][n] = iteration_exp(stl, [](std::vector<double> const& v) {
				double d = 0;
				for (uint i = 0; i < v.size(); ++i) d += (v[i] * v[i]);
				//using sig::operator*;
				//d = v * v;
			});
			time[1][n] = iteration_exp(stl_val, [](std::valarray<double> const& v) {
				double d = 0;
				for (uint i = 0; i < v.size(); ++i) d += v[i] * v[i];
				//double d = (v * v).sum();
			});
			time[2][n] = iteration_exp(eigen, [](EigenVec const& v) {
				double d = v.dot(v);
			});
			time[3][n] = iteration_exp(eigen_sparse, [](EigenSparseVec const& v) {
				double d = 0;
				EigenSparseVec::InnerIterator it(v);
				for (; it; ++it) d += it.value();
			});
			time[4][n] = iteration_exp(ublas, [](UblasVec const& v) {
				double d = inner_prod(v, v);
			});
			time[5][n] = iteration_exp(ublas_map, [](UblasVec const& v) {
				double d = inner_prod(v, v);
			});
			time[6][n] = iteration_exp(ublas_comp, [](UblasCompVec const& v) {
				double d = inner_prod(v, v);
			});
			time[7][n] = iteration_exp(ublas_coord, [](UblasCoordVec const& v) {
				double d = inner_prod(v, v);
			});
		}
	};

	auto print_func = [](std::ofstream& ofs, uint average, uint num_element, double sparseness) {
		ofs << "\n vector iteration time (ns)" << std::endl;
		ofs << "sparseness: " << sparseness << ", number of element: " << num_element << ", average: " << average << std::endl;
	};

	std::cout << "vector_inner_prod_exp" << std::endl;

	std::ofstream ofs(result_pass, std::ios::app);
	vector_exp_impl(func, print_func, ofs, average, num_element, 0.1);
	vector_exp_impl(func, print_func, ofs, average, num_element, 0.9);
}


using EigenMat = Eigen::MatrixXd;
using EigenSparseMat = Eigen::SparseMatrix<double>;
using UblasMat = boost::numeric::ublas::matrix<double>;
using UblasMapMat = boost::numeric::ublas::mapped_matrix<double>;
using UblasCompMat = boost::numeric::ublas::compressed_matrix<double>;
using UblasCoordMat = boost::numeric::ublas::coordinate_matrix<double>;

template <class F1, class F2>
void matrix_exp_impl(F1 const& func, F2 const& print_func, std::ofstream& ofs, uint average, uint num_element, double sparseness)
{
	EigenMat eigen(num_element, num_element);
	EigenSparseMat eigen_sparse(num_element, num_element);
	UblasMat ublas(num_element, num_element);
	UblasMapMat ublas_map(num_element, num_element);
	UblasCompMat ublas_comp(num_element, num_element);
	UblasCoordMat ublas_coord(num_element, num_element);

	auto src = make_matrix<double>(num_element, num_element, 0, 1, sparseness);

	for (uint i = 0; i < num_element; ++i) {
		for (uint j = 0; j < num_element; ++j) {
			eigen(i, j) = src[i][j];
			ublas(i, j) = src[i][j];
			if (src[i][j] != 0) {
				eigen_sparse.insert(i, j) = src[i][j];
				ublas_map(i, j) = src[i][j];
				ublas_comp(i, j) = src[i][j];
				ublas_coord(i, j) = src[i][j];
			}
		}
	}

	sig::array<std::vector<int64_t>, 6> time(6, std::vector<int64_t>(average));

	func(time, eigen, eigen_sparse, ublas, ublas_map, ublas_comp, ublas_coord);

	print_func(ofs, average, num_element, sparseness);
	ofs << "eigen matrix:\t" << sig::average(time[0]) << "(" << std::sqrt(sig::variance(time[0])) << ")" << std::endl;
	ofs << "eigen sparse_matrix:\t" << sig::average(time[1]) << "(" << std::sqrt(sig::variance(time[1])) << ")" << std::endl;
	ofs << "ublas matrix:\t" << sig::average(time[2]) << "(" << std::sqrt(sig::variance(time[2])) << ")" << std::endl;
	ofs << "ublas mapped_matrix:\t" << sig::average(time[3]) << "(" << std::sqrt(sig::variance(time[3])) << ")" << std::endl;
	ofs << "ublas compressed_matrix:\t" << sig::average(time[4]) << "(" << std::sqrt(sig::variance(time[4])) << ")" << std::endl;
	ofs << "ublas coordinate_matrix:\t" << sig::average(time[5]) << "(" << std::sqrt(sig::variance(time[5])) << ")" << std::endl;
}

void matrix_random_access_exp()
{
	const uint iteration = 10000;
	const uint average = 100;
	const uint ne = 100;
	const std::string result_pass = "./matrix_random_access_exp.txt";

	auto func = [&](
		sig::array<std::vector<int64_t>, 6>& time,
		EigenMat& eigen,
		EigenSparseMat& eigen_sparse,
		UblasMat& ublas,
		UblasMapMat& ublas_map,
		UblasCompMat& ublas_comp,
		UblasCoordMat& ublas_coord
		)
	{
		for (uint n = 0; n < average; ++n) {
			std::cout << n << std::endl;
			time[0][n] = random_access_exp(eigen, ne, ne, [](EigenMat const& m, uint i, uint j) { return m.coeff(i, j); }, iteration);
			time[1][n] = random_access_exp(eigen_sparse, ne, ne, [](EigenSparseMat const& m, uint i, uint j) { return m.coeff(i, j); }, iteration);
			time[2][n] = random_access_exp(ublas, ne, ne, [](UblasMat const& m, uint i, uint j) { return m(i, j); }, iteration);
			time[3][n] = random_access_exp(ublas_map, ne, ne, [](UblasMat const& m, uint i, uint j) { return m(i, j); }, iteration);	// too slow
			time[4][n] = random_access_exp(ublas_comp, ne, ne, [](UblasCompMat const& m, uint i, uint j) { return m(i, j); }, iteration);
			time[5][n] = random_access_exp(ublas_coord, ne, ne, [](UblasCoordMat const& m, uint i, uint j) { return m(i, j); }, iteration);
		}
	};

	auto print_func = [iteration](std::ofstream& ofs, uint average, uint num_element, double sparseness) {
		ofs << "\n matrix access time (μs)" << std::endl;
		ofs << "sparseness: " << sparseness << ", iteration: " << iteration << ", average: " << average << std::endl;
	};

	std::cout << "matrix_random_access_exp" << std::endl;

	std::ofstream ofs(result_pass, std::ios::app);
	matrix_exp_impl(func, print_func, ofs, average, ne, 0.1);
	matrix_exp_impl(func, print_func, ofs, average, ne, 0.9);
}

void matrix_prod_exp()
{
	const uint num_element = 100;
	const uint average = 100;
	const std::string result_pass = "./matrix_prod_exp.txt";

	using boost::numeric::ublas::prod;

	auto func = [&](
		sig::array<std::vector<int64_t>, 6>& time,
		EigenMat& eigen,
		EigenSparseMat& eigen_sparse,
		UblasMat& ublas,
		UblasMapMat& ublas_map,
		UblasCompMat& ublas_comp,
		UblasCoordMat& ublas_coord
		)
	{
		for (uint n = 0; n < average; ++n) {
			std::cout << n << std::endl;
			time[0][n] = mprod_exp(eigen, [](EigenMat const& m) {
				EigenMat tm = m * m;
			});
			time[1][n] = mprod_exp(eigen_sparse, [](EigenSparseMat const& m) {
				EigenSparseMat tm = m * m;
			});
			time[2][n] = mprod_exp(ublas, [](UblasMat const& m) {
				UblasMat tm = prod(m, m);
			});
			time[3][n] = mprod_exp(ublas_map, [](UblasMapMat const& m) {
				UblasMapMat tm = prod(m, m);
			});
			time[4][n] = mprod_exp(ublas_comp, [](UblasCompMat const& m) {
				UblasCompMat tm = prod(m, m);
			});
			time[5][n] = mprod_exp(ublas_coord, [](UblasCoordMat const& m) {
				UblasCoordMat tm = prod(m, m);
			});
		}
	};

	std::cout << "matrix_prod_exp" << std::endl;

	auto print_func = [](std::ofstream& ofs, uint average, uint num_element, double sparseness) {
		ofs << "\n matrix prod time (μs)" << std::endl;
		ofs << "sparseness: " << sparseness << ", number of element: " << num_element << ", average: " << average << std::endl;
	};

	std::ofstream ofs(result_pass, std::ios::app);
	matrix_exp_impl(func, print_func, ofs, average, num_element, 0.1);
	matrix_exp_impl(func, print_func, ofs, average, num_element, 0.9);
}

/*
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
*/

void test() {
	const uint V_ = 10;
	const uint K_ = 5;
	Eigen::MatrixXd beta_ = Eigen::MatrixXd::Zero(K_, V_);
	std::cout << beta_ << std::endl;

		for (uint k = 0; k < K_; ++k) {
			auto beta_v = beta_.row(k);
			for (uint v = 0; v < V_; ++v) {
				beta_v.coeffRef(v) = k * v;
			}
		}

	std::cout << std::endl << beta_.coeff(1,2) << std::endl;

	std::cout << std::endl;
	std::cout << beta_ << std::endl;

	EigenSparseVec es(10);
	es.insertBack(1) = 1;
	es.insertBack(3) = 3;
	es.insertBack(10) = 10;

	double t1 = es.coeff(1);
	double t2 = es.coeff(2);
	double t3 = es.coeff(3);

	EigenSparseVec::InnerIterator it(es);
	for (; it; ++it) {
		std::cout << "row:" << it.row() << " col:" << it.col() << " index:" << it.index() << " value:" << it.value() << std::endl;
	}
}

int main()
{
	//test();
	
	vector_random_access_exp();
	vector_iteration_exp();
	vector_inner_prod_exp();

	matrix_random_access_exp();
	matrix_prod_exp();
	//matrix_LU_decomposition_exp();

	//matrix_factorization_exp();

	return 0;
}
