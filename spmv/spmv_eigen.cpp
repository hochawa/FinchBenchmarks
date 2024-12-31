#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

int main(int argc, char **argv) {
	auto params = parse(argc, argv);

	FILE *fpA = fopen((params.input + "/A.ttx").c_str(), "r");
	FILE *fpB = fopen((params.input + "/x.ttx").c_str(), "r");

	Eigen::SparseMatrix<double> A;
	Eigen::VectorXd x;
	Eigen::VectorXd y;

	Eigen::loadMarket(A, fpA);
	Eigen::loadMarket(x, fpB);
	fclose(fpA);
	fclose(fpB);

	// Assemble output indices and numerically compute the result
	auto time = benchmark(
		[&A, &x, &y]() {
			y = A * x;
		}
	);

	FILE *fpC = fopen((params.input + "/y.ttx").c_str(), "w");

	fprintf(fpC, "%%%%MatrixMarket tensor array real general\n");
	fprintf(fpC, "%ld\n", y.size());

	for (int k = 0; k < y.size(); ++k) {
		fprintf(fpC, "%lf\n", y(k));
	}

	fclose(fpC);

	json measurements;
	measurements["time"] = time;
	measurements["memory"] = 0;
	std::ofstream measurements_file(params.output + "/measurements.json");
	measurements_file << measurements;
	measurements_file.close();

	return 0;
}
