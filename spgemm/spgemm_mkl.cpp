#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <mkl.h>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include <sys/time.h>

int main(int argc, char **argv) {

 mkl_set_num_threads(1);	
  auto params = parse(argc, argv);

    // Convert Eigen matrix A to MKL format using Eigen's internal data
    const int* outerIndexPtr_A = eigen_A.outerIndexPtr();
    const int* innerIndexPtr_A = eigen_A.innerIndexPtr();
    const double* valuePtr_A = eigen_A.valuePtr();

    MKL_INT *csr_row_pointer_A = (MKL_INT *)mkl_malloc((eigen_A.rows() + 1) * sizeof(MKL_INT), 64);
    MKL_INT *csr_columns_A = (MKL_INT *)mkl_malloc(eigen_A.nonZeros() * sizeof(MKL_INT), 64);
    double *csr_values_A = (double *)mkl_malloc(eigen_A.nonZeros() * sizeof(double), 64);

    for (int i = 0; i <= eigen_A.rows(); ++i) {
        csr_row_pointer_A[i] = outerIndexPtr_A[i];
    }

    for (int i = 0; i < eigen_A.nonZeros(); ++i) {
        csr_columns_A[i] = innerIndexPtr_A[i];
        csr_values_A[i] = valuePtr_A[i];
    }

    sparse_matrix_t A;

    sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, eigen_A.rows(), eigen_A.cols(),
                                                     csr_row_pointer_A, csr_row_pointer_A + 1,
                                                     csr_columns_A, csr_values_A);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
        return -1;
    }

    // Convert Eigen matrix B to MKL format using Eigen's internal data
    const int* outerIndexPtr_B = eigen_B.outerIndexPtr();
    const int* innerIndexPtr_B = eigen_B.innerIndexPtr();
    const double* valuePtr_B = eigen_B.valuePtr();

    MKL_INT *csr_row_pointer_B = (MKL_INT *)mkl_malloc((eigen_B.rows() + 1) * sizeof(MKL_INT), 64);
    MKL_INT *csr_columns_B = (MKL_INT *)mkl_malloc(eigen_B.nonZeros() * sizeof(MKL_INT), 64);
    double *csr_values_B = (double *)mkl_malloc(eigen_B.nonZeros() * sizeof(double), 64);

    for (int i = 0; i <= eigen_B.rows(); ++i) {
        csr_row_pointer_B[i] = outerIndexPtr_B[i];
    }

    for (int i = 0; i < eigen_B.nonZeros(); ++i) {
        csr_columns_B[i] = innerIndexPtr_B[i];
        csr_values_B[i] = valuePtr_B[i];
    }

    sparse_matrix_t B;

    sparse_status_t status = mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, eigen_B.rows(), eigen_B.cols(),
                                                     csr_row_pointer_B, csr_row_pointer_B + 1,
                                                     csr_columns_B, csr_values_B);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
        return -1;
    }

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

	matrix_descr descrC;
	descrC.type = SPARSE_MATRIX_TYPE_GENERAL;

	matrix_descr descrA, descrB;
	descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
	descrB.type = SPARSE_MATRIX_TYPE_GENERAL;

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
		  [&A, &descr_A, &B, &descr_B, &C, &descr_C]() {},
		  [&A, &descr_A, &B, &descr_B, &C, &descr_C]() {

		mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A, SPARSE_OPERATION_NON_TRANSPOSE,descrB, B, SPARSE_STAGE_FULL_MULT, &C);
	mkl_sparse_order(C);
		mkl_sparse_d_export_csr(C, &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
		
			mkl_sparse_destroy(A);
			mkl_sparse_destroy(B);
			mkl_free_buffers();
    }
  );



  FILE *fpC = fopen((params.output+"/C.ttx").c_str(), "w");

// Convert MKL matrix C to Eigen format
Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_C(n_rows_A, n_cols_B);

eigen_C.resizeNonZeros(rows_start[r][n_rows_A]);

for (int i = 0; i < n_rows_A; ++i) {
		eigen_C.outerIndexPtr()[i] = rows_start[r][i];
}
eigen_C.outerIndexPtr()[n_rows_A] = rows_start[r][n_rows_A];

for (int i = 0; i < rows_start[r][n_rows_A]; ++i) {
		eigen_C.innerIndexPtr()[i] = columns[r][i];
		eigen_C.valuePtr()[i] = values[r][i];
}

// Save the Eigen matrix to MatrixMarket format
Eigen::saveMarket(eigen_C, (params.output + "/C.mtx").c_str());


  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
