#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <mkl.h>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

struct EdgeList {
	MKL_INT src, dst;
	double val;
};  

int n_rows_A;

sparse_matrix_t A;


double *loadXd(FILE *fp)
{
	char buf[1024];
	int n;

	while (fgets(buf, sizeof(buf), fp)) {
		if (buf[0] != '%') {
			break;
		}
	}
	sscanf(buf, "%d", &n);
	double *x = (double *)malloc(sizeof(double)*n);

	for(int i=0;i<n;i++) {
		int idx; double val;
		fscanf(fp, "%d %lf", &idx, &val);
		x[idx-1] = val;
	}
	return x;
}

void debug_csr_matrix(int n_rows, int n_cols, int nnz,
                      const int* csr_row_pointer, const int* csr_columns, const double* csr_values) {
    bool has_errors = false;

    std::cout << "Debugging CSR Matrix:\n";
    std::cout << "Number of rows: " << n_rows << "\n";
    std::cout << "Number of columns: " << n_cols << "\n";
    std::cout << "Number of nonzeros: " << nnz << "\n";

    // 1. Check Out-of-Bounds Indices in csr_columns
    for (int i = 0; i < nnz; ++i) {
        if (csr_columns[i] < 0 || csr_columns[i] >= n_cols) {
            std::cerr << "Error: csr_columns[" << i << "] = " << csr_columns[i]
                      << " is out of bounds. Valid range: [0, " << n_cols - 1 << "]\n";
            has_errors = true;
        }
    }

    // 2. Check csr_row_pointer Offsets
    if (csr_row_pointer[0] != 0) {
        std::cerr << "Error: csr_row_pointer[0] = " << csr_row_pointer[0] << " should be 0.\n";
        has_errors = true;
    }
    for (int i = 0; i <= n_rows; ++i) {
        if (csr_row_pointer[i] < 0 || csr_row_pointer[i] > nnz) {
            std::cerr << "Error: csr_row_pointer[" << i << "] = " << csr_row_pointer[i]
                      << " is out of bounds. Valid range: [0, " << nnz << "]\n";
            has_errors = true;
        }
        if (i > 0 && csr_row_pointer[i] < csr_row_pointer[i - 1]) {
            std::cerr << "Error: csr_row_pointer[" << i << "] = " << csr_row_pointer[i]
                      << " is less than csr_row_pointer[" << i - 1 << "] = " << csr_row_pointer[i - 1] << "\n";
            has_errors = true;
        }
    }

    // 3. Check for Empty Rows
    for (int i = 0; i < n_rows; ++i) {
        if (csr_row_pointer[i] == csr_row_pointer[i + 1]) {
            std::cout << "Note: Row " << i << " is empty.\n";
        }
    }

    // Final Report
    if (!has_errors) {
        std::cout << "CSR matrix passed all checks successfully.\n";
    } else {
        std::cerr << "CSR matrix failed one or more checks. Please fix the errors.\n";
    }
}


void loadTTX(FILE *fp)
{
		char buf[1024];
		int nflag, sflag;
		int pre_count=0;
		long i;
		int32_t nr, nc, ne;

		fgets(buf, 1024, fp);
		if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; 
		else sflag = 0;
		if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
		else if(strstr(buf, "complex") != NULL) nflag = -1;
		else nflag = 1;
#ifdef SYM
		sflag = 1;
#endif

		while(1) {
			pre_count++;
			fgets(buf, 1024, fp);
			if(strstr(buf, "%") == NULL) break;
		}


		sscanf(buf, "%d %d %d", &nr, &nc, &ne);

		n_rows_A = nr;

		ne *= (sflag+1);

		EdgeList *inputEdge = (EdgeList *)malloc(sizeof(EdgeList)*(ne+1));

		for(i=0;i<ne;i++) {
			fscanf(fp, "%d %d", &inputEdge[i].src, &inputEdge[i].dst);
			inputEdge[i].src--; inputEdge[i].dst--;

			if(inputEdge[i].src < 0 || inputEdge[i].src >= nr || inputEdge[i].dst < 0 || inputEdge[i].dst >= nc) {
				fprintf(stdout, "A vertex id is out of range %d %d\n", inputEdge[i].src, inputEdge[i].dst);
				exit(0);
			}

			if(nflag == 1) {
				double ftemp;
				fscanf(fp, " %lf ", &ftemp);
				inputEdge[i].val = ftemp;
			} else if(nflag == -1) { // complex
				double ftemp1, ftemp2;
				fscanf(fp, " %lf %lf ", &ftemp1, &ftemp2);
				inputEdge[i].val = ftemp1;
			}

			if(sflag == 1) {
				i++;
				inputEdge[i].src = inputEdge[i-1].dst;
				inputEdge[i].dst = inputEdge[i-1].src;
				inputEdge[i].val = inputEdge[i-1].val;
			}
		}
		std::sort(inputEdge, inputEdge+ne, [](EdgeList x, EdgeList y) {
				if(x.src < y.src) return true;
				else if(x.src > y.src) return false;
				else return (x.dst < y.dst);
				});

		EdgeList *unique_end = std::unique(inputEdge, inputEdge + ne, [](EdgeList x, EdgeList y) {
				return x.src == y.src && x.dst == y.dst;
				});
		ne = unique_end - inputEdge;

		double *csr_values = (double *)mkl_malloc(sizeof(double)*ne, 64);
		MKL_INT *csr_columns = (MKL_INT *)mkl_malloc(sizeof(MKL_INT)*ne, 64);
		MKL_INT *csr_row_pointer = (MKL_INT *)mkl_malloc(sizeof(MKL_INT)*(nr+1), 64);

		for (uint32_t i = 0; i <= nr; i++) {
			csr_row_pointer[i] = 0;
		}

		for (uint32_t i = 0; i < ne; i++) {
			csr_row_pointer[inputEdge[i].src + 1]++;
		}

		for (uint32_t i = 1; i <= nr; i++) {
			csr_row_pointer[i] += csr_row_pointer[i - 1];
		}

		for (uint32_t i = 0; i < ne; i++) {
			uint32_t src = inputEdge[i].src;
			uint32_t idx = csr_row_pointer[src]++;
			csr_columns[idx] = inputEdge[i].dst;
			csr_values[idx] = inputEdge[i].val;
		}

		for (uint32_t i = nr; i > 0; i--) {
			csr_row_pointer[i] = csr_row_pointer[i - 1];
		}
		csr_row_pointer[0] = 0;

		free(inputEdge);

		debug_csr_matrix(nr, nc, ne, csr_row_pointer, csr_columns, csr_values);


		sparse_status_t status = mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nr, nc,
				csr_row_pointer, csr_row_pointer + 1,
				csr_columns, csr_values);
		if (status != SPARSE_STATUS_SUCCESS) {
		    std::cerr << "Failed to create CSR matrix with MKL. Error code: " << status << "\n";
		    return;
		}
}

void print_mkl_matrix_stats(sparse_matrix_t A) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols, *row_start = nullptr, *row_end=nullptr, *col_indices = nullptr;
    double *values = nullptr;

    // Export the CSR representation
    sparse_status_t status = mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &row_start, &row_end, &col_indices, &values);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to export CSR matrix.\n";
        return;
    }

    // Print matrix stats
    std::cout << "Matrix Statistics:\n";
    std::cout << "Indexing: " << (indexing == SPARSE_INDEX_BASE_ZERO ? "0-based" : "1-based") << "\n";
    std::cout << "Number of rows: " << rows << "\n";
    std::cout << "Number of columns: " << cols << "\n";

    MKL_INT nnz = row_start[rows] - row_start[0];
    std::cout << "Number of nonzeros (nnz): " << nnz << "\n";
    std::cout << "Average nnz per row: " << static_cast<double>(nnz) / rows << "\n";

	// Validate MKL optimization
    sparse_status_t optimize_status = mkl_sparse_optimize(A);
    if (optimize_status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "MKL optimization failed.\n";
    } else {
        std::cout << "MKL optimization successful.\n";
    }
}

void naive_spmv_from_mkl(sparse_matrix_t A, const double* x, double* y, MKL_INT n_rows) {
    // Export the CSR matrix from the MKL matrix
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT* row_start = nullptr;
    MKL_INT* row_end = nullptr;
    MKL_INT* col_indices = nullptr;
    double* values = nullptr;

    sparse_status_t status = mkl_sparse_d_export_csr(A, &indexing, &rows, &cols, &row_start, &row_end, &col_indices, &values);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "Failed to export CSR matrix from MKL. Error code: " << status << "\n";
        return;
    }

    // Verify dimensions
    if (rows != n_rows) {
        std::cerr << "Error: Matrix row count does not match provided n_rows.\n";
        return;
    }

    // Initialize result vector
    std::fill(y, y + n_rows, 0.0);

    // Perform naive SpMV
    for (MKL_INT i = 0; i < n_rows; ++i) {
        double sum = 0.0;
        for (MKL_INT j = row_start[i]; j < row_start[i + 1]; ++j) {
            sum += values[j] * x[col_indices[j]];
        }
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/x.ttx").c_str(), "r");
  
   mkl_set_num_threads(1);

 loadTTX(fpA);
  double *x = loadXd(fpB);
  double *y = (double *)mkl_malloc(sizeof(double)*n_rows_A, 64);
  fclose(fpA);
  fclose(fpB);


  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL; 
  descr.diag = SPARSE_DIAG_NON_UNIT;



  mkl_sparse_set_mv_hint(A, SPARSE_OPERATION_NON_TRANSPOSE, descr, 1000);
  mkl_sparse_optimize(A);
  print_mkl_matrix_stats(A);


// Naive Implementation
auto start = std::chrono::high_resolution_clock::now();
naive_spmv_from_mkl(A, x, y, n_rows_A);
auto end = std::chrono::high_resolution_clock::now();
std::cout << "Naive SpMV Time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";

// MKL Implementation
start = std::chrono::high_resolution_clock::now();
mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
end = std::chrono::high_resolution_clock::now();
std::cout << "MKL SpMV Time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " us\n";


  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&x, &y, &descr]() {
    },
    [&x, &y, &descr]() {
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
    }
  );




  FILE *fpC = fopen((params.input+"/y.ttx").c_str(), "w");

  fprintf(fpC, "%%%%MatrixMarket tensor array real general\n");
  fprintf(fpC, "%d\n", n_rows_A);

  for (int k = 0; k < n_rows_A; ++k) {
	  fprintf(fpC, "%lf\n", y[k]);
  }

  fclose(fpC);

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
