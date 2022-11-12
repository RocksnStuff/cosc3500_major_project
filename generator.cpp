#include <mkl_spblas.h>
#include <mkl.h>
#include <iostream>
#include <fstream>
#include <random>
#include <limits.h>
#include <string.h>

using namespace std;

string DATA_DIR = "/data/s4585711/matrices/";

void print_sparse_status(sparse_status_t status) {
    switch (status)
    {
    case SPARSE_STATUS_SUCCESS:
        cout << "The operation was successful." << endl;
        break;
    case SPARSE_STATUS_NOT_INITIALIZED:
        cout << "The routine encountered an empty handle or matrix array." << endl;
        break;
    case SPARSE_STATUS_ALLOC_FAILED:
        cout << "Internal memory allocation failed." << endl;
        break;
    case SPARSE_STATUS_INVALID_VALUE:
        cout << "The input parameters contain an invalid value." << endl;
        break;
    case SPARSE_STATUS_EXECUTION_FAILED:
        cout << "Execution failed." << endl;
        break;
    case SPARSE_STATUS_INTERNAL_ERROR:
        cout << "An error in algorithm implementation occurred." << endl;
        break;
    case SPARSE_STATUS_NOT_SUPPORTED:
        cout << "The requested operation is not supported." << endl;
        break;
    default:
        break;
    }
}

void save_matrix_to_file(int N, int n, int id, double* acoo, int* rowind, int* colind) {
    struct Element {
        int row;
        int col;
        double val;
    };

    Element* elements = new Element[n];
    int numElements = 0;
    for (int i = 0; i < n; i++) {

        if (colind[i] <= rowind[i]) {
            elements[numElements++] = {rowind[i], colind[i], acoo[i]};
        }
    }

    ofstream f(DATA_DIR + "mat_" + to_string(N) + "_" + to_string(numElements) + "_" + to_string(id), ios::out | ios::binary);
    f.write((char*) elements, numElements * sizeof(Element));
    f.close();

    delete elements;
}

void generate_matrix(int N, int n, int id, default_random_engine& engine) {
    sparse_matrix_t A;

    uniform_int_distribution<int> valueDistribution(-10, 10);
    uniform_int_distribution<int> indexDistribution(0, N-1);

    int* row_indx = new int[n];
    int* col_indx = new int[n];
    double* values = new double[n];

    for (int i = 0; i < n; i++) {
        row_indx[i] = indexDistribution(engine);
        col_indx[i] = indexDistribution(engine);

        double value;
        while (true) {
            value = valueDistribution(engine);
            if (value != 0) {
                break;
            }
        }
        values[i] = value;
    }

    print_sparse_status(mkl_sparse_d_create_coo(&A, SPARSE_INDEX_BASE_ZERO, N, N, n, row_indx, col_indx, values));
    print_sparse_status(mkl_sparse_convert_csr(A, SPARSE_OPERATION_NON_TRANSPOSE, &A));

    delete row_indx;
    delete col_indx;
    delete values;

    print_sparse_status(mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, A, &A));

    int* rows_start;
    int* rows_end;
    sparse_index_base_t indexing;
    print_sparse_status(mkl_sparse_d_export_csr(A, &indexing, &N, &N, &rows_start, &rows_end, &col_indx, &values));
    int nind = rows_end[N - 1];

    int* ia = new int[N+1];
    memcpy(ia, rows_start, N * sizeof(int));
    ia[N] = nind;

    double* acoo = new double[nind];
    int* rowind = new int[nind];
    int* colind = new int[nind];
    int job[] = {0, 0, 0, 0, nind, 3};
    int nnz;
    int info;
    mkl_dcsrcoo(&job[0], &N, values, col_indx, ia, &nnz, acoo, rowind, colind, &info);
    cout << (info==0 ? "Conversion successful" : "Conversion unsuccessful") << endl;

    delete ia;
    print_sparse_status(mkl_sparse_destroy(A));

    save_matrix_to_file(N, nind, id, acoo, rowind, colind);

    delete acoo;
    delete rowind;
    delete colind;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << argv[0] << " N density numMatrices" << endl;
        return 1;
    }
    int N = stoi(argv[1]);
    int n = (int) (N*N * stod(argv[2]));
    int numMatrices = stoi(argv[3]);

    default_random_engine engine;

    for (int i = 0; i < numMatrices; i++) {
        generate_matrix(N, n, i, engine);
    }

    return 0;
}