#ifndef CUDASPARSEMAT_H
#define CUDASPARSEMAT_H

#include <cmath>
#include <curand.h>

#include "sparsemat.hpp"
#include "serialsparsemat.hpp"

using namespace std;

double* reduceBuffer = nullptr;
double* matReduceBuffer = nullptr;
double* hostBuffer = nullptr;

// Helper function to check for errors
void checkError(cudaError_t e)
{
   if (e != cudaSuccess)
   {
      std::cerr << "CUDA error: " << int(e) << " : " << cudaGetErrorString(e) << '\n';
      abort();
   }
}

class CudaSquareMatrix : public SquareMatrix {
    public:
    size_t fullMemSize;
    size_t memSize;

    CudaSquareMatrix(int N, int numElements);

    void getFromFile(string fileName) override;

    void importElements(MatrixElement* in, int newNumElements);
};

class CudaSpecVector : public SpecVector {
    private:
    size_t fullMemSize;
    curandGenerator_t generator;
    double randMin;
    double randMax;
    
    public:
    unsigned int threads;
    unsigned int blocks;

    CudaSpecVector(int N, int seed, double randMax, double randMin);

    ~CudaSpecVector() override;

    inline void reset() override { cudaMemset(elements, 0, memSize); }

    inline void reInitialise() override { cudaMalloc(&elements, fullMemSize); }

    inline void matMul(SquareMatrix* mat, SpecVector* out) override;

    void normalise(double factor) override;

    double dot(SpecVector* vec) override;

    void generateRandomVector() override;

    void exportElements(double* out) override;

    vector<VectorElement> zeroVector(double vectorTolerance) override;

    double generateConstant(CudaSquareMatrix* mat);

    void hotellingDeflation(SquareMatrix* mat, double vectorTolerance) override;
};

#endif