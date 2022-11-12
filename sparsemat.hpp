#ifndef SPARSEMAT_H
#define SPARSEMAT_H

#include <cmath>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string.h>
#include <random>

using namespace std;

/* structs to store both matrix and vector elements in sparse form */
struct MatrixElement {
    int row;
    int col;
    double val;
};

struct VectorElement {
    int i;
    double val;
};

class SquareMatrix {
    public:
    int N;
    int numElements;
    /* I originally stored elements in a C++ Standard Vector, but I found a bug that I couldn't fix,
       so I implemented dynamic arrays manually. This had the added benefit of having less overhead. */
    MatrixElement* elements;

    SquareMatrix(int N, int numElements) {
        this->N = N;
        this->numElements = numElements;
    }

    virtual void getFromFile(string fileName) = 0;
};

class SpecVector {
    protected:
    double invN;
    size_t memSize;

    public:
    int N;
    double* elements;
    SpecVector(int N) {
        this->N = N;

        /* These constants are used frequently, so I store them so they do not need to be recalculated. */
        this->invN = 1 / (double) N;
        this->memSize = sizeof(double) * N;
    }

    virtual void reset() = 0;

    virtual void reInitialise() = 0;

    inline void swapElements(SpecVector* vec) {
        double* swap = elements;
        elements = vec->elements;
        vec->elements = swap;
    }

    virtual void matMul(SquareMatrix* mat, SpecVector* out) = 0;

    virtual void normalise(double factor) = 0;

    virtual double dot(SpecVector* vec) = 0;

    virtual void generateRandomVector() = 0;

    virtual void exportElements(double* out) = 0;

    virtual vector<VectorElement> zeroVector(double vectorTolerance) = 0;

    virtual void hotellingDeflation(SquareMatrix* mat, double vectorTolerance) = 0;
};

#endif