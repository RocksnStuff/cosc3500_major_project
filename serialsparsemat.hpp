#ifndef SERIALSPARSEMAT_H
#define SERIALSPARSEMAT_H

#include <cmath>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <iostream>
#include <string.h>

#include "sparsemat.hpp"

using namespace std;

class SerialSquareMatrix : public SquareMatrix {
    private:
    int capacity;

    public:
    SerialSquareMatrix(int N, int numElements);

    /* Function inlined to improve performance */
    inline MatrixElement getElem(int i) { return elements[i]; }

    inline void setElem(int i, double val) { elements[i].val = val; }

    inline void setElemAdd(int i, double val) { elements[i].val += val; }

    inline void addElem(int row, int col, double val);

    void removeElem(int i);

    void getFromFile(string fileName) override;
};

class SerialSpecVector : public SpecVector {
    private:
    default_random_engine engine;
    uniform_real_distribution<double> distribution;

    public:
    SerialSpecVector(int N, int seed, double randMax, double randMin);

    ~SerialSpecVector();

    inline void reset() override { memset(elements, 0, memSize); }

    inline void reInitialise() override { elements = new double[N]; }

    inline double getElem(int i) { return elements[i]; }

    inline void setElem(int i, double val) { elements[i] = val; }

    inline void setElemAdd(int i, double val) { elements[i] += val; }

    inline void matMul(SquareMatrix* mat, SpecVector* out) override;

    void normalise(double factor) override;

    double dot(SpecVector* vec) override;

    void generateRandomVector() override;

    inline void exportElements(double* out) override;

    vector<VectorElement> zeroVector(double vectorTolerance) override;

    void hotellingDeflation(SquareMatrix* mat, double vectorTolerance) override;
};

#endif