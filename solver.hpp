#ifndef SOLVER_H
#define SOLVER_H

#include <cstdlib>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <random>

#include IMPLEMENTATION_INCLUDE

using namespace std;

struct Eigenpair {
    double eigenvalue;
    double* eigenvector;
};

class Solver {
    private:
    double tolerance;
    double vectorTolerance;
    int maxIterations;
    int seed;
    double randMin;
    double randMax;

    void powerIteration(Eigenpair* out, SquareMatrix* mat, SpecVector* vec0, SpecVector* vec1, int N);
    
    public:
    Solver(double tolerance, double vectorTolerance, int maxIterations, double randMin, double randMax, int seed);

    Eigenpair* solveEigenpairs(SquareMatrix* mat, int numEigenpairs);
};

#endif