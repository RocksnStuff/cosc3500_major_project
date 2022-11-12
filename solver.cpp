#include "solver.hpp"

/* Functions I wrote to print vectors and matrices for debugging purposes */
/*void printVector(SpecVector* vec) {
    cout << "[";
    for (int i = 0; i < vec->N; i++) {
        cout << vec->getElem(i);
        cout << ",";
    }
    cout << "]" << endl;
}*/

/*void printMatrix(SquareMatrix* mat) {
    cout << "------------------------------" << endl;
    cout << "N: " << mat->N << endl;
    for (int i = 0; i < mat->numElements; i++) {
        MatrixElement elem = mat->getElem(i);
        cout << "(" << elem.row << "," << elem.col << "): " << elem.val << endl;
    }
    cout << "------------------------------" << endl;
}*/

/* The compare function to be used with qsort. Did not end up using this as it did not show a performance benefit. */
int compare(const void* a, const void* b) {
    return ((MatrixElement*) a)->col - ((MatrixElement*) b)->col;
}

Solver::Solver(double tolerance, double vectorTolerance, int maxIterations, double randMin, double randMax, int seed) {
    this->tolerance = tolerance;
    this->vectorTolerance = vectorTolerance;
    this->maxIterations = maxIterations;
    this->randMin = randMin;
    this->randMax = randMax;
    this->seed = seed;
}

Eigenpair* Solver::solveEigenpairs(SquareMatrix* mat, int numEigenpairs) {
    Eigenpair* out = new Eigenpair[numEigenpairs];
    int N = mat->N;
    /* Initialise both vectors. Because of how I handle memory, these only need to be allocated once. */
    SPECVECTOR_T vec = SPECVECTOR_T(N, seed, randMax, randMin);
    SPECVECTOR_T newVec = SPECVECTOR_T(N, seed, randMax, randMin);
    newVec.reset();

    /* This is commented out as it did not end up benefitting performance. */
    //qsort(mat->elements, mat->numElements, sizeof(MatrixElement), compare);

    for (int i = 0; i < numEigenpairs; i++) {
        vec.generateRandomVector();

        /* Once power iteration is finished, the eigenvector is written to the output vector. */
        powerIteration(out + i, mat, &vec, &newVec, N);

        /* Dont run hotelling deflation if this is the last eigenvector */
        if (i != numEigenpairs - 1) {
            newVec.hotellingDeflation(mat, vectorTolerance);
            newVec.reset();
        }
    }

    return out;
}

void Solver::powerIteration(Eigenpair* out, SquareMatrix* mat, SpecVector* vec0, SpecVector* vec1, int N) {
    double oldNorm = 0.0;
    double newNorm;

    for (int i = 0; i < maxIterations; i++) {
        vec0->matMul(mat, vec1);
        newNorm = vec1->dot(vec1);
        vec1->normalise( 1.0 / sqrt(newNorm) );

        if (abs(newNorm - oldNorm) / N < tolerance) {
            cout << "Iterations: ";
            cout << i + 1 << endl;
            break;
        }
        oldNorm = newNorm;

        if (i == maxIterations - 1) {
            cout << "Exhausted max iterations" << endl;
            cout << "Iterations: ";
            cout << i + 1 << endl;
            break;
        }

        //printVector(vec1);
        /* This just swaps the reference to memory so new memory does not have to be written or allocated */
        vec0->swapElements(vec1);
        vec1->reset();
    }

    vec0->reset();
    vec1->matMul(mat, vec0);
    double eigenvalue = vec1->dot(vec0) / vec1->dot(vec1);

    cout << "Eigenvalue: " << eigenvalue << endl;
    cout << "Eigenvector: ";

    out->eigenvalue = eigenvalue;
    out->eigenvector = new double[N];
    vec1->exportElements(out->eigenvector);
}