#include "serialsparsemat.hpp"

SerialSquareMatrix::SerialSquareMatrix(int N, int numElements) : SquareMatrix(N, numElements) {
    this->capacity = numElements + 20;

    this->elements = new MatrixElement[capacity];
}

inline void SerialSquareMatrix::addElem(int row, int col, double val) {
    /* If the number of elements surpasses the size allocated in memory, need to allocate more memory */
    if (numElements == capacity) {
        capacity *= 2;
        elements = (MatrixElement*) realloc(elements, sizeof(MatrixElement) * capacity);
    }

    elements[numElements++] = {row, col, val};
}

void SerialSquareMatrix::removeElem(int i) {
    /* Since elements are stored as an array, removing an element would make the memory non-continuous,
       so need to move all element after it. This is slow, but this function is rarely called. */
    for (int j = i; j < --numElements; j++) {
        elements[j] = elements[j+1];
    }
}

void SerialSquareMatrix::getFromFile(string fileName) {
    ifstream file(fileName, ios_base::binary);

    /* The data is already sorted correctly, so I can write the raw binary directly. */
    file.read((char*) elements, sizeof(MatrixElement) * numElements);

    file.close();
}

SerialSpecVector::SerialSpecVector(int N, int seed, double randMax, double randMin)
        : SpecVector(N), engine(seed), distribution(randMin, randMax) {
    reInitialise();
}

SerialSpecVector::~SerialSpecVector() {
    delete elements;
}

inline void SerialSpecVector::matMul(SquareMatrix* mat, SpecVector* out) {
    SerialSquareMatrix* serialMat = (SerialSquareMatrix*) mat;
    SerialSpecVector* serialOut = (SerialSpecVector*) out;

    for (int i = 0; i < mat->numElements; i++) {
        MatrixElement matElement = serialMat->getElem(i);

        // The matrices are symmetric, so need to calculate twice,
        // once for each orientation.
        serialOut->setElemAdd(matElement.row, matElement.val * elements[matElement.col]);
        if (matElement.row != matElement.col) {
            serialOut->setElemAdd(matElement.col, matElement.val * elements[matElement.row]);
        }
    }
}

void SerialSpecVector::normalise(double factor) {
    for (int i = N-1; i >= 0; i--) {
        elements[i] *= factor;
    }
}

double SerialSpecVector::dot(SpecVector* vec) {
    SerialSpecVector* serialVec = (SerialSpecVector*) vec;

    double total = 0;
    for (int i = 0; i < N; i++) {
        total += elements[i] * serialVec->getElem(i);
    }
    return total;
}

void SerialSpecVector::generateRandomVector() {
    for (int i = 0; i < N; i++) {
        elements[i] = distribution(engine);
    }
}

inline void SerialSpecVector::exportElements(double* out) {
    memcpy(out, elements, memSize);
}

vector<VectorElement> SerialSpecVector::zeroVector(double vectorTolerance) {
    vector<VectorElement> vectorElements;
    for (int i = 0; i < N; i++) {
        double elem = elements[i];
        /* Once power iteration is finished, some elements will become zero, but not exactly zero,
           so I treat them as exactly zero if they are within some tolerance. */
        if (abs(elem) < vectorTolerance) {
            elements[i] = 0;
        } else {
            vectorElements.push_back({i, elem});
        }
    }
    //printVector(vec);

    return vectorElements;
}

void SerialSpecVector::hotellingDeflation(SquareMatrix* mat, double vectorTolerance) {
    SerialSquareMatrix* serialMat = (SerialSquareMatrix*) mat;

    vector<VectorElement> vectorElements = zeroVector(vectorTolerance);
    int numVectorElements = vectorElements.size();

    unordered_map<int, int> sparseRelation;

    double nonDiagTotal = 0;
    double diagTotal = 0;
    for (int i = 0; i < mat->numElements; i++) {
        MatrixElement elem = serialMat->getElem(i);
        double ak = elements[elem.row];
        double al = elements[elem.col];
        
        if (ak != 0 && al != 0) {
            /* I respesent matrix coordinates as a single integer by using this simple formula */
            sparseRelation[elem.row*N + elem.col] = i;

            if (elem.row != elem.col) {
                nonDiagTotal += elem.val * ak * al;
            } else {
                diagTotal += elem.val * ak * ak;
            }
        }
    }
    /* using addition instead of multiplication for performance reasons */
    double total = nonDiagTotal + nonDiagTotal + diagTotal;

    for (int i = 0; i < numVectorElements; i++) {
        VectorElement row = vectorElements[i];
        for (int j = 0; j < numVectorElements; j++) {
            VectorElement col = vectorElements[j];
            /* if col > row then it would not be within the matrix data structure */
            if (col.i > row.i) {
                break;
            }

            /* There is no function to test if a key is in a map, so need to use try catch instead */
            try {
                int k = sparseRelation.at(row.i*N + col.i);
                double newVal = serialMat->getElem(k).val - row.val * col.val * total;
                if (abs(newVal) < vectorTolerance) {
                    serialMat->removeElem(k);
                } else {
                    serialMat->setElem(k, newVal);
                }
            } catch (const out_of_range& e) {
                serialMat->addElem(row.i, col.i, - row.val * col.val * total);
            }
        }
    }
}