#include "cudasparsemat.cuh"

CudaSquareMatrix::CudaSquareMatrix(int N, int numElements) : SquareMatrix(N, numElements) {
    this->memSize = sizeof(double) * numElements;
    this->fullMemSize = (size_t) pow(2, ceil(log2(1.0 * memSize)));

    checkError(cudaMalloc(&(this->elements), fullMemSize));
    checkError(cudaMemset(elements, 0, fullMemSize));
}

void CudaSquareMatrix::getFromFile(string fileName) {
    ifstream file(fileName, ios_base::binary);

    MatrixElement* hostElements = new MatrixElement[numElements];

    /* The data is already sorted correctly, so I can write the raw binary directly. */
    file.read((char*) hostElements, memSize);
    checkError(cudaMemcpy(elements, hostElements, memSize, cudaMemcpyHostToDevice));
    
    file.close();
    delete hostElements;
}

void CudaSquareMatrix::importElements(MatrixElement* in, int newNumElements) {
    size_t newMemSize = newNumElements * sizeof(MatrixElement);

    if (newMemSize > fullMemSize) {
        MatrixElement* newElements;
        fullMemSize = (size_t) pow(2, ceil(log2(1.0 * memSize)));

        checkError(cudaMalloc(&newElements, fullMemSize));
        checkError(cudaFree(elements));

        elements = newElements;
    }

    checkError(cudaMemset(elements, 0, fullMemSize));
    checkError(cudaMemcpy(elements, in, newMemSize, cudaMemcpyHostToDevice));

    memSize = newMemSize;
    numElements = newNumElements;
}

CudaSpecVector::CudaSpecVector(int N, int seed, double randMax, double randMin) : SpecVector(N) {
    reInitialise();

    this->threads = 256;
    this->blocks = (N + threads - 1) / (2*threads);

    this->fullMemSize = (size_t) pow(2, ceil(log2(1.0 * memSize)));
    checkError(cudaMemset(elements, 0, fullMemSize));

    if (reduceBuffer == nullptr) {
        checkError(cudaMalloc(&reduceBuffer, sizeof(double) * blocks));
    }

    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, seed);
}

CudaSpecVector::~CudaSpecVector() {
    checkError(cudaFree(elements));
    curandDestroyGenerator(generator);
}

__global__
void matMulGPU(int N, int numElements, const MatrixElement* elements, const double* in, double* out) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < numElements) {
        const MatrixElement matElement = elements[i];

        // The matrices are symmetric, so need to calculate twice,
        // once for each orientation.
        out[matElement.row] = matElement.val * in[matElement.col];
        if (matElement.row != matElement.col) {
            out[matElement.col] = matElement.val * in[matElement.row];
        }
    }
}

inline void CudaSpecVector::matMul(SquareMatrix* mat, SpecVector* out) {
    unsigned int blocks = (mat->numElements+threads-1) / threads;

    matMulGPU<<<blocks, threads>>>(mat->N, mat->numElements, mat->elements, elements, out->elements);

    checkError(cudaDeviceSynchronize());
}

__global__
void normaliseGPU(int N, double* elements, double factor) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        elements[i] *= factor;
    }
}

void CudaSpecVector::normalise(double factor) {
    unsigned int blocks = (N+threads-1) / threads;

    normaliseGPU<<<blocks, threads>>>(N, elements, factor);

    checkError(cudaDeviceSynchronize());
}

template <unsigned int blockSize>
__device__
void warp_reduce(volatile double *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__device__
inline void reduce_shared(volatile double* sharedData, volatile double* out, unsigned int threadId, unsigned int blockId) {
    if (blockSize >= 512) { if (threadId < 256) { sharedData[threadId] += sharedData[threadId + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (threadId < 128) { sharedData[threadId] += sharedData[threadId + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (threadId < 64) { sharedData[threadId] += sharedData[threadId + 64]; } __syncthreads(); }

    if (tid < 32) warp_reduce(sharedData, threadId);

    if (tid == 0) out[blockId] = sharedData[0];
}

template <unsigned int blockSize>
__global__
void reduce_dot(double* a, double* b, double* out, unsigned int n) {
    extern __shared__ double sharedData[];

    unsigned int threadId = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;

    if (i < n) {
        sharedData[threadId] = a[i] * b[i] + a[i+blockSize] * b[i+blockSize];
    } else {
        sharedData[threadId] = 0;
    }
    __syncthreads();

    reduce_shared(sharedData, out, threadId, blockIdx.x);
}

template <unsigned int blockSize>
__global__
void reduce_sum(double *in, double *out, unsigned int n) {
    extern __shared__ double sharedData[];

    unsigned int threadId = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadId;

    if (i < n) {
        sharedData[threadId] = in[i] + in[i+blockSize];
    } else {
        sharedData[threadId] = 0;
    }
    __syncthreads();

    reduce_shared(sharedData, out, threadId, blockIdx.x);
}

inline void selectReduceSum(
        unsigned int blocks, unsigned int threads, size_t memSize,
        double* in, double* out, unsigned int n) {
    switch (threads) {
        case 512:
            reduce_sum<512><<< blocks, threads, memSize >>>(in, out, n); break;
        case 256:
            reduce_sum<256><<< blocks, threads, memSize >>>(in, out, n); break;
        case 128:
            reduce_sum<128><<< blocks, threads, memSize >>>(in, out, n); break;
        case 64:
            reduce_sum< 64><<< blocks, threads, memSize >>>(in, out, n); break;
        case 32:
            reduce_sum< 32><<< blocks, threads, memSize >>>(in, out, n); break;
        case 16:
            reduce_sum< 16><<< blocks, threads, memSize >>>(in, out, n); break;
        case 8:
            reduce_sum<  8><<< blocks, threads, memSize >>>(in, out, n); break;
        case 4:
            reduce_sum<  4><<< blocks, threads, memSize >>>(in, out, n); break;
        case 2:
            reduce_sum<  2><<< blocks, threads, memSize >>>(in, out, n); break;
        case 1:
            reduce_sum<  1><<< blocks, threads, memSize >>>(in, out, n); break;
    }
}

inline void selectReduceDot(
        unsigned int blocks, unsigned int threads, size_t memSize,
        double* inA, double* inB, double* out, unsigned int n) {
    switch (threads) {
        case 512:
            reduce_dot<512><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 256:
            reduce_dot<256><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 128:
            reduce_dot<128><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 64:
            reduce_dot< 64><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 32:
            reduce_dot< 32><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 16:
            reduce_dot< 16><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 8:
            reduce_dot<  8><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 4:
            reduce_dot<  4><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 2:
            reduce_dot<  2><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
        case 1:
            reduce_dot<  1><<< blocks, threads, memSize >>>(inA, inB, out, n); break;
    }
}

double CudaSpecVector::dot(SpecVector* vec) {
    double value;

    unsigned int blocks = this->blocks;
    unsigned int threads = this->threads;
    size_t memSize = this->fullMemSize;
    double* inA = elements;
    double* inB = vec->elements;
    double* out = reduceBuffer;
    unsigned int n = N;

    selectReduceDot(blocks, threads, memSize, inA, inB, out, n);
    
    bool done = false;
    while (!done) {
        checkError(cudaDeviceSynchronize());

        n = blocks;

        if (n > threads) {
            blocks = n / (2*threads);
        } else {
            blocks = 1;
            threads = n / 2;
            done = true;
        }

        inA = out;
        memSize = blocks * threads * sizeof(double);

        selectReduceSum(blocks, threads, memSize, inA, out, n);
    }
    checkError(cudaDeviceSynchronize());

    checkError(cudaMemcpy(&value, reduceBuffer, sizeof(double), cudaMemcpyDeviceToHost));

    return value;
}

__global__
void scaleRandomNumbers(double randMin, double randMax, double* elements, int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        elements[i] = randMin + (randMax - randMin) * elements[i];
    }
}

void CudaSpecVector::generateRandomVector() {
    curandGenerateUniformDouble(generator, elements, N);

    checkError(cudaDeviceSynchronize());

    unsigned int blocks = (N+threads-1) / threads;
    scaleRandomNumbers<<<blocks, threads>>>(randMin, randMax, elements, N);

    checkError(cudaDeviceSynchronize());
}

inline void CudaSpecVector::exportElements(double* out) {
    checkError(cudaMemcpy(out, elements, memSize, cudaMemcpyDeviceToHost));
}

__global__
void zeroValues(int vectorTolerance, double* elements, int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    if (i < N) {
        double elem = elements[i];

        if (elem < vectorTolerance) elements[i] = 0;
    }
}

vector<VectorElement> CudaSpecVector::zeroVector(double vectorTolerance) {
    unsigned int blocks = (N+threads-1) / threads;
    zeroValues<<<blocks, threads>>>(vectorTolerance, elements, N);

    if (hostBuffer == nullptr) hostBuffer = (double*) malloc(fullMemSize);
    checkError(cudaMemcpy(hostBuffer, elements, fullMemSize, cudaMemcpyDeviceToHost));

    vector<VectorElement> vectorElements;
    for (int i = 0; i < N; i++) {
        double elem = hostBuffer[i];
        /* Once power iteration is finished, some elements will become zero, but not exactly zero,
           so I treat them as exactly zero if they are within some tolerance. */
        if (elem != 0) {
            vectorElements.push_back({i, elem});
        }
    }
    //printVector(vec);

    return vectorElements;
}

template <unsigned int blockSize>
__global__
void reduce_deflate(MatrixElement* in, double* vectorElements, double *out, unsigned int n) {
    extern __shared__ double sharedData[];

    unsigned int threadId = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadId;

    if (i < n) {
        MatrixElement elem = in[i];
        double ak = vectorElements[elem.row];
        double al = vectorElements[elem.col];

        if (elem.row != elem.col) {
            sharedData[threadId] = 2 * elem.val * ak * al;
        } else {
            sharedData[threadId] = elem.val * ak * al;
        }

        elem = in[i + blockSize];
        ak = vectorElements[elem.row];
        al = vectorElements[elem.col];

        if (elem.row != elem.col) {
            sharedData[threadId] += 2 * elem.val * ak * al;
        } else {
            sharedData[threadId] += elem.val * ak * al;
        }
    } else {
        sharedData[threadId] = 0;
    }
    __syncthreads();

    reduce_shared(sharedData, out, threadId, blockIdx.x);
}

inline void selectReduceDeflate(
        unsigned int blocks, unsigned int threads, size_t memSize,
        MatrixElement* in, double* vectorElements, double* out, unsigned int n) {
    switch (threads) {
        case 512:
            reduce_deflate<512><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 256:
            reduce_deflate<256><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 128:
            reduce_deflate<128><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 64:
            reduce_deflate< 64><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 32:
            reduce_deflate< 32><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 16:
            reduce_deflate< 16><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 8:
            reduce_deflate<  8><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 4:
            reduce_deflate<  4><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 2:
            reduce_deflate<  2><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
        case 1:
            reduce_deflate<  1><<< blocks, threads, memSize >>>(in, vectorElements, out, n); break;
    }
}

double CudaSpecVector::generateConstant(CudaSquareMatrix* mat) {
    double value;

    unsigned int blocks = (mat->numElements + threads - 1) / (2*threads);
    unsigned int threads = this->threads;
    size_t memSize = this->fullMemSize;
    MatrixElement* in = mat->elements;
    double* out = matReduceBuffer;
    unsigned int n = mat->numElements;

    selectReduceDeflate(blocks, threads, memSize, in, elements, out, n);
    
    bool done = false;
    while (!done) {
        checkError(cudaDeviceSynchronize());

        n = blocks;

        if (n > threads) {
            blocks = n / (2*threads);
        } else {
            blocks = 1;
            threads = n / 2;
            done = true;
        }

        memSize = blocks * threads * sizeof(double);

        selectReduceSum(blocks, threads, memSize, out, out, n);
    }
    checkError(cudaDeviceSynchronize());

    checkError(cudaMemcpy(&value, reduceBuffer, sizeof(double), cudaMemcpyDeviceToHost));

    return value;
}

void CudaSpecVector::hotellingDeflation(SquareMatrix* mat, double vectorTolerance) {
    vector<VectorElement> vectorElements = zeroVector(vectorTolerance);
    int numVectorElements = vectorElements.size();

    unordered_map<int, int> sparseRelation;

    CudaSquareMatrix* cudaMat = (CudaSquareMatrix*) mat;

    double total = generateConstant(cudaMat);

    SerialSquareMatrix serialMat(N, cudaMat->numElements);
    checkError(cudaMemcpy(serialMat.elements, cudaMat->elements, cudaMat->memSize, cudaMemcpyDeviceToHost));

    for (int i = 0; i < serialMat.numElements; i++) {
        MatrixElement elem = serialMat.getElem(i);
        double ak = elements[elem.row];
        double al = elements[elem.col];
        
        if (ak != 0 && al != 0) {
            /* I respesent matrix coordinates as a single integer by using this simple formula */
            sparseRelation[elem.row*N + elem.col] = i;
        }
    }

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
                double newVal = serialMat.getElem(k).val - row.val * col.val * total;
                if (abs(newVal) < vectorTolerance) {
                    serialMat.removeElem(k);
                } else {
                    serialMat.setElem(k, newVal);
                }
            } catch (const out_of_range& e) {
                serialMat.addElem(row.i, col.i, - row.val * col.val * total);
            }
        }
    }

    cudaMat->importElements(serialMat.elements, serialMat.numElements);
}