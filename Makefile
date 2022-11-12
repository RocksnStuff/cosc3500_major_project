CFLAGS = -std=c++11 -O3 -Wall
SERIAL_DEFNS = -DSQUAREMATRIX_T=SerialSquareMatrix -DSPECVECTOR_T=SerialSpecVector -DIMPLEMENTATION_INCLUDE=\"serialsparsemat.hpp\"
CUDA_DEFNS = -DSQUAREMATRIX_T=CudaSquareMatrix -DSPECVECTOR_T=CudaSpecVector -DIMPLEMENTATION_INCLUDE=\"cudasparsemat.hpp\"

default: all

all: tester_serial, generator

generator: generator.cpp
	icc generator.cpp ${CFLAGS} -o generator -mkl -qopenmp

tester_serial: solver_serial.o serialsparsemat_serial.o tester_serial.o
	g++ solver_serial.o serialsparsemat_serial.o tester_serial.o ${CFLAGS} ${SERIAL_DEFNS} -o tester_serial

tester_serial.o: tester.cpp solver.hpp sparsemat.hpp serialsparsemat.hpp
	g++ -c ${CFLAGS} ${SERIAL_DEFNS} tester.cpp

solver_serial.o: solver.cpp solver.hpp sparsemat.hpp
	g++ -c ${CFLAGS} ${SERIAL_DEFNS} solver.cpp

serialsparsemat_serial.o: sparsemat.hpp serialsparsemat.hpp serialsparsemat.cpp
	g++ -c ${CFLAGS} serialsparsemat.cpp

clean:
	rm tester.o solver.o serialsparsemat.o tester generator

.PHONY: clean default all