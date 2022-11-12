#include <iostream>
#include <time.h>
#include <chrono>
#include <vector>
#include <dirent.h>
#include <string>
#include <fstream>

#include "solver.hpp"
#include IMPLEMENTATION_INCLUDE

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 9) {
        cerr << "Not enough args" << endl;
        cerr << argv[0] << " N numTrials numEigenpairs tolerance vectorTolerance maxIterations randMin randMax [seed]" << endl;
        return -1;
    }

    int N = atoi(argv[1]);
    int numTrials = atoi(argv[2]);
    int numEigenpairs = atoi(argv[3]);

    vector<SQUAREMATRIX_T> matrices;

    Solver solver(atof(argv[4]), atof(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]), argc >= 10 ? atoi(argv[9]) : time(NULL));

    /* This scans the directory containing the data set of matrices, and fetches the ones with the correct value of N */
    DIR* dir = opendir("../../../data/s4585711/matrices/");
    struct dirent* ent;
    while((ent = readdir(dir)) != NULL) {
        int scanN;
        int numElements;
        int id;
        int ret = sscanf(ent->d_name, "mat_%d_%d_%d", &scanN, &numElements, &id);

        if (ret != EOF && scanN == N && id < numTrials) {
            /* Once the correct file has been found, add it to the vector of matrices. */
            matrices.emplace_back(N, numElements);
            matrices.back().getFromFile("../../../data/s4585711/matrices/" + string(ent->d_name));
        }
    }
    closedir(dir);

    /* This keeps track of the total time to calculate the Eigenpairs so it can be averaged */
    long time = 0;
    for (int i = 0; i < numTrials; i++) {
        auto start = chrono::high_resolution_clock::now();
        Eigenpair* eigenpairs = solver.solveEigenpairs(&matrices.at(i), numEigenpairs);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(stop - start).count();
        
        time += duration;

        /* clear memory now we are done with these Eigenvectors */
        for (int i = 0; i < numEigenpairs; i++) {
            delete eigenpairs[i].eigenvector;
        }
        delete eigenpairs;

        cout << "Time: " << duration << endl;
    }

    cout << "Total time: " << time << endl;
    cout << "Average time: " << time / numTrials << endl;

    return 0;
}