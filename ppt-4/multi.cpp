#include <stdio.h>
#include <stdlib.h>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include "helper.h"
#include <mpi.h>

std::vector<double> multiply_matvec(std::vector<double> &matrix, std::vector<double> &vector)
{
    int cols = vector.size();
    int rows = matrix.size() / cols;

    std::vector<double> result(rows, 0.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[cols*i + j] * vector[j]; 
        }
    }

    return result;
}

int main(int argc, char **argv)
{
	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = 0;

	bool verbose = false;

	bool filenameFound = false;
	std::string filename;

	// HANDLE ARGUMENTS BEGIN

	for (int i = 0; i < argc; ++i) {
		if (strcmp(argv[i], "N") == 0) {
			if (i + 1 >= argc) {
				std::cerr << "Invalid arguments!" << std::endl;
				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			N = atoi(argv[i + 1]);
		}
		else if (strcmp(argv[i], "VERBOSE") == 0) {
			verbose = true;
		}
		else if (strcmp(argv[i], "FILENAME") == 0) {
			if (i + 1 >= argc) {
				std::cerr << "Invalid arguments!" << std::endl;

				MPI_Abort(MPI_COMM_WORLD, 1);
			}

			filename = argv[i + 1];

			filenameFound = true;
		}
	}

	if (N == 0) {
		std::cerr << "Invalid arguments! Either nx or ny is zero or negative value." << std::endl;

		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (filename.empty()) {
		std::cerr << "Invalid arguments! Filename is empty string." << std::endl;

		MPI_Abort(MPI_COMM_WORLD, 1);
	}

    std::vector<double> matrix;
    if (rank == 0) {
		Helper::set_seed(5);
        matrix = Helper::random(N*N, 0, 1);

		std::cout << fmt::format("[{}] Matrix:", rank) << std::endl;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < N; ++j) {
				std::cout << matrix[i*N + j] << " ";
			}
			std::cout << std::endl;
		}
    }

    std::vector<double> vector(N);
    if (rank == 0) {
        Helper::set_seed(5);
        vector = Helper::random(N, 1, 10);

        std::cout << fmt::format("[{}] Vector:", rank) << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << vector[i] << " ";
        }
        std::cout << std::endl;
    }

	double time = 0;

	if (rank == 0) {
		time = MPI_Wtime();
	}

    MPI_Bcast(vector.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<int> sendCounts(size);
    std::vector<int> displacements(size);
    int rowsPerProcess = N / size;
    int remainingRows = N % size;
    int displacement = 0;
    for (int i = 0; i < size; ++i) {
        sendCounts[i] = rowsPerProcess * N;
        if (i < remainingRows) {
            sendCounts[i] += N;
        }
        displacements[i] = displacement * N;
        displacement += sendCounts[i] / N;
    }

    int recvCount = (rank < remainingRows) ? (rowsPerProcess + 1) * N : rowsPerProcess * N;

    std::vector<double> localMatrix(recvCount);
    MPI_Scatterv(matrix.data(), sendCounts.data(), displacements.data(), MPI_DOUBLE,
                 localMatrix.data(), recvCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> recvBuffer = multiply_matvec(localMatrix, vector);

    std::vector<int> recvCounts(size);
    std::vector<int> displacementsResult(size);
    int rowsPerProcessRecv = N / size;
    int remainingRowsRecv = N % size;
    int displacementResult = 0;
    for (int i = 0; i < size; ++i) {
        recvCounts[i] = rowsPerProcessRecv;
        if (i < remainingRowsRecv) {
            recvCounts[i] += 1;
        }
        displacementsResult[i] = displacementResult;
        displacementResult += recvCounts[i];
    }

    int recvCountVec = (rank < remainingRowsRecv) ? (rowsPerProcessRecv + 1) : rowsPerProcessRecv;

    std::vector<double> globalResult(N);
    MPI_Gatherv(recvBuffer.data(), recvCountVec, MPI_DOUBLE, globalResult.data(), recvCounts.data(),
                displacementsResult.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
		time = MPI_Wtime() - time;

        std::cout << fmt::format("[{}] Result:", rank) << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << globalResult[i] << " ";
        }
        std::cout << std::endl;

		Helper::writeFile(filename, std::to_string(time));
    }

	MPI_Finalize();
    return 0;
}