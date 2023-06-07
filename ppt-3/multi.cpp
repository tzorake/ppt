#include "mpi.h"
#include <iostream>
#include <string>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include "utilities.h"

double riemann_sum_double_integral(std::function<double(double, double)> f, double a, double b, double c, double d, int nx, int ny) {
    double hx = (b - a) / nx;
    double hy = (d - c) / ny;
    double integral = 0.0;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            double x = a + (i + 0.5) * hx;
            double y = c + (j + 0.5) * hy;

            integral += f(x, y) * hx * hy;
        }
    }

    return integral;
}

int main(int argc, char **argv)
{
	// for (int i = 0; i < argc; ++i) {
	// 	std::cout << argv[i] << std::endl;
	// }

	int rank, size;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int nx = 0; 
	int ny = 0;

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

			int N = atoi(argv[i + 1]);

			nx = N;
			ny = N;
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

	if (nx == 0 || ny == 0) {
		std::cerr << "Invalid arguments! Either nx or ny is zero or negative value." << std::endl;

		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (filename.empty()) {
		std::cerr << "Invalid arguments! Filename is empty string." << std::endl;

		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	// HANDLE ARGUMENTS END

    double a = 0.0;
    double b = 1.0;
    double c = 0.0;
    double d = 1.0;

    double hx = (b - a) / nx;
    double hy = (d - c) / ny;

    double A = 1.0;
    double B = 2.0;
    double C = 2.0;

    std::function<std::function<double(double, double)>(double, double, double)> F = [](double A, double B, double C) {
        return [A, B, C](double x, double y) {
            return A * (std::pow(x, B) + std::pow(y, C));
        };
    };

    // f(x, y) = A * (x^B + y^C)
    std::function<double(double, double)> f = F(A, B, C);

	double temp = 0.0;
	double integral = 0.0;
	double elapsed = 0;
	
	if (rank == 0) {
		elapsed = MPI_Wtime();
	}

	// BODY BEGIN

	/*      0   1   2   3   4   5
	 *    *---*---*---*---*---*---* --> nx
	 *  0 | 0 | 1 | 2 | 3 | 4 | 5 |
	 *    *---*---*---*---*---*---*
	 *  1 | 6 | 7 | 8 | 9 | 10| 11|
	 *    *---*---*---*---*---*---*
	 *  2 | 12| 13| 14| 15| 16| 17|
	 *    *---*---*---*---*---*---*
	 *  3 | 18| 19| 20| 21| 22| 23|
	 *    *---*---*---*---*---*---*
	 *  4 | 24| 25| 26| 27| 28| 29|
	 *    *---*---*---*---*---*---*
	 *    |
	 * ny v 
	 * 
	 *    |   1   |   2   |   3   |
	 *         split for '-n 3'
	*/

	int part = nx / size;
	int start = part * rank;
	int end = rank != size - 1? start + part : start + nx - start;

	int local_nx = end - start;
	double local_a = part*hx * rank;
	double local_b = part*hx * (rank + 1);
	integral += riemann_sum_double_integral(f, local_a, local_b, c, d, local_nx, ny);
	
	if (verbose) {
		std::cout << fmt::format("[{}] Start: {}, End: {}, a: {}, b: {}", rank, start, end, local_a, local_b) << std::endl 
				  << fmt::format("[{}] Integral: {}", rank, integral) << std::endl;
	}
	
	// BODY END

	if (rank == 0) {
		for (int i = 1; i < size; ++i) {
			MPI_Recv(&temp, 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &status);
			integral = integral + temp;
		}
		elapsed = MPI_Wtime() - elapsed;

		std::cout << fmt::format(
			"integrate ({}) dy from y={} to {} = {}",
			fmt::format("integrate ({} * (x^{} + y^{})) dx from x={} to {}", A, B, C, a, b),
			c, d, integral ) << std::endl
		<< "Elapsed time: " << elapsed  << std::endl;

	    FileSystem::writeFile(filename, std::to_string(elapsed));
	}
	else {
		MPI_Send(&integral, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}