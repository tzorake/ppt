#ifndef MATRIX_HELPER_S_HPP
#define MATRIX_HELPER_S_HPP

#include <vector>
#include <algorithm>
#include <random>
#include "matrix_single_threaded.hpp"

class MatrixHelper_S
{
public:
	static void setSeed(int seed)
	{
		s_seed = seed;
	}

	static MatrixD_S randomMatrix(int rows, int cols)
	{
		vector<double> result(rows * cols);

		std::mt19937 gen(s_seed);
		std::uniform_int_distribution<int> dis(0, 10);

		std::generate(result.begin(), result.end(), [&]() { return dis(gen); });

		return MatrixD_S(rows, cols, result);
	}

	static MatrixD_S randomNonZeroDeterminantMatrix(int size)
	{
		MatrixD_S A;
		double determinant;

		while (true)
		{
			A = MatrixHelper_S::randomMatrix(size, size);
			determinant = A.determinant();

			if (std::abs(determinant) > s_tolerance)
			{
				break;
			}
		}

		return A;
	}

private:
	static int s_seed;
	static constexpr auto s_tolerance = 1e-9;
};

int MatrixHelper_S::s_seed = 3;

#endif //MATRIX_HELPER_S_HPP