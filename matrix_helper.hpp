#ifndef MATRIX_HELPER_HPP
#define MATRIX_HELPER_HPP

#include "matrix.hpp"
#include <vector>
#include <algorithm>
#include <random>

constexpr auto TOLERANCE = 1e-9;

class MatrixHelper
{
public:
	static void setSeed(int seed)
	{
		s_seed = seed;
	}

	static MatrixD getRandomMatrix(int rows, int cols)
	{
		vector<double> result(rows * cols);

		std::mt19937 gen(s_seed);
		std::uniform_int_distribution<int> dis(0, 10);

		std::generate(result.begin(), result.end(), [&]() { return dis(gen); });

		return MatrixD(rows, cols, result);
	}

	static MatrixD getRandomNonZeroDeterminantMatrix(int size)
	{
		MatrixD A;
		double determinant;

		while (true)
		{
			A = MatrixHelper::getRandomMatrix(size, size);
			determinant = A.determinant();

			if (std::abs(determinant) > TOLERANCE)
			{
				break;
			}
		}

		return A;
	}

private:
	static int s_seed;
};

int MatrixHelper::s_seed = 3;

#endif //MATRIX_HELPER_HPP