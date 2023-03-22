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
	static MatrixD getRandomMatrix(size_t rows, size_t cols)
	{
		vector<double> result(rows * cols);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<int> dis(0, 10);

		std::generate(result.begin(), result.end(), [&]() { return dis(gen); });

		return MatrixD(rows, cols, result);
	}

	static MatrixD getRandomNonZeroDeterminantMatrix(size_t size)
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
};

#endif //MATRIX_HELPER_HPP