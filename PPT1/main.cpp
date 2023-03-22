#include <iostream>
#include "matrix.hpp"
#include "matrix_helper.hpp"

void firstProblem()
{
    MatrixD A = MatrixHelper::getRandomNonZeroDeterminantMatrix(5);
    MatrixD B = MatrixHelper::getRandomMatrix(5, 1);

    double det = A.determinant();
    MatrixD A_adj = A.adjugate();
    MatrixD A_inv = 1.0 / det * A_adj;
    MatrixD X = A_inv * B;

    std::cout << "A = \n" << A << std::endl;
    std::cout << "B = \n" << B << std::endl;
    std::cout << "X = \n" << X << std::endl;
    std::cout << "B = A * X = \n" << A * X << std::endl;
}

int main()
{
    firstProblem();

    return 0;
}